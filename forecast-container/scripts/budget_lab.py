"""Лаборатория бюджета деревьев: base-модель при лимите 1500 vs 5000 итераций.

Все предыдущие лаборатории сравнивали варианты на равном бюджете (1500
деревьев), и каждый раз обучение упиралось в лимит без срабатывания early
stopping — мы ни разу не наблюдали плато кривой качества. Этот прогон отвечает
на единственный оставшийся вопрос: выигрывает ли модель от большего бюджета
(и где вообще останавливается честный ES по валидации).

Условия — как в ``feature_lab.py``/``m5_lab.py`` (те же 7 обучающих срезов,
отбор, 8 тестовых, те же параметры и seed; отличается только ``n_estimators``).
Порог продвижения: индекс <= 0.997 и SMAPE не хуже.

Запуск (из директории forecast-container/, ~15 мин)::

    python scripts/budget_lab.py                  # отчёт logs/budget_lab.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import (
    FEATURE_COLUMNS,
    HORIZON,
    KEY,
    LOOKBACK_DAYS,
    build_panel,
    make_training_frame,
)
from app.metrics import METRIC_NAMES, METRIC_WEIGHTS
from evaluate import full_cutoffs

from feature_lab import predict_metrics, train_model  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лаборатория бюджета деревьев: 1500 vs 5000")
    parser.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    parser.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--step", type=int, default=28, help="Шаг между срезами, дней")
    parser.add_argument("--max-cutoffs", type=int, default=8, help="Срезов бэктеста")
    parser.add_argument("--train-cutoffs", type=int, default=7, help="Обучающих срезов перед окном")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1500, 5000])
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="logs/budget_lab.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    horizons = list(range(1, args.horizon + 1))

    print("Загрузка данных и сборка панели...", flush=True)
    sales = pd.read_csv(args.sales)
    stock = pd.read_csv(args.stock)
    panel = build_panel(sales, stock)
    print(f"  панель: {len(panel):,} строк, {panel[KEY].drop_duplicates().shape[0]:,} серий",
          flush=True)

    test_cutoffs = full_cutoffs(panel, args.lookback, args.horizon, args.step, args.max_cutoffs)
    valid_cutoff = test_cutoffs[0] - pd.Timedelta(days=args.step)
    train_cutoffs = [
        test_cutoffs[0] - pd.Timedelta(days=args.step * k)
        for k in range(2, args.train_cutoffs + 2)
    ]
    print(f"Обучение: {len(train_cutoffs)} срезов {train_cutoffs[-1].date()}..{train_cutoffs[0].date()}, "
          f"отбор: {valid_cutoff.date()}, тест: {len(test_cutoffs)} срезов "
          f"{test_cutoffs[0].date()}..{test_cutoffs[-1].date()}", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    for label, cutoffs in (("train", train_cutoffs), ("valid", [valid_cutoff])):
        parts = [
            make_training_frame(panel, cutoff, horizons, args.lookback) for cutoff in cutoffs
        ]
        frames[label] = pd.concat(parts, ignore_index=True)
        print(f"  {label}: {len(frames[label]):,} строк ({len(cutoffs)} срезов, "
              f"прошло {time.time() - started:.0f} с)", flush=True)
    test_frames = {
        cutoff: make_training_frame(panel, cutoff, horizons, args.lookback)
        for cutoff in test_cutoffs
    }
    print(f"  test: {len(test_cutoffs)} срезов (прошло {time.time() - started:.0f} с)", flush=True)

    params = {
        "num_leaves": 127,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 8.0,
        "random_state": args.seed,
        "n_jobs": args.threads,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    columns = list(FEATURE_COLUMNS)

    results: dict[str, dict] = {}
    models: dict[str, dict] = {}
    for budget in args.budgets:
        variant = f"b{budget}"
        print(f"\n=== Вариант {variant}: n_estimators={budget} ===", flush=True)
        variant_params = {**params, "n_estimators": budget}
        model, categories, info = train_model(
            frames["train"], frames["valid"], columns, variant_params, args.early_stopping
        )
        stopped = info["best_iteration"] < budget
        models[variant] = {**info, "budget": budget, "es_triggered": stopped}
        print(f"  обучение {info['seconds']} с, best_iteration={info['best_iteration']}"
              f"{' (ES сработал)' if stopped else ' (упёрлись в лимит)'}", flush=True)

        per_cutoff: dict[str, dict] = {}
        for cutoff, part in test_frames.items():
            metrics = predict_metrics(model, categories, part, columns)
            per_cutoff[str(cutoff.date())] = metrics
            print(f"  [{cutoff.date()}] SMAPE={metrics['smape']:.2f} WAPE={metrics['wape']:.2f} "
                  f"RMSE={metrics['rmse']:.4f} Bias={metrics['bias']:.2f} "
                  f"RMSLE={metrics['rmsle']:.4f}", flush=True)

        means = {
            name: float(np.mean([row[name] for row in per_cutoff.values()]))
            for name in METRIC_NAMES
        }
        results[variant] = {"per_cutoff": per_cutoff, "means": means}
        print("  среднее: " + " ".join(f"{name}={means[name]:.4f}" for name in METRIC_NAMES),
              flush=True)

    base_variant = f"b{args.budgets[0]}"
    base_means = results[base_variant]["means"]
    verdicts: dict[str, dict] = {}
    print(f"\nСводка (индекс против {base_variant}, порог <= 0.997 и SMAPE не хуже):")
    for budget in args.budgets:
        variant = f"b{budget}"
        means = results[variant]["means"]
        index = float(
            sum(METRIC_WEIGHTS[name] * means[name] / base_means[name] for name in METRIC_NAMES)
        )
        smape_index = means["smape"] / base_means["smape"]
        promote = index <= 0.997 and smape_index <= 1.0
        verdicts[variant] = {"index": index, "smape_index": smape_index, "promote": promote}
        stop = f"останов на {models[variant]['best_iteration']}" if models[variant]["es_triggered"] \
            else f"лимит {budget}"
        print(f"  {variant:<8} индекс {index:.4f}  SMAPE-отношение {smape_index:.4f}  "
              f"({stop})  {'ПРОДВИГАТЬ' if promote else 'отклонить'}")

    report = {
        "meta": {
            "sales": args.sales,
            "stock": args.stock,
            "lookback_days": args.lookback,
            "horizon": args.horizon,
            "step": args.step,
            "train_cutoffs": [str(cutoff.date()) for cutoff in train_cutoffs],
            "valid_cutoff": str(valid_cutoff.date()),
            "test_cutoffs": [str(cutoff.date()) for cutoff in test_cutoffs],
            "features": columns,
            "params": params,
            "budgets": args.budgets,
            "early_stopping": args.early_stopping,
        },
        "models": models,
        "results": results,
        "verdicts": verdicts,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт сохранён: {args.out} (готово за {time.time() - started:.0f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
