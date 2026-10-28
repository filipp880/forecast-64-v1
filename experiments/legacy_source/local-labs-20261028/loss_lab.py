"""Стенд функции потерь (docs/333, векторы 1-2) — веса строк, сток-маска, ES по SMAPE.

Аудит предлагает «обмануть» L2-лосс на 82% нулей: перевесить ненулевые строки
и строки регулярных серий (вектор 1), понизить вес строк «нет товара и нет
продаж» (вектор 2, цензура стокаутов) и останавливаться по SMAPE, а не RMSE.
Стенд переиспользует разметку ``scripts/feature_lab.py`` (те же 7 обучающих
срезов, отбор 2025-09-20, 8 тестовых срезов, те же params/seed) и обучает
варианты, отличающиеся только весами строк / eval-метрикой:

- ``base``        — без весов (референс, воспроизводит base из feature_lab);
- ``w_pos5``      — нули 1.0, плюсы 5.0;
- ``w_pos5_reg10``— плюсы 5.0, плюсы регулярных серий (``mean_28 >= 1``) 10.0;
- ``stock_mask``  — строки «stock целевого дня == 0 и факт == 0» получают 0.1;
- ``es_smape``    — без весов, early stopping по кастомному SMAPE (в исходном
  пространстве), а не по RMSE.

Порог продвижения тот же, что в feature_lab: композит <= 0.997 и SMAPE не хуже.

Запуск (из директории forecast-container/, ~25 мин)::

    python scripts/loss_lab.py                 # отчёт logs/loss_lab.json
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

import lightgbm as lgb

from app.features import CAT_FEATURES, FEATURE_COLUMNS, HORIZON, KEY, LOOKBACK_DAYS, build_panel
from app.metrics import METRIC_NAMES, METRIC_WEIGHTS
from app.models.base import transform_target
from evaluate import full_cutoffs
from feature_lab import TRANSFORM, encode_frame, features_with_extras, predict_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="docs/333: веса строк и eval-метрика")
    parser.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    parser.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--step", type=int, default=28)
    parser.add_argument("--max-cutoffs", type=int, default=8)
    parser.add_argument("--train-cutoffs", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="logs/loss_lab.json")
    return parser.parse_args(argv)


def smape_feval(y_true_log: np.ndarray, y_pred_log: np.ndarray):
    """SMAPE в исходном пространстве (глобальный) — метрика early stopping."""
    y = np.expm1(np.asarray(y_true_log, dtype="float64"))
    pred = np.expm1(np.asarray(y_pred_log, dtype="float64"))
    numerator = np.abs(y - pred)
    denominator = (np.abs(y) + np.abs(pred)) / 2.0
    ratio = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    return "smape", float(np.mean(ratio) * 100.0), False


def stock_out_weights(panel: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """0.1 для строк, где на целевой день stock == 0 и продажи == 0 (цензура сток-аутов)."""
    stock = panel[["date", *KEY, "stock"]]
    merged = frame[["date", *KEY]].merge(stock, on=["date", *KEY], how="left")
    target_stock = merged["stock"].to_numpy("float64")
    y = frame["target"].to_numpy("float64")
    mask = (target_stock == 0.0) & (y == 0.0)
    return np.where(mask, 0.1, 1.0)


def train_variant(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    columns: list[str],
    params: dict,
    early_stopping: int,
    sample_weight: np.ndarray | None = None,
    use_smape_eval: bool = False,
) -> tuple[object, dict[str, list[str]], dict]:
    categories: dict[str, list[str]] = {}
    model = lgb.LGBMRegressor(**params)
    started = time.time()
    y_train = transform_target(train_frame["target"].to_numpy("float64"), TRANSFORM)
    y_valid = transform_target(valid_frame["target"].to_numpy("float64"), TRANSFORM)
    model.fit(
        encode_frame(train_frame, columns, categories, fit=True),
        y_train,
        sample_weight=sample_weight,
        eval_X=encode_frame(valid_frame, columns, categories, fit=False),
        eval_y=y_valid,
        eval_metric=smape_feval if use_smape_eval else "rmse",
        categorical_feature=CAT_FEATURES,
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    info = {
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "seconds": round(time.time() - started, 1),
    }
    return model, categories, info


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    horizons = list(range(1, args.horizon + 1))

    print("Загрузка данных и сборка панели...", flush=True)
    sales = pd.read_csv(args.sales)
    stock = pd.read_csv(args.stock)
    panel = build_panel(sales, stock)
    print(f"  панель: {len(panel):,} строк", flush=True)

    test_cutoffs = full_cutoffs(panel, args.lookback, args.horizon, args.step, args.max_cutoffs)
    from app.temporal import research_split
    train_cutoffs, valid_cutoff = research_split(
        test_cutoffs, args.horizon, args.step, args.train_cutoffs
    )
    print(f"Обучение: {len(train_cutoffs)} срезов {train_cutoffs[-1].date()}..{train_cutoffs[0].date()}, "
          f"отбор: {valid_cutoff.date()}", flush=True)

    train_frame = pd.concat(
        [features_with_extras(panel, cutoff, horizons, args.lookback) for cutoff in train_cutoffs],
        ignore_index=True,
    )
    valid_frame = features_with_extras(panel, valid_cutoff, horizons, args.lookback)
    print(f"  train {len(train_frame):,}, valid {len(valid_frame):,} "
          f"(прошло {time.time() - started:.0f} с)", flush=True)
    test_frames = {
        cutoff: features_with_extras(panel, cutoff, horizons, args.lookback)
        for cutoff in test_cutoffs
    }

    y_train = train_frame["target"].to_numpy("float64")
    positive = y_train > 0.0
    mean_28 = train_frame["mean_28"].to_numpy("float64")

    weights = {
        "base": None,
        "w_pos5": np.where(positive, 5.0, 1.0),
        "w_pos5_reg10": np.where(positive & (mean_28 >= 1.0), 10.0, np.where(positive, 5.0, 1.0)),
        "stock_mask": stock_out_weights(panel, train_frame),
        "es_smape": None,
    }
    for label, values in weights.items():
        if values is None:
            continue
        print(f"  веса {label}: средний {values.mean():.3f}, "
              f"масса на плюсах {values[positive].sum() / values.sum():.1%}", flush=True)

    params = {
        "n_estimators": args.iterations,
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
    for label, values in weights.items():
        print(f"\n=== Вариант {label} ===", flush=True)
        model, categories, info = train_variant(
            train_frame,
            valid_frame,
            columns,
            params,
            args.early_stopping,
            sample_weight=values,
            use_smape_eval=(label == "es_smape"),
        )
        models[label] = info
        print(f"  обучение {info['seconds']} с, best_iteration={info['best_iteration']}", flush=True)
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
        results[label] = {"per_cutoff": per_cutoff, "means": means}
        print("  среднее: " + " ".join(f"{name}={means[name]:.4f}" for name in METRIC_NAMES), flush=True)

    base_means = results["base"]["means"]
    verdicts = {}
    print("\nСводка (индекс против base, порог <= 0.997 и SMAPE не хуже):")
    for label in weights:
        means = results[label]["means"]
        index = float(sum(METRIC_WEIGHTS[name] * means[name] / base_means[name] for name in METRIC_NAMES))
        smape_ratio = means["smape"] / base_means["smape"]
        promote = index <= 0.997 and smape_ratio <= 1.0
        verdicts[label] = {"index": index, "smape_ratio": smape_ratio, "promote": promote}
        print(f"  {label:<14} индекс {index:.4f}  SMAPE-отношение {smape_ratio:.4f}  "
              f"{'ПРОДВИГАТЬ' if promote else 'отклонить'}")

    report = {
        "meta": {
            "sales": args.sales,
            "stock": args.stock,
            "train_cutoffs": [str(cutoff.date()) for cutoff in train_cutoffs],
            "valid_cutoff": str(valid_cutoff.date()),
            "test_cutoffs": [str(cutoff.date()) for cutoff in test_cutoffs],
            "params": params,
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
