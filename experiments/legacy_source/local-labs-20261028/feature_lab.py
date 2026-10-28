"""Лаборатория признаков (доп. проверка после Очереди 3): базовые 23 фичи vs +5 recency.

Прогон учебного исследования вокруг единственного неисследованного рычага
точности: набора признаков. Сравниваются две LightGBM-модели на ОДНИХ и тех же
срезах/cutoff-ах/параметрах/seed — отличается только набор столбцов:

- ``base``  — текущие ``FEATURE_COLUMNS`` (23 признака);
- ``ext``   — они же + 5 recency-признаков: ``mean_3``, ``zero_share_7``,
  ``n_sale_days_28``, ``max_28``, ``trend_7_28``.

Схема без утечки: обучение — 7 полных срезов перед отборочным, early
stopping — на последнем срезе перед окном бэктеста (отложенном от обучения),
оценка — на 8 полных срезах бэктеста (тех же, что в ``evaluate.py``/лабораториях
постобработки). Модели живут только в памяти: лаборатория не трогает
``model/`` и артефакты прода. Признаки считаются по строго той же схеме, что
в проде (``app.features``, lookback 180 дней), поэтому при выигрыше набор
переносится в ``app/features.py`` и переобучается полный бленд.

Порог продвижения (зафиксирован заранее): композитный индекс ТЗ ``ext``
против ``base`` <= 0.997 (выигрыш >= 0.3%) И SMAPE не хуже базы; иначе —
отрицательный результат в README, скрипт и отчёт коммитятся как есть.

Запуск (из директории forecast-container/, ~15 мин)::

    python scripts/feature_lab.py                 # отчёт logs/feature_lab.json
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

from app.features import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    HORIZON,
    KEY,
    LOOKBACK_DAYS,
    _history,
    build_panel,
    make_training_frame,
)
from app.metrics import METRIC_NAMES, METRIC_WEIGHTS, bias_abs, rmsle, rmse, smape, wape
from app.models.base import inverse_target, postprocess_forecast, transform_target
from evaluate import full_cutoffs

EXTRA_FEATURES = ["mean_3", "zero_share_7", "n_sale_days_28", "max_28", "trend_7_28"]
TRANSFORM = "log1p"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Признаковая лаборатория: base 23 vs +5 recency")
    parser.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    parser.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--step", type=int, default=28, help="Шаг между срезами, дней")
    parser.add_argument("--max-cutoffs", type=int, default=8, help="Срезов бэктеста")
    parser.add_argument("--train-cutoffs", type=int, default=7, help="Обучающих срезов перед окном")
    parser.add_argument("--iterations", type=int, default=1500, help="Верхняя граница деревьев")
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="logs/feature_lab.json")
    return parser.parse_args(argv)


def extra_series_features(panel: pd.DataFrame, cutoff: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """5 recency-признаков уровня серии, посчитанных строго до ``cutoff`` включительно."""
    index = pd.MultiIndex.from_frame(panel[KEY].drop_duplicates())
    win3 = _history(panel, cutoff, 3)
    win7 = _history(panel, cutoff, 7)
    win28 = _history(panel, cutoff, 28)

    mean_3 = win3.groupby(KEY, sort=False)["quantity"].mean().reindex(index).fillna(0.0)
    mean_7 = win7.groupby(KEY, sort=False)["quantity"].mean().reindex(index).fillna(0.0)
    mean_28 = win28.groupby(KEY, sort=False)["quantity"].mean().reindex(index).fillna(0.0)

    flags7 = win7[[*KEY, "quantity"]].assign(_f=(win7["quantity"] == 0).astype("float32"))
    flags28 = win28[[*KEY, "quantity"]].assign(_f=(win28["quantity"] > 0).astype("float32"))

    feats = pd.DataFrame(index=index)
    feats["mean_3"] = mean_3
    feats["zero_share_7"] = (
        flags7.groupby(KEY, sort=False)["_f"].mean().reindex(index).fillna(1.0)
    )
    feats["n_sale_days_28"] = (
        flags28.groupby(KEY, sort=False)["_f"].sum().reindex(index).fillna(0.0)
    )
    feats["max_28"] = win28.groupby(KEY, sort=False)["quantity"].max().reindex(index).fillna(0.0)
    feats["trend_7_28"] = mean_7 / (mean_28 + 1e-6)
    return feats.astype("float32")


def features_with_extras(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizons: list[int],
    lookback: int,
) -> pd.DataFrame:
    """``make_training_frame`` (признаки + таргет) + recency-столбцы."""
    frame = make_training_frame(panel, cutoff, horizons, lookback)
    extras = extra_series_features(panel, cutoff, lookback).reset_index()
    return frame.merge(extras, on=KEY, how="left")


def encode_frame(
    frame: pd.DataFrame,
    columns: list[str],
    categories: dict[str, list[str]],
    fit: bool,
) -> pd.DataFrame:
    """Кодировка категорий по прод-схеме: фиксация на обучении, восстановление на прогнозе."""
    encoded = frame[columns].copy()
    for col in CAT_FEATURES:
        values = encoded[col].astype(str)
        if fit:
            categories[col] = sorted(values.unique())
        encoded[col] = pd.Categorical(values, categories=categories[col])
    return encoded


def train_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    columns: list[str],
    params: dict,
    early_stopping: int,
) -> tuple[object, dict[str, list[str]], dict]:
    """LightGBM по правилам прод-класса (категории-строки, log1p-таргет, ES по отбору)."""
    categories: dict[str, list[str]] = {}
    model = lgb.LGBMRegressor(**params)
    started = time.time()
    model.fit(
        encode_frame(train_frame, columns, categories, fit=True),
        transform_target(train_frame["target"].to_numpy("float64"), TRANSFORM),
        eval_X=encode_frame(valid_frame, columns, categories, fit=False),
        eval_y=transform_target(valid_frame["target"].to_numpy("float64"), TRANSFORM),
        eval_metric="rmse",
        categorical_feature=CAT_FEATURES,
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    info = {
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "seconds": round(time.time() - started, 1),
    }
    return model, categories, info


def predict_metrics(
    model,
    categories: dict[str, list[str]],
    part: pd.DataFrame,
    columns: list[str],
) -> dict:
    raw = model.predict(
        encode_frame(part, columns, categories, fit=False),
        num_iteration=model.best_iteration_,
    )
    pred = postprocess_forecast(inverse_target(raw, TRANSFORM), 0.0, True)
    y = part["target"].to_numpy("float64")
    series = (part["warehouse_name"].astype(str) + "|" + part["article"].astype(str)).to_numpy()
    return {
        "smape": smape(y, pred, series),
        "wape": wape(y, pred),
        "rmse": rmse(y, pred),
        "bias": bias_abs(y, pred, series),
        "rmsle": rmsle(y, pred),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    horizons = list(range(1, args.horizon + 1))

    print("Загрузка данных и сборка панели...", flush=True)
    sales = pd.read_csv(args.sales)
    stock = pd.read_csv(args.stock)
    panel = build_panel(sales, stock)
    print(f"  панель: {len(panel):,} строк, {panel['date'].nunique()} дней, "
          f"{panel[KEY].drop_duplicates().shape[0]:,} серий", flush=True)

    test_cutoffs = full_cutoffs(panel, args.lookback, args.horizon, args.step, args.max_cutoffs)
    from app.temporal import research_split
    train_cutoffs, valid_cutoff = research_split(
        test_cutoffs, args.horizon, args.step, args.train_cutoffs
    )
    print(f"Обучение: {len(train_cutoffs)} срезов {train_cutoffs[-1].date()}..{train_cutoffs[0].date()}, "
          f"отбор: {valid_cutoff.date()}, тест: {len(test_cutoffs)} срезов "
          f"{test_cutoffs[0].date()}..{test_cutoffs[-1].date()}", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    for label, cutoffs in (("train", train_cutoffs), ("valid", [valid_cutoff])):
        parts = [
            features_with_extras(panel, cutoff, horizons, args.lookback) for cutoff in cutoffs
        ]
        frames[label] = pd.concat(parts, ignore_index=True)
        print(f"  {label}: {len(frames[label]):,} строк ({len(cutoffs)} срезов, "
              f"прошло {time.time() - started:.0f} с)", flush=True)
    test_frames = {
        cutoff: features_with_extras(panel, cutoff, horizons, args.lookback)
        for cutoff in test_cutoffs
    }
    print(f"  test: {len(test_cutoffs)} срезов по "
          f"{len(next(iter(test_frames.values()))):,} строк (прошло {time.time() - started:.0f} с)",
          flush=True)

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

    results: dict[str, dict] = {}
    models: dict[str, dict] = {}
    importances: dict[str, list[dict]] = {}
    for variant, columns in (
        ("base", list(FEATURE_COLUMNS)),
        ("ext", list(FEATURE_COLUMNS) + EXTRA_FEATURES),
    ):
        print(f"\n=== Вариант {variant}: {len(columns)} признаков ===", flush=True)
        model, categories, info = train_model(
            frames["train"], frames["valid"], columns, params, args.early_stopping
        )
        models[variant] = info
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
        gain = model.booster_.feature_importance(importance_type="gain")
        importances[variant] = sorted(
            (
                {"feature": str(feature), "gain": float(value), "is_new": feature in EXTRA_FEATURES}
                for feature, value in zip(columns, gain)
            ),
            key=lambda row: row["gain"],
            reverse=True,
        )
        results[variant] = {"per_cutoff": per_cutoff, "means": means}
        print("  среднее: " + " ".join(f"{name}={means[name]:.4f}" for name in METRIC_NAMES), flush=True)

    base_means = results["base"]["means"]
    ext_means = results["ext"]["means"]
    index = float(sum(METRIC_WEIGHTS[name] * ext_means[name] / base_means[name] for name in METRIC_NAMES))
    smape_index = ext_means["smape"] / base_means["smape"]
    promote = index <= 0.997 and smape_index <= 1.0

    print(f"\nКомпозитный индекс ext/base: {index:.4f} (SMAPE-отношение {smape_index:.4f})")
    print("Топ-10 по gain (ext):")
    for row in importances["ext"][:10]:
        mark = " *new" if row["is_new"] else ""
        print(f"  {row['feature']:<22} {row['gain']:.0f}{mark}")
    print(f"\nВердикт (порог <= 0.997 и SMAPE не хуже): "
          f"{'ПРОДВИГАТЬ' if promote else 'отклонить'}", flush=True)

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
            "base_features": list(FEATURE_COLUMNS),
            "extra_features": EXTRA_FEATURES,
            "params": params,
            "early_stopping": args.early_stopping,
        },
        "models": models,
        "results": results,
        "composite_index": index,
        "smape_index": smape_index,
        "promote": promote,
        "importance_ext": importances["ext"][:25],
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт сохранён: {args.out} (готово за {time.time() - started:.0f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
