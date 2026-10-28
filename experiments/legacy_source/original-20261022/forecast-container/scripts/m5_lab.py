"""Лаборатория M5-идей: birth-фичи, иерархические энкодинги, ценовые прокси.

Три идеи из препроцессинга победителя Kaggle M5 (A1-Yeon), перенесённые на
наши данные; проверяются аддитивно к текущим 28 признакам на одинаковых
срезах/параметрах/seed (модели только в памяти, прод не трогается):

- ``birth``  — ``age_days``: дней с первой продажи серии (cap по lookback).
  Прецедент: у M5 «leading zeros» до release товара — не спрос. В наших
  данных 97.5% серий стартовали позже начала выгрузки (медиана 144 дня), у
  молодых серий окна ``mean_90``/``zero_share_90`` разбавлены дородовыми нулями.
- ``hier``   — иерархические mean-энкодинги (группа фич M5 после лагов):
  ``article_mean_90``, ``warehouse_mean_90``, ``cat1_mean_90`` и нормировки
  ``series_vs_*``. У нас все HISTORY-признаки внутрисерийные — кросс-серийных
  агрегатов нет.
- ``price``  — ценовые прокси из обязательных на инференсе ``sales_sum`` /
  ``discount_sum``: ``discount_share_28/90``, ``price_28_90``, ``price_7_28``
  (аналог price_norm / price_momentum).

Схема без утечки и разметка — как в ``feature_lab.py`` (7 обучающих срезов,
отбор, 8 тестовых). Порог продвижения тот же: индекс <= 0.997 и SMAPE не хуже.

Запуск (из директории forecast-container/, ~20-25 мин)::

    python scripts/m5_lab.py                  # отчёт logs/m5_lab.json
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
    _history,
    build_panel,
    make_training_frame,
)
from app.metrics import METRIC_NAMES, METRIC_WEIGHTS
from evaluate import full_cutoffs

from feature_lab import predict_metrics, train_model  # noqa: E402

BIRTH_FEATURES = ["age_days"]
HIER_FEATURES = [
    "article_mean_90",
    "warehouse_mean_90",
    "cat1_mean_90",
    "series_vs_article_28",
    "series_vs_warehouse_28",
]
PRICE_FEATURES = ["discount_share_28", "discount_share_90", "price_28_90", "price_7_28"]
M5_FEATURES = BIRTH_FEATURES + HIER_FEATURES + PRICE_FEATURES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лаборатория M5-идей: birth/hier/price")
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
    parser.add_argument("--out", default="logs/m5_lab.json")
    return parser.parse_args(argv)


def build_promo(sales: pd.DataFrame) -> pd.DataFrame:
    """Дневная таблица ``date × серия``: sales_sum / discount_sum / quantity продаж."""
    frame = sales[["date", *KEY, "quantity", "sales_sum", "discount_sum"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ("quantity", "sales_sum", "discount_sum"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame.groupby(["date", *KEY], as_index=False)[
        ["quantity", "sales_sum", "discount_sum"]
    ].sum()


def m5_series_features(
    panel: pd.DataFrame,
    promo: pd.DataFrame,
    cutoff: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """10 M5-признаков уровня серии, посчитанных строго по данным до ``cutoff``."""
    index = pd.MultiIndex.from_frame(panel[KEY].drop_duplicates())
    hist = _history(panel, cutoff, lookback)

    # birth: первая продажа внутри lookback-окна (на инференсе доступно ровно оно)
    first_sale = (
        hist.loc[hist["quantity"] > 0]
        .groupby(KEY, sort=False)["date"]
        .min()
        .reindex(index)
    )
    age = (cutoff - first_sale).dt.days.fillna(lookback).clip(upper=lookback)

    # иерархия: средние по 90-дневному окну через все серии группы
    w90 = _history(panel, cutoff, 90)
    w28 = _history(panel, cutoff, 28)
    mean_90 = w90.groupby(KEY, sort=False)["quantity"].mean().reindex(index).fillna(0.0)
    mean_28 = w28.groupby(KEY, sort=False)["quantity"].mean().reindex(index).fillna(0.0)
    attrs_df = panel[[*KEY, "category_lvl_1"]].drop_duplicates(subset=KEY)
    series90 = (
        mean_90.rename("m90").to_frame().join(attrs_df.set_index(KEY)).reset_index()
    )
    article_mean = series90.groupby("article")["m90"].mean()
    warehouse_mean = series90.groupby("warehouse_name")["m90"].mean()
    cat_mean = series90.groupby("category_lvl_1")["m90"].mean()

    keys = index.to_frame(index=False).merge(attrs_df, on=KEY, how="left")
    art = pd.Series(keys["article"]).map(article_mean).to_numpy()
    whs = pd.Series(keys["warehouse_name"]).map(warehouse_mean).to_numpy()
    cat = pd.Series(keys["category_lvl_1"]).map(cat_mean).to_numpy()
    mean28 = mean_28.to_numpy()

    # цены: только строки продаж внутри окна
    def window_stats(window: int) -> pd.DataFrame:
        win = promo[
            (promo["date"] > cutoff - pd.Timedelta(days=window)) & (promo["date"] <= cutoff)
        ]
        agg = (
            win.groupby(KEY, sort=False)[["quantity", "sales_sum", "discount_sum"]]
            .sum()
            .reindex(index)
            .fillna(0.0)
        )
        total = agg["sales_sum"] + agg["discount_sum"]
        price = agg["sales_sum"] / agg["quantity"].where(agg["quantity"] > 0, 1.0)
        discount_share = np.where(total > 0, agg["discount_sum"] / total.where(total > 0, 1.0), 0.0)
        return pd.DataFrame(
            {"price": price.where(agg["quantity"] > 0), "discount": discount_share},
            index=index,
        )

    s28 = window_stats(28)
    s7 = window_stats(7)
    s90 = window_stats(90)

    feats = pd.DataFrame(index=index)
    feats["age_days"] = age.to_numpy(dtype="float32")
    feats["article_mean_90"] = art.astype("float32")
    feats["warehouse_mean_90"] = whs.astype("float32")
    feats["cat1_mean_90"] = cat.astype("float32")
    feats["series_vs_article_28"] = (mean28 / (art + 1e-6)).astype("float32")
    feats["series_vs_warehouse_28"] = (mean28 / (whs + 1e-6)).astype("float32")
    feats["discount_share_28"] = s28["discount"].to_numpy(dtype="float32")
    feats["discount_share_90"] = s90["discount"].to_numpy(dtype="float32")
    feats["price_28_90"] = (
        (s28["price"] / s90["price"]).fillna(1.0).to_numpy(dtype="float32")
    )
    feats["price_7_28"] = (
        (s7["price"] / s28["price"]).fillna(1.0).to_numpy(dtype="float32")
    )
    return feats


def features_all(
    panel: pd.DataFrame,
    promo: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizons: list[int],
    lookback: int,
) -> pd.DataFrame:
    """``make_training_frame`` (признаки + таргет) + 10 M5-столбцов."""
    frame = make_training_frame(panel, cutoff, horizons, lookback)
    extras = m5_series_features(panel, promo, cutoff, lookback).reset_index()
    return frame.merge(extras, on=KEY, how="left")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    horizons = list(range(1, args.horizon + 1))

    print("Загрузка данных и сборка панели...", flush=True)
    sales = pd.read_csv(args.sales)
    stock = pd.read_csv(args.stock)
    panel = build_panel(sales, stock)
    promo = build_promo(sales)
    print(f"  панель: {len(panel):,} строк, {panel[KEY].drop_duplicates().shape[0]:,} серий; "
          f"promo-строк: {len(promo):,}", flush=True)

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
            features_all(panel, promo, cutoff, horizons, args.lookback) for cutoff in cutoffs
        ]
        frames[label] = pd.concat(parts, ignore_index=True)
        print(f"  {label}: {len(frames[label]):,} строк ({len(cutoffs)} срезов, "
              f"прошло {time.time() - started:.0f} с)", flush=True)
    test_frames = {
        cutoff: features_all(panel, promo, cutoff, horizons, args.lookback)
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

    variants = (
        ("base", list(FEATURE_COLUMNS)),
        ("birth", list(FEATURE_COLUMNS) + BIRTH_FEATURES),
        ("hier", list(FEATURE_COLUMNS) + HIER_FEATURES),
        ("price", list(FEATURE_COLUMNS) + PRICE_FEATURES),
        ("all", list(FEATURE_COLUMNS) + M5_FEATURES),
    )

    results: dict[str, dict] = {}
    models: dict[str, dict] = {}
    importances: dict[str, list[dict]] = {}
    for variant, columns in variants:
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
                {"feature": str(feature), "gain": float(value), "is_new": feature in M5_FEATURES}
                for feature, value in zip(columns, gain)
            ),
            key=lambda row: row["gain"],
            reverse=True,
        )
        results[variant] = {"per_cutoff": per_cutoff, "means": means}
        print("  среднее: " + " ".join(f"{name}={means[name]:.4f}" for name in METRIC_NAMES), flush=True)

    base_means = results["base"]["means"]
    verdicts: dict[str, dict] = {}
    print("\nСводка (индекс против base, порог <= 0.997 и SMAPE не хуже):")
    for variant, _ in variants:
        means = results[variant]["means"]
        index = float(
            sum(METRIC_WEIGHTS[name] * means[name] / base_means[name] for name in METRIC_NAMES)
        )
        smape_index = means["smape"] / base_means["smape"]
        promote = index <= 0.997 and smape_index <= 1.0
        verdicts[variant] = {"index": index, "smape_index": smape_index, "promote": promote}
        print(f"  {variant:<8} индекс {index:.4f}  SMAPE-отношение {smape_index:.4f}  "
              f"{'ПРОДВИГАТЬ' if promote else 'отклонить'}")

    print("\nТоп-10 по gain (all):")
    for row in importances["all"][:10]:
        mark = " *new" if row["is_new"] else ""
        print(f"  {row['feature']:<24} {row['gain']:.0f}{mark}")

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
            "birth_features": BIRTH_FEATURES,
            "hier_features": HIER_FEATURES,
            "price_features": PRICE_FEATURES,
            "params": params,
            "early_stopping": args.early_stopping,
        },
        "models": models,
        "results": results,
        "verdicts": verdicts,
        "importance_all": importances["all"][:25],
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт сохранён: {args.out} (готово за {time.time() - started:.0f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
