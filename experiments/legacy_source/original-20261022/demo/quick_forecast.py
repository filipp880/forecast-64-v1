#!/usr/bin/env python3
"""Быстрое визуальное демо решения ML-задачи из forecast-64-v1.

Запуск (из корня репозитория):

    py -3.10 demo/quick_forecast.py

Что делает один прогон (~1-2 минуты на CPU):

1. читает train-данные организаторов (продажи + остатки);
2. для скорости берёт выборку серий «склад x артикул» (по умолчанию 800 из 8777);
3. строит панель и признаки общим пайплайном ``app.features`` из
   forecast-container (тот же код, что у прод-модели: lookback 180 дней);
4. обучает LightGBM на ``log1p(quantity)`` со строгим holdout по времени:
   срезы на обучение -> окно отбора для early stopping -> тестовые 64 дня;
5. считает 5 метрик ТЗ (SMAPE 50%, WAPE 20%, RMSE 15%, |Bias| 10%, RMSLE 5%)
   функциями ``app.metrics`` и сравнивает с наивными бейзлайнами zero/mean_28;
6. рисует 3 наглядные картинки (EDA, качество модели, примеры прогнозов);
7. дообучается на всех срезах и сохраняет прогноз на будущие 64 дня.

Артефакты — в ``demo/out/``. Полное прод-решение (blend hurdle+lgbm, FastAPI,
бэктесты, лаборатории) — в ``forecast-container/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Переиспользуем готовый пайплайн решения, а не копируем его -------------
REPO_DIR = Path(__file__).resolve().parents[1]
CONTAINER_DIR = REPO_DIR / "forecast-container"
if str(CONTAINER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTAINER_DIR))

import lightgbm as lgb  # noqa: E402

from app.features import (  # noqa: E402
    CAT_FEATURES,
    FEATURE_COLUMNS,
    HORIZON,
    KEY,
    LOOKBACK_DAYS,
    build_panel,
    make_features,
    make_training_frame,
)
from app.metrics import METRIC_WEIGHTS, compute_all_metrics  # noqa: E402

STEP = 28  # шаг сетки срезов обучения, как в train.py (--cutoff-step)
ZERO_THRESHOLD = 0.5  # гейт обнуления из прод-постобработки
CORAL = "#e4572e"
BLUE = "#2f6db3"
GREEN = "#2e8b57"
GRAY = "#8a8a8a"


def log(message: str) -> None:
    print(message, flush=True)


def human(seconds: float) -> str:
    return f"{seconds:.0f} c" if seconds < 90 else f"{seconds / 60:.1f} мин"


# ---------------------------------------------------------------------------
# 1. Данные и выборка серий
# ---------------------------------------------------------------------------


def read_data(sales_path: Path, stock_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(
        sales_path,
        parse_dates=["date"],
        dtype={"quantity": "float32", "sales_sum": "float32", "discount_sum": "float32"},
    )
    stock = pd.read_csv(stock_path, parse_dates=["date"])
    if "char_article" in stock.columns and "article" not in stock.columns:
        stock = stock.rename(columns={"char_article": "article"})
    stock["stock"] = pd.to_numeric(stock["stock"], errors="coerce").clip(lower=0.0)
    return sales, stock


def select_series(
    sales: pd.DataFrame, last_date: pd.Timestamp, max_series: int, seed: int
) -> list[tuple[str, str]]:
    """Случайная выборка серий, репрезентативная по всей популяции серий.

    Все серии с продажами за последние 180 дней нумеруются и перемешиваются
    ГПСЧ с фиксированным сидом — метрики демо должны быть сопоставимы с полным
    прогоном (а не смещены в сторону «топовых» активных серий).
    """
    recent = sales[sales["date"] >= last_date - pd.Timedelta(days=180)]
    pairs = [tuple(row) for row in recent[KEY].drop_duplicates().to_numpy()]
    if max_series >= len(pairs):
        return pairs

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pairs), size=max_series, replace=False))
    return [pairs[i] for i in idx]


def filter_pairs(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    pairs_df = pd.DataFrame(pairs, columns=KEY)
    return frame.merge(pairs_df, on=KEY, how="inner")


# ---------------------------------------------------------------------------
# 2. Срезы: обучение -> окно отбора -> тест (строгий holdout, как в train.py)
# ---------------------------------------------------------------------------


def make_cutoffs(
    panel: pd.DataFrame, n_train: int, horizon: int, lookback: int, step: int
) -> tuple[list[pd.Timestamp], pd.Timestamp, pd.Timestamp]:
    first = panel["date"].min()
    last = panel["date"].max()
    test_cutoff = last - pd.Timedelta(days=horizon)
    select_cutoff = test_cutoff - pd.Timedelta(days=horizon)

    start = first + pd.Timedelta(days=lookback)
    end = select_cutoff - pd.Timedelta(days=horizon)
    if start > end:
        raise ValueError("Недостаточно истории: нужно минимум lookback + 2 * horizon дней")

    grid = list(pd.date_range(start, end, freq=f"{step}D"))
    if not grid:
        raise ValueError("Не удалось построить сетку обучающих срезов")
    return grid[-n_train:], select_cutoff, test_cutoff


def category_dtypes(panel: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    dtypes: dict[str, pd.CategoricalDtype] = {}
    for column in CAT_FEATURES:
        values = panel[column].fillna("na").astype(str).unique()
        dtypes[column] = pd.CategoricalDtype(categories=sorted(set(values)))
    return dtypes


def encode_categories(frame: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    frame = frame.copy()
    for column, dtype in dtypes.items():
        frame[column] = frame[column].fillna("na").astype(str).astype(dtype)
    return frame


# ---------------------------------------------------------------------------
# 3. Модель и постобработка
# ---------------------------------------------------------------------------


def lgbm_params(args: argparse.Namespace, n_estimators: int) -> dict:
    """Параметры LightGBM — как в прод-модели forecast-container (leaves 127, lambda 8)."""
    return dict(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=8.0,
        deterministic=True,
        force_col_wise=True,
        random_state=args.seed,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbose=-1,
    )


def train_model(
    train: pd.DataFrame, select: pd.DataFrame, args: argparse.Namespace
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**lgbm_params(args, args.n_estimators))
    model.fit(
        train[FEATURE_COLUMNS],
        np.log1p(train["target"].to_numpy(dtype="float64")),
        eval_X=select[FEATURE_COLUMNS],
        eval_y=np.log1p(select["target"].to_numpy("float64")),
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def raw_predictions(model: lgb.LGBMRegressor, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(model.predict(frame[FEATURE_COLUMNS])), 0.0, None)


def postprocess(raw: np.ndarray, zero_threshold: float, round_int: bool) -> np.ndarray:
    out = raw.copy()
    if zero_threshold and zero_threshold > 0:
        out = np.where(out < zero_threshold, 0.0, out)
    if round_int:
        out = np.rint(out)
    return out


def metrics_for(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    truth = frame[["date", *KEY, "target"]].rename(columns={"target": "quantity"})
    forecast = frame[["date", *KEY]].copy()
    forecast["quantity"] = predictions
    return compute_all_metrics(truth, forecast)


def metrics_by_horizon(frame: pd.DataFrame, predictions: np.ndarray) -> list[dict]:
    buckets = [(1, 7), (8, 28), (29, HORIZON)]
    rows = []
    for low, high in buckets:
        mask = frame["h"].between(low, high).to_numpy()
        metrics = metrics_for(frame[mask], predictions[mask])
        rows.append({"bucket": f"{low}-{high}", **metrics})
    return rows


# ---------------------------------------------------------------------------
# 4. Графики
# ---------------------------------------------------------------------------


def plot_eda(panel: pd.DataFrame, out_dir: Path) -> Path:
    daily = panel.groupby("date", as_index=False)["quantity"].sum()

    last_day = panel["date"].max()
    recent = panel[panel["date"] > last_day - pd.Timedelta(days=90)]
    zero_share = recent.groupby(KEY)["quantity"].apply(lambda s: float((s == 0).mean()))
    weekday = recent.assign(dow=recent["date"].dt.dayofweek).groupby("dow")["quantity"].mean()
    top_articles = (
        panel.groupby("article")["quantity"].sum().sort_values(ascending=False).head(10)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    fig.suptitle(
        "EDA: выборка серий из train-данных организаторов", fontsize=14, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.plot(daily["date"], daily["quantity"], color=BLUE, linewidth=0.8, alpha=0.5, label="день")
    ax.plot(
        daily["date"],
        daily["quantity"].rolling(7, min_periods=1).mean(),
        color=CORAL,
        linewidth=1.6,
        label="7-дневное среднее",
    )
    ax.set_title("Суммарные продажи по дням (выборка)")
    ax.set_ylabel("quantity, шт/день")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    ax.hist(zero_share.to_numpy(), bins=30, color=BLUE, edgecolor="white")
    ax.set_title("Доля «нулевых» дней по сериям (последние 90 дней)")
    ax.set_xlabel("доля дней без продаж")
    ax.set_ylabel("число серий")
    ax.axvline(
        float(zero_share.mean()),
        color=CORAL,
        linestyle="--",
        label=f"среднее {zero_share.mean():.0%}",
    )
    ax.legend()

    ax = axes[1, 0]
    names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    ax.bar(names, weekday.to_numpy(), color=[GREEN if d < 5 else CORAL for d in range(7)])
    ax.set_title("Профиль спроса по дням недели (последние 90 дней)")
    ax.set_ylabel("средние продажи, шт/день")

    ax = axes[1, 1]
    y = np.arange(len(top_articles))
    ax.barh(y, top_articles.to_numpy(), color=BLUE)
    ax.set_yticks(y, [str(a) for a in top_articles.index])
    ax.invert_yaxis()
    ax.set_title("Топ-10 артикулов по объёму продаж")
    ax.set_xlabel("quantity, шт за всю историю выборки")

    fig.tight_layout()
    path = out_dir / "01_eda.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_quality(
    importance: pd.Series,
    horizon_metrics: list[dict],
    baseline_metrics: dict[str, dict],
    test: pd.DataFrame,
    forecast: np.ndarray,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle("Качество модели на честных тестовых 64 днях", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    top = importance.sort_values(ascending=False).head(15)
    y = np.arange(len(top))
    ax.barh(y, top.to_numpy() / top.sum(), color=BLUE)
    ax.set_yticks(y, top.index)
    ax.invert_yaxis()
    ax.set_title("Важность признаков (gain, топ-15)")
    ax.set_xlabel("доля в суммарном gain")

    ax = axes[0, 1]
    x = np.arange(len(horizon_metrics))
    model_smape = [row["smape"] for row in horizon_metrics]
    naiv_smape = [row["naive_smape"] for row in horizon_metrics]
    ax.bar(x - 0.2, model_smape, width=0.4, color=BLUE, label="модель")
    ax.bar(x + 0.2, naiv_smape, width=0.4, color=GRAY, label="mean_28 (наивный)")
    ax.set_xticks(x, [row["bucket"] for row in horizon_metrics])
    ax.set_title("SMAPE по горизонтам, %")
    ax.set_xlabel("горизонт, дней")
    ax.legend()

    ax = axes[1, 0]
    sample = np.random.default_rng(0).choice(
        len(test), size=min(20000, len(test)), replace=False
    )
    ax.scatter(
        test["target"].to_numpy()[sample],
        forecast[sample],
        s=6,
        alpha=0.15,
        color=BLUE,
    )
    limit = max(1.0, float(np.percentile(test["target"].to_numpy()[sample], 99.9)))
    ax.plot([0, limit], [0, limit], color=CORAL, linestyle="--", linewidth=1)
    ax.set_xlim(-0.5, limit)
    ax.set_ylim(-0.5, limit)
    ax.set_title("Факт vs прогноз (тест, точка = серия x день)")
    ax.set_xlabel("факт, шт")
    ax.set_ylabel("прогноз, шт")

    ax = axes[1, 1]
    variants = ["zero", "mean_28", "model"]
    labels = {"zero": "нулевой", "mean_28": "mean_28", "model": "LightGBM"}
    values = [baseline_metrics[v]["smape"] for v in variants]
    ax.bar([labels[v] for v in variants], values, color=[GRAY, GREEN, CORAL])
    for i, value in enumerate(values):
        ax.text(i, value + 0.4, f"{value:.1f}", ha="center", fontsize=10)
    ax.set_title("SMAPE: модель против наивных бейзлайнов, % (меньше — лучше)")
    ax.set_ylabel("SMAPE, %")

    fig.tight_layout()
    path = out_dir / "02_quality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_examples(
    panel: pd.DataFrame,
    test: pd.DataFrame,
    forecast: np.ndarray,
    test_cutoff: pd.Timestamp,
    out_dir: Path,
) -> Path:
    totals = test.groupby(KEY, observed=True)["target"].sum().sort_values(ascending=False)
    positive = totals[totals > 0]
    picks: list[tuple[str, str]] = []
    if len(positive) >= 3:
        picks = [
            positive.index[0],
            positive.index[len(positive) // 2],
            positive.index[-1],
        ]

    fig, axes = plt.subplots(1, max(1, len(picks)), figsize=(15, 4.2), squeeze=False)
    fig.suptitle(
        f"Примеры прогноза: история 120 дней + тест 64 дня (отсечка {test_cutoff.date()})",
        fontsize=13,
        fontweight="bold",
    )

    for ax, pair in zip(axes[0], picks):
        series = panel[
            (panel["warehouse_name"] == pair[0]) & (panel["article"] == pair[1])
        ].sort_values("date")
        history = series[series["date"] > test_cutoff - pd.Timedelta(days=120)]
        pairs_mask = (test["warehouse_name"] == pair[0]) & (test["article"] == pair[1])
        pair_test = test[pairs_mask].sort_values("h")
        pair_forecast = forecast[pairs_mask.to_numpy()]

        ax.plot(history["date"], history["quantity"], color=GRAY, linewidth=1.0, label="история")
        ax.plot(pair_test["date"], pair_test["target"], color=BLUE, linewidth=1.4, label="факт (тест)")
        ax.plot(
            pair_test["date"], pair_forecast, color=CORAL, linewidth=1.4, label="прогноз"
        )
        ax.axvline(test_cutoff, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"{pair[1]}\n{pair[0]}", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "03_forecast_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. CLI и main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Быстрое визуальное демо прогноза на 64 дня (LightGBM + 5 метрик ТЗ)"
    )
    parser.add_argument(
        "--sales",
        type=Path,
        default=REPO_DIR / "data" / "sales_dataset_hakaton_train.csv",
        help="CSV продаж организаторов",
    )
    parser.add_argument(
        "--stock",
        type=Path,
        default=REPO_DIR / "data" / "stock_dataset_hakaton_train.csv",
        help="CSV остатков организаторов",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent / "out"
    )
    parser.add_argument("--max-series", type=int, default=800, help="Сколько серий взять")
    parser.add_argument("--train-cutoffs", type=int, default=10, help="Обучающих срезов")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-round", action="store_true", help="Не округлять прогноз")
    return parser.parse_args()


def main() -> None:
    try:  # консоль Windows: чтобы русский текст не ломался при перенаправлении вывода
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - не критично
        pass

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for path in (args.sales, args.stock):
        if not path.exists():
            raise SystemExit(f"Нет файла: {path}")

    log("=" * 78)
    log("Быстрый визуальный демо-пайплайн: прогноз quantity на 64 дня")
    log("=" * 78)

    t0 = time.time()
    sales, stock = read_data(args.sales, args.stock)
    log(f"[1/7] Данные прочитаны: продажи {len(sales):,} строк, остатки {len(stock):,}")

    last_date = sales["date"].max()
    pairs = select_series(sales, last_date, args.max_series, args.seed)
    log(
        f"[2/7] Выбрано серий: {len(pairs)} из "
        f"{sales.groupby(KEY).ngroups} (случайная репрезентативная выборка), "
        f"{human(time.time() - t0)}"
    )

    select_preview = last_date - pd.Timedelta(days=2 * HORIZON)
    min_date = select_preview - pd.Timedelta(
        days=LOOKBACK_DAYS + STEP * (args.train_cutoffs + 2)
    )
    sales_sample = filter_pairs(sales[sales["date"] >= min_date], pairs)
    stock_sample = filter_pairs(stock[stock["date"] >= min_date], pairs)
    del sales, stock

    panel = build_panel(sales_sample, stock_sample)
    del sales_sample, stock_sample
    log(
        f"[3/7] Панель собрана: {len(panel):,} строк, "
        f"{panel['date'].min().date()} .. {panel['date'].max().date()}, "
        f"{human(time.time() - t0)}"
    )

    train_cutoffs, select_cutoff, test_cutoff = make_cutoffs(
        panel, args.train_cutoffs, HORIZON, LOOKBACK_DAYS, STEP
    )
    horizons = range(1, HORIZON + 1)
    train_frames = [
        make_training_frame(panel, cutoff, horizons, LOOKBACK_DAYS)
        for cutoff in train_cutoffs
    ]
    select_frame = make_training_frame(panel, select_cutoff, horizons, LOOKBACK_DAYS)
    test_frame = make_training_frame(panel, test_cutoff, horizons, LOOKBACK_DAYS)

    dtypes = category_dtypes(panel)
    train = encode_categories(pd.concat(train_frames, ignore_index=True), dtypes)
    select = encode_categories(select_frame, dtypes)
    test = encode_categories(test_frame, dtypes)
    log(
        f"[4/7] Признаки готовы: обучение {len(train):,} строк "
        f"({len(train_cutoffs)} срезов), отбор {len(select):,}, тест {len(test):,}"
    )
    log(
        f"       срез отбора {select_cutoff.date()}, тест {test_cutoff.date() + pd.Timedelta(days=1)}"
        f"..{last_date.date()} (honest holdout)"
    )

    model = train_model(train, select, args)
    best_iteration = int(model.best_iteration_ or args.n_estimators)
    log(f"[5/7] LightGBM обучен: early stopping на {best_iteration} деревьях")

    test_raw = raw_predictions(model, test)
    test_forecast = postprocess(test_raw, ZERO_THRESHOLD, not args.no_round)

    model_metrics = metrics_for(test, test_forecast)
    raw_metrics = metrics_for(test, test_raw)
    zero_metrics = metrics_for(test, np.zeros(len(test)))
    mean28_raw = test["mean_28"].to_numpy(dtype="float64")
    mean28_metrics = metrics_for(
        test, postprocess(mean28_raw, ZERO_THRESHOLD, not args.no_round)
    )

    horizon_metrics = metrics_by_horizon(test, test_forecast)
    naive_by_horizon = metrics_by_horizon(test, postprocess(mean28_raw, ZERO_THRESHOLD, True))
    for row, naive in zip(horizon_metrics, naive_by_horizon):
        row["naive_smape"] = naive["smape"]

    log("[6/7] Метрики на тестовых 64 днях (5 метрик ТЗ):")
    header = f"    {'вариант':<26}{'SMAPE':>8}{'WAPE':>8}{'RMSE':>8}{'|Bias|':>8}{'RMSLE':>8}"
    log(header)
    log("    " + "-" * 66)
    for name, metrics in (
        ("LightGBM + гейт 0.5 + round", model_metrics),
        ("LightGBM без постобработки", raw_metrics),
        ("наивный mean_28", mean28_metrics),
        ("нулевой прогноз", zero_metrics),
    ):
        log(
            f"    {name:<26}{metrics['smape']:>8.2f}{metrics['wape']:>8.2f}"
            f"{metrics['rmse']:>8.3f}{metrics['bias']:>8.2f}{metrics['rmsle']:>8.4f}"
        )
    log(
        f"    веса ТЗ: "
        + ", ".join(f"{key} {value:.0%}" for key, value in METRIC_WEIGHTS.items())
    )

    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=FEATURE_COLUMNS
    )
    log("    топ-5 признаков по gain: " + ", ".join(importance.sort_values(ascending=False).head(5).index))

    eda_path = plot_eda(panel, out_dir)
    quality_path = plot_quality(
        importance,
        horizon_metrics,
        {"zero": zero_metrics, "mean_28": mean28_metrics, "model": model_metrics},
        test,
        test_forecast,
        out_dir,
    )
    examples_path = plot_examples(panel, test, test_forecast, test_cutoff, out_dir)

    # Финальный refit на всех срезах (включая тест) и прогноз на будущие 64 дня
    t0 = time.time()
    all_frames = train_frames + [select_frame, test_frame]
    all_rows = encode_categories(pd.concat(all_frames, ignore_index=True), dtypes)
    final_model = lgb.LGBMRegressor(**lgbm_params(args, best_iteration))
    final_model.fit(
        all_rows[FEATURE_COLUMNS], np.log1p(all_rows["target"].to_numpy(dtype="float64"))
    )

    future = encode_categories(
        make_features(panel, last_date, horizons, LOOKBACK_DAYS), dtypes
    )
    future_raw = raw_predictions(final_model, future)
    future_forecast = postprocess(future_raw, ZERO_THRESHOLD, not args.no_round)

    test_export = test[["date", *KEY, "h", "target"]].rename(columns={"target": "actual"})
    test_export["forecast"] = test_forecast
    test_export["forecast_raw"] = test_raw
    test_export.to_csv(out_dir / "test_predictions_64d.csv", index=False)

    future_export = future[["date", *KEY, "h"]].copy()
    future_export["quantity"] = future_forecast
    future_export.to_csv(out_dir / "future_forecast_64d.csv", index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": {
            "series_in_sample": len(pairs),
            "series_total": int(panel[KEY].drop_duplicates().shape[0]),
            "panel_rows": int(len(panel)),
            "date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        },
        "holdout": {
            "train_cutoffs": [str(c.date()) for c in train_cutoffs],
            "select_cutoff": str(select_cutoff.date()),
            "test_cutoff": str(test_cutoff.date()),
            "horizon": HORIZON,
            "lookback_days": LOOKBACK_DAYS,
        },
        "model": {
            "type": "LGBMRegressor (log1p target)",
            "best_iteration": best_iteration,
            "params": {
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_estimators": args.n_estimators,
            },
        },
        "postprocess": {"zero_threshold": ZERO_THRESHOLD, "round_int": not args.no_round},
        "metrics": {
            "model": {k: round(v, 4) for k, v in model_metrics.items()},
            "model_no_postprocess": {k: round(v, 4) for k, v in raw_metrics.items()},
            "baseline_mean_28": {k: round(v, 4) for k, v in mean28_metrics.items()},
            "baseline_zero": {k: round(v, 4) for k, v in zero_metrics.items()},
        },
        "metrics_by_horizon": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in horizon_metrics
        ],
        "top_features_gain": {
            str(name): round(float(value), 1)
            for name, value in importance.sort_values(ascending=False).head(10).items()
        },
        "outputs": [p.name for p in (eda_path, quality_path, examples_path)],
        "duration_sec": round(time.time() - t_start, 1),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    log(f"[7/7] Артефакты сохранены в {out_dir}:")
    log(f"      - {eda_path.name}, {quality_path.name}, {examples_path.name}")
    log(
        f"      - metrics.json, test_predictions_64d.csv ({len(test_export):,} строк), "
        f"future_forecast_64d.csv ({len(future_export):,} строк)"
    )
    log(
        f"Готово за {human(time.time() - t_start)}. "
        f"Прод-решение со blend-моделью и API — в forecast-container/."
    )


if __name__ == "__main__":
    main()
