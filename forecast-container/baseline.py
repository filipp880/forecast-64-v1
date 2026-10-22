#!/usr/bin/env python3
"""Бейзлайн-прогнозы (наивный и сезонный) — точка отсчёта для метрик ТЗ.

Считает те же пять метрик ТЗ (SMAPE/WAPE/RMSE/|Bias|/RMSLE) по тем же полным
срезам, что и ``evaluate.py``, но без модели — для простых «наивных» правил.
Смысл: модель обязана быть лучше этих чисел, иначе сложные признаки и обучение
не дают пользы. Сравнивайте: ``python evaluate.py`` против ``python baseline.py``.

Бейзлайны (прогноз на все ``--horizon`` дней одинаковый, кроме ``dow``):

- ``zero``   — всегда 0; ключевая точка отсчёта: по решению судей дни, где
  факт=0 и прогноз=0, входят в SMAPE как 0% (п. 5.5 ТЗ), поэтому нулевой
  прогноз — главный конкурент любой модели на этой метрике;
- ``last``   — последнее наблюдение ``quantity`` на срез (наивный прогноз);
- ``mean28`` — среднее за последние 28 дней (включая нулевые дни);
- ``mean90`` — среднее за последние 90 дней;
- ``dow``    — среднее по тому же дню недели за последние 28 дней (сезонный
  наивный); если по дню недели данных нет — подставляется ``mean28``.

История для прогноза берётся строго по ``cutoff`` включительно — как на
инференсе в API. Срезы и разбивка по горизонтам 1-7 / 8-28 / 29-64 — как в
``evaluate.py`` (полные срезы, шаг ``--cutoff-step``).

Примеры (из директории forecast-container/):
    python baseline.py                                  # все бейзлайны, 8 срезов
    python baseline.py --baseline mean28 --max-cutoffs 4
    python baseline.py --out baseline_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import HORIZON, KEY, LOOKBACK_DAYS, build_panel
from app.metrics import METRIC_NAMES, compute_all_metrics, format_metrics
from evaluate import DATA_KEY, HORIZON_BUCKETS, full_cutoffs

BASELINE_NAMES = ("zero", "last", "mean28", "mean90", "dow", "mean28_round", "mean90_round", "dow_round")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Бейзлайн-прогнозы: 5 метрик ТЗ по срезам истории")
    parser.add_argument(
        "--sales",
        default="../data/sales_dataset_hakaton_train.csv",
        help="CSV с продажами (формат sales_dataset_hakaton.csv)",
    )
    parser.add_argument(
        "--stock",
        default="../data/stock_dataset_hakaton_train.csv",
        help="CSV с остатками (формат stock_dataset_hakaton.csv)",
    )
    parser.add_argument(
        "--baseline",
        default="all",
        choices=(*BASELINE_NAMES, "all"),
        help="Какой бейзлайн считать (по умолчанию все)",
    )
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Горизонт прогноза, дней")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Глубина истории (как в train/evaluate)")
    parser.add_argument("--cutoff-step", type=int, default=28, help="Шаг между срезами, дней")
    parser.add_argument("--max-cutoffs", type=int, default=8, help="Сколько последних полных срезов оценить")
    parser.add_argument("--out", default=None, help="Путь для JSON-отчёта (опционально)")
    return parser.parse_args(argv)


def truth_for(panel: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Факт на горизонте 1..horizon после среза (панель полная — серии не теряются)."""
    mask = (panel["date"] > cutoff) & (panel["date"] <= cutoff + pd.Timedelta(days=horizon))
    truth = panel.loc[mask, [*DATA_KEY, "quantity"]].copy()
    truth["h"] = (truth["date"] - cutoff).dt.days
    return truth


def baseline_forecast(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    method: str,
    horizon: int,
    series_index: pd.DataFrame,
) -> pd.DataFrame:
    """Прогноз бейзлайна: серия × горизонт, посчитан строго по данным до среза."""
    if method.endswith("_round"):
        result = baseline_forecast(panel, cutoff, method[:-6], horizon, series_index)
        result["quantity"] = np.rint(result["quantity"])
        return result
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    grid = series_index.assign(_k=1).merge(
        pd.DataFrame({"date": dates, "_k": 1}), on="_k"
    ).drop(columns="_k")

    if method == "zero":
        grid["quantity"] = 0.0
        grid["h"] = (grid["date"] - cutoff).dt.days
        return grid[[*DATA_KEY, "h", "quantity"]]

    hist = panel[
        (panel["date"] <= cutoff) & (panel["date"] > cutoff - pd.Timedelta(days=90))
    ]
    win28 = hist[hist["date"] > cutoff - pd.Timedelta(days=28)]

    if method == "last":
        values = (
            hist[hist["date"] == cutoff]
            .groupby(KEY, sort=False)["quantity"]
            .last()
            .rename("quantity")
            .reset_index()
        )
        grid = grid.merge(values, on=KEY, how="left")
    elif method in ("mean28", "mean90"):
        days = 28 if method == "mean28" else 90
        window = hist[hist["date"] > cutoff - pd.Timedelta(days=days)]
        values = (
            window.groupby(KEY, sort=False)["quantity"]
            .mean()
            .rename("quantity")
            .reset_index()
        )
        grid = grid.merge(values, on=KEY, how="left")
    else:
        dow_stats = (
            win28.assign(dow=win28["date"].dt.dayofweek)
            .groupby(KEY + ["dow"], sort=False)["quantity"]
            .mean()
            .rename("quantity")
            .reset_index()
        )
        fallback = (
            win28.groupby(KEY, sort=False)["quantity"]
            .mean()
            .rename("fallback")
            .reset_index()
        )
        grid["dow"] = grid["date"].dt.dayofweek
        grid = grid.merge(dow_stats, on=KEY + ["dow"], how="left").merge(
            fallback, on=KEY, how="left"
        )
        grid["quantity"] = grid["quantity"].fillna(grid["fallback"])
        grid = grid.drop(columns="fallback")

    grid["quantity"] = grid["quantity"].fillna(0.0).clip(lower=0.0)
    grid["h"] = (grid["date"] - cutoff).dt.days
    return grid[[*DATA_KEY, "h", "quantity"]]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    methods = list(BASELINE_NAMES) if args.baseline == "all" else [args.baseline]

    print("Загрузка данных...", flush=True)
    sales = pd.read_csv(args.sales, parse_dates=["date"])
    stock = pd.read_csv(args.stock, parse_dates=["date"])
    panel = build_panel(sales, stock)
    series_index = panel[KEY].drop_duplicates().reset_index(drop=True)
    print(f"  панель: {len(panel):,} строк, {len(series_index):,} серий", flush=True)

    cutoffs = full_cutoffs(panel, args.lookback, args.horizon, args.cutoff_step, args.max_cutoffs)
    print(
        f"Оценка бейзлайнов {', '.join(methods)} по {len(cutoffs)} полным срезам "
        f"(шаг {args.cutoff_step} дн.):",
        flush=True,
    )

    per_cutoff: dict[str, list[dict]] = {method: [] for method in methods}
    pooled: dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame]]] = {
        method: ([], []) for method in methods
    }

    for cutoff in cutoffs:
        truth = truth_for(panel, cutoff, args.horizon)
        truth["origin"] = cutoff.strftime("%Y-%m-%d")
        for method in methods:
            forecast = baseline_forecast(panel, cutoff, method, args.horizon, series_index)
            forecast["origin"] = cutoff.strftime("%Y-%m-%d")
            metrics = compute_all_metrics(
                truth.drop(columns="h"), forecast.drop(columns="h")
            )
            print(f"  срез {cutoff.date()} | {method:<6} | {format_metrics(metrics)}", flush=True)
            per_cutoff[method].append(
                {"cutoff": str(cutoff.date()), "rows": int(len(truth)), "metrics": metrics}
            )
            truth["origin"] = cutoff.strftime("%Y-%m-%d")
            forecast["origin"] = cutoff.strftime("%Y-%m-%d")
            pooled[method][0].append(truth.copy())
            pooled[method][1].append(forecast)

        best = min(
            methods,
            key=lambda m: per_cutoff[m][-1]["metrics"]["smape"],
        )
        print(f"    лучший на срезе по SMAPE: {best}", flush=True)

    reports: dict[str, dict] = {}
    print("Среднее по срезам:", flush=True)
    for method in methods:
        mean_metrics = {
            name: float(np.mean([item["metrics"][name] for item in per_cutoff[method]]))
            for name in METRIC_NAMES
        }
        truth_all = pd.concat(pooled[method][0], ignore_index=True)
        pred_all = pd.concat(pooled[method][1], ignore_index=True)
        bucket_reports: list[dict] = []
        for low, high in HORIZON_BUCKETS:
            if low > args.horizon:
                continue
            high = min(high, args.horizon)
            mask = (truth_all["h"] >= low) & (truth_all["h"] <= high)
            metrics = compute_all_metrics(
                truth_all[mask].drop(columns="h"), pred_all[mask].drop(columns="h")
            )
            bucket_reports.append(
                {"h_low": low, "h_high": high, "rows": int(mask.sum()), "metrics": metrics}
            )
        reports[method] = {
            "per_cutoff": per_cutoff[method],
            "mean": mean_metrics,
            "horizon_buckets": bucket_reports,
        }
        print(f"  {method:<6} | {format_metrics(mean_metrics)}", flush=True)

    print("Среднее по горизонтам (все срезы):", flush=True)
    for method in methods:
        for bucket in reports[method]["horizon_buckets"]:
            print(
                f"  {method:<6} | горизонты {bucket['h_low']}-{bucket['h_high']} дн. "
                f"({bucket['rows']:,} строк) | {format_metrics(bucket['metrics'])}",
                flush=True,
            )

    if args.out:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cutoff_step": args.cutoff_step,
            "horizon": args.horizon,
            "lookback_days": args.lookback,
            "baselines": reports,
        }
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Отчёт сохранён: {args.out}")

    print(f"Готово за {time.time() - started:.0f} с.")
    print("Сравнение с моделью: python evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
