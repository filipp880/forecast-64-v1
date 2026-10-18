#!/usr/bin/env python3
"""Бэктест обученной модели: пять метрик ТЗ по нескольким прошлым срезам.

Для каждого среза признаки считаются только по данным «до среза» (ровно как
на инференсе), прогноз сравнивается с фактом. Вывод: метрики по каждому срезу,
среднее по срезам и разбивка по горизонтам 1-7 / 8-28 / 29-64 (как
F1/F2/F3 в плане решения).

Постобработка прогноза (порог обнуления и округление до целых) берётся из
``config.json`` → ``postprocess`` и переопределяется флагами ``--zero-threshold``
/ ``--round-int``. ``--sweep-thresholds`` считает метрики для набора порогов
(с округлением и без) на уже посчитанных прогнозах — подбор порога без
переобучения.

Учитываются только «полные» срезы — те, до конца данных от которых хватает
ровно ``--horizon`` дней; иначе горизонт обрезан и метрики несравнимы.

Важно: если модель обучена на всей истории (``train.py`` без обрезки данных),
исторические срезы могли попадать в её обучение — метрики будут оптимистичными.
Для честной вневыборочной оценки обучите модель на данных, обрезанных до срезов
оценки (уберите последние ``--max-cutoffs * --cutoff-step + horizon`` дней продаж),
и прогоните ``evaluate.py`` на этой модели.

Примеры (из директории forecast-container/):
    python evaluate.py                              # model/, шаг 28 дней, 8 последних срезов
    python evaluate.py --max-cutoffs 4 --cutoff-step 56
    python evaluate.py --out report.json            # JSON-отчёт
    python evaluate.py --sweep-thresholds 0,0.25,0.5,1   # подобрать порог обнуления
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

from app.features import HORIZON, LOOKBACK_DAYS, build_panel, make_features
from app.metrics import METRIC_NAMES, compute_all_metrics, format_metrics
from app.models import get_model_class, inverse_target, postprocess_forecast
from app.models.base import format_duration

HORIZON_BUCKETS = ((1, 7), (8, 28), (29, 64))
DATA_KEY = ["date", "warehouse_name", "article"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Бэктест модели: 5 метрик ТЗ по срезам истории")
    parser.add_argument("--model-dir", default="model", help="Директория с config.json и файлом модели")
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
    parser.add_argument("--cutoff-step", type=int, default=28, help="Шаг между срезами, дней")
    parser.add_argument("--max-cutoffs", type=int, default=8, help="Сколько последних полных срезов оценить")
    parser.add_argument("--out", default=None, help="Путь для JSON-отчёта (опционально)")
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=None,
        help="Обнулять прогнозы ниже порога (по умолчанию — из config.json -> postprocess)",
    )
    parser.add_argument(
        "--round-int",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Округлять прогноз до целых (по умолчанию — из config.json -> postprocess)",
    )
    parser.add_argument(
        "--sweep-thresholds",
        default=None,
        help="Список порогов через запятую (например '0,0.25,0.5,1'): метрики для каждого, с округлением и без",
    )
    return parser.parse_args(argv)


def full_cutoffs(
    panel: pd.DataFrame,
    lookback_days: int,
    horizon: int,
    step_days: int,
    max_cutoffs: int,
) -> list[pd.Timestamp]:
    """Последние ``max_cutoffs`` срезов, до конца данных от которых есть весь горизонт."""
    first = panel["date"].min()
    last = panel["date"].max()
    start = first + pd.Timedelta(days=lookback_days)
    end = last - pd.Timedelta(days=1)
    if start > end:
        raise ValueError(f"Истории не хватает: нужно минимум {lookback_days + horizon} дней данных")
    cutoffs = [
        cutoff
        for cutoff in pd.date_range(start, end, freq=f"{step_days}D")
        if cutoff + pd.Timedelta(days=horizon) <= last
    ]
    if not cutoffs:
        raise ValueError(
            f"Нет ни одного полного среза: шаг {step_days} дн. и горизонт {horizon} дн. "
            f"не помещаются в данные ({first.date()} .. {last.date()})"
        )
    return cutoffs[-max_cutoffs:]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Ошибка: не найден конфиг модели: {config_path}", file=sys.stderr)
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model_type = str(config.get("model_type", "lgbm"))
    model_cls = get_model_class(model_type)
    model_path = model_dir / (config.get("model_file") or f"model{model_cls.file_suffix}")
    if not model_path.exists():
        print(f"Ошибка: модель не найдена: {model_path}. Сначала обучите: python train.py", file=sys.stderr)
        return 2
    model = model_cls.load(model_path)

    horizon = int(config.get("horizon", HORIZON))
    lookback = int(config.get("lookback_days", LOOKBACK_DAYS))
    transform = str(config.get("target_transform", "log1p"))
    print(f"Модель: {model_type} из {model_dir} (горизонт {horizon}, lookback {lookback})")

    post = config.get("postprocess") or {}
    zero_threshold = (
        float(args.zero_threshold)
        if args.zero_threshold is not None
        else float(post.get("zero_threshold", 0.0))
    )
    round_int = (
        bool(args.round_int)
        if args.round_int is not None
        else bool(post.get("round_int", False))
    )
    print(
        f"Постобработка: порог обнуления {zero_threshold:g}, "
        f"округление до целых: {'вкл' if round_int else 'выкл'}"
    )

    print("Загрузка данных...", flush=True)
    sales = pd.read_csv(args.sales, parse_dates=["date"])
    stock = pd.read_csv(args.stock, parse_dates=["date"])
    panel = build_panel(sales, stock)
    truth_source = panel[[*DATA_KEY, "quantity"]]

    cutoffs = full_cutoffs(panel, lookback, horizon, args.cutoff_step, args.max_cutoffs)
    print(f"Оценка по {len(cutoffs)} полным срезам (шаг {args.cutoff_step} дн.):", flush=True)

    per_cutoff: list[dict] = []
    pooled_truth: list[pd.DataFrame] = []
    pooled_pred: list[pd.DataFrame] = []
    raw_by_cutoff: list[tuple[pd.DataFrame, np.ndarray]] = []
    started_eval = time.time()
    for index, cutoff in enumerate(cutoffs, start=1):
        features = make_features(panel, cutoff, range(1, horizon + 1), lookback)
        raw = inverse_target(model.predict(features), transform)
        predictions = postprocess_forecast(raw, zero_threshold, round_int)

        truth = features[DATA_KEY].merge(truth_source, on=DATA_KEY, how="left")
        truth["h"] = features["h"].to_numpy()
        forecast = features[DATA_KEY].copy()
        forecast["quantity"] = predictions
        forecast["h"] = features["h"].to_numpy()
        raw_by_cutoff.append((truth.drop(columns="h"), raw))

        metrics = compute_all_metrics(truth.drop(columns="h"), forecast.drop(columns="h"))
        elapsed = time.time() - started_eval
        eta = elapsed / index * (len(cutoffs) - index)
        print(
            f"  [{index}/{len(cutoffs)}] срез {cutoff.date()} ({len(features):,} строк, "
            f"осталось ~{format_duration(eta)}): {format_metrics(metrics)}",
            flush=True,
        )
        per_cutoff.append({"cutoff": str(cutoff.date()), "rows": int(len(features)), "metrics": metrics})
        pooled_truth.append(truth)
        pooled_pred.append(forecast)

    mean_metrics = {
        name: float(np.mean([item["metrics"][name] for item in per_cutoff])) for name in METRIC_NAMES
    }
    print(f"  среднее по {len(cutoffs)} срезам: {format_metrics(mean_metrics)}", flush=True)

    truth_all = pd.concat(pooled_truth, ignore_index=True)
    pred_all = pd.concat(pooled_pred, ignore_index=True)
    bucket_reports: list[dict] = []
    for low, high in HORIZON_BUCKETS:
        if low > horizon:
            continue
        high = min(high, horizon)
        mask = (truth_all["h"] >= low) & (truth_all["h"] <= high)
        metrics = compute_all_metrics(
            truth_all[mask].drop(columns="h"), pred_all[mask].drop(columns="h")
        )
        bucket_reports.append({"h_low": low, "h_high": high, "rows": int(mask.sum()), "metrics": metrics})
        print(f"  горизонты {low}-{high} дн. ({int(mask.sum()):,} строк): {format_metrics(metrics)}", flush=True)

    sweep_report: list[dict] = []
    if args.sweep_thresholds:
        try:
            sweep_values = [
                float(value) for value in str(args.sweep_thresholds).split(",") if value.strip()
            ]
        except ValueError:
            print("Ошибка: --sweep-thresholds — числа через запятую, например '0,0.25,0.5,1'", file=sys.stderr)
            return 2
        print(f"Свип порога обнуления (среднее по {len(cutoffs)} срезам):", flush=True)
        for tau in sweep_values:
            entry: dict = {"zero_threshold": tau}
            for label, rnd in (("without_rounding", False), ("with_rounding", True)):
                means = {name: [] for name in METRIC_NAMES}
                for truth_frame, raw in raw_by_cutoff:
                    forecast = truth_frame[DATA_KEY].copy()
                    forecast["quantity"] = postprocess_forecast(raw, tau, rnd)
                    metrics = compute_all_metrics(truth_frame, forecast)
                    for name in METRIC_NAMES:
                        means[name].append(metrics[name])
                entry[label] = {name: float(np.mean(values)) for name, values in means.items()}
            sweep_report.append(entry)
            print(f"  порог={tau:<6g} без округл.: {format_metrics(entry['without_rounding'])}", flush=True)
            print(f"  {'':<12} с округл.:   {format_metrics(entry['with_rounding'])}", flush=True)

    if args.out:
        report = {
            "model_type": model_type,
            "model_dir": str(model_dir),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cutoff_step": args.cutoff_step,
            "horizon": horizon,
            "lookback_days": lookback,
            "postprocess": {"zero_threshold": zero_threshold, "round_int": round_int},
            "per_cutoff": per_cutoff,
            "mean": mean_metrics,
            "horizon_buckets": bucket_reports,
            "threshold_sweep": sweep_report,
        }
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Отчёт сохранён: {args.out}")

    print(f"Готово за {time.time() - started:.0f} с.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
