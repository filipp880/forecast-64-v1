#!/usr/bin/env python3
"""Describe weekday transfer of a frozen raw model; never select or refit it.

Seven consecutive Monday--Sunday input dates share one future nine-week
truth window. Their errors are correlated and are not independent folds.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.calibration import forecast_quantities
from app.features import (
    COMPETITION_CUTOFF, FORECAST_WEEKS, KEY, build_panel, forecast_end,
    make_features, next_week_start, truncate_history,
)
from app.metrics import compute_all_metrics
from app.predictor import Predictor
from app.submission import normalized_pairs, validate_forecast
from app.validators import validate_sales_csv, validate_stock_csv
from baseline import truth_for


def normalize_origins(origins):
    """Require one complete calendar week of explicit midnight as-of dates."""
    dates = [pd.Timestamp(value) for value in origins]
    if len(dates) != 7 or any(pd.isna(value) or value.tzinfo is not None or value != value.normalize() for value in dates):
        raise ValueError("Provide seven valid timezone-naive calendar dates")
    expected = list(pd.date_range(dates[0], periods=7, freq="D"))
    if dates != expected or dates[0].weekday() != 0:
        raise ValueError("Origins must be seven consecutive dates from Monday through Sunday")
    return dates


def evaluate_weekdays(predictor, sales, stock, origins, *, history_cutoff=COMPETITION_CUTOFF):
    """Evaluate an unchanged artifact against the same pair/week truth keys.

    Pair coverage is fixed to pairs known at the earliest origin. Each feature
    panel is rebuilt from that origin's past-only input, matching API data
    availability. The sparse daily truth table contains all recorded sales;
    ``truth_for`` adds zero-sales keys, not missing predictions.
    """
    origins = normalize_origins(origins)
    cutoff = pd.Timestamp(history_cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize() or cutoff > COMPETITION_CUTOFF:
        raise ValueError("History cutoff must be a calendar date no later than 2026-07-26")
    if not predictor.is_loaded:
        raise ValueError("Load the frozen raw artifact before evaluation")
    if predictor.config.get("calibration"):
        raise ValueError("Use a raw artifact without selection/decision calibration")
    predictor.assert_history_compatible(origins[0])
    configuration_before = json.dumps(predictor.config, sort_keys=True)
    sales, stock = truncate_history(sales, stock, cutoff)
    if sales.empty:
        raise ValueError("No sales remain within the authorized history")
    sales["date"] = pd.to_datetime(sales.date)
    stock["date"] = pd.to_datetime(stock.date)
    last_day = forecast_end(origins[0], FORECAST_WEEKS)
    if last_day > cutoff or last_day > sales.date.max():
        raise ValueError("All nine truth weeks must include their final Sunday in available history")
    if any(next_week_start(origin) != next_week_start(origins[0]) for origin in origins):
        raise ValueError("Weekday comparison requires an identical future truth window")
    common_pairs = normalized_pairs(sales.loc[sales.date <= origins[0], KEY])
    if common_pairs.empty:
        raise ValueError("No pairs were known at the earliest origin")
    # Truth needs daily quantities only; a second dense stock/category panel
    # would waste memory and does not change truth_for's exact weekly totals.
    daily_truth = sales.groupby(["date", *KEY], observed=True, as_index=False).quantity.sum()
    truth = truth_for(daily_truth, origins[0], FORECAST_WEEKS, common_pairs)
    first_week = next_week_start(origins[0])
    post = predictor.config.get("postprocess", {})
    records = {}
    for origin in origins:
        started = time.perf_counter()
        past_sales, past_stock = truncate_history(sales, stock, origin)
        if past_sales.empty or pd.to_datetime(past_sales.date).max() != origin:
            raise ValueError(f"No sale is recorded on {origin.date()}; API max(sales.date) would use a different origin")
        errors = validate_sales_csv(past_sales) + validate_stock_csv(past_stock)
        if errors:
            raise ValueError(f"API input at {origin.date()}: {'; '.join(errors)}")
        panel = build_panel(past_sales, past_stock, history_end=origin)
        features = make_features(
            panel, origin, lookback_days=int(predictor.config.get("lookback_days", 180)),
            required_pairs=common_pairs,
        )
        prediction = features[["week", *KEY]].copy()
        prediction["quantity"] = forecast_quantities(
            predictor.model, features, float(post.get("zero_threshold", 0.0)),
            bool(post.get("round_int", False)), calibration=None,
        )
        validate_forecast(prediction, common_pairs, first_week)
        metrics = compute_all_metrics(truth, prediction)
        total_actual, total_prediction = float(truth.quantity.sum()), float(prediction.quantity.sum())
        records[str(origin.date())] = {
            "weekday": origin.day_name(), "first_week": str(first_week.date()),
            "last_day": str(last_day.date()), "rows": len(prediction),
            "pairs": len(common_pairs), "history_sales_rows": len(past_sales),
            "history_stock_rows": len(past_stock), "max_input_sales_date": str(origin.date()),
            "metrics": metrics, "actual_total": total_actual, "prediction_total": total_prediction,
            "volume_bias": (total_prediction / total_actual - 1) * 100 if total_actual else None,
            "seconds": round(time.perf_counter() - started, 3),
        }
        del panel, features, prediction, past_sales, past_stock
        gc.collect()
    if json.dumps(predictor.config, sort_keys=True) != configuration_before:
        raise RuntimeError("The frozen artifact configuration changed during evaluation")
    return {
        "evaluation_kind": "descriptive_weekday_transfer", "granularity": "week",
        "forecast_weeks": FORECAST_WEEKS, "history_cutoff": str(cutoff.date()),
        "model_training_data_end": predictor.config["training_data_end"],
        "model_selection_data_end": predictor.config["selection_data_end"],
        "model_artifact_sha256": predictor.config.get("model_artifact_sha256"),
        "common_first_week": str(first_week.date()), "common_last_day": str(last_day.date()),
        "pair_policy": "fixed_observed_pairs_known_at_earliest_origin", "pairs": len(common_pairs),
        "truth_windows_overlap": True, "calibration_used": False,
        "refit_performed": False, "selection_performed": False, "promotion_performed": False,
        "note": "All seven origins use the same nine-week truth window and pair catalogue. Errors are correlated; this descriptive check is not seven independent folds and must not select a candidate or promote an artifact.",
        "by_origin": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, default=PROJECT.parent / "data/sales_dataset_hakaton_train.csv")
    parser.add_argument("--stock", type=Path, default=PROJECT.parent / "data/stock_dataset_hakaton_train.csv")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--origins", nargs="+", default=pd.date_range("2026-02-16", periods=7, freq="D").strftime("%Y-%m-%d").tolist())
    parser.add_argument("--history-cutoff", default="2026-07-26")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {args.out}")
    predictor = Predictor(args.model_dir).load()
    dtypes = {"warehouse_name": "string", "article": "string", "char_article": "string"}
    sales = pd.read_csv(args.sales, parse_dates=["date"], dtype=dtypes)
    stock = pd.read_csv(args.stock, parse_dates=["date"], dtype=dtypes)
    report = evaluate_weekdays(predictor, sales, stock, args.origins, history_cutoff=args.history_cutoff)
    report["model_dir"] = str(args.model_dir.resolve())
    report["input_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.sales, args.stock)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
