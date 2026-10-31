#!/usr/bin/env python3
"""Train a direct nine-week model with chronological selection and final refit."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import (
    CAT_FEATURES, FEATURE_COLUMNS, HORIZON, KEY, LOOKBACK_DAYS,
    PREPROCESSING_VERSION, build_panel, forecast_end, make_training_frame,
    preprocessing_spec, truncate_history,
)
from app.metrics import compute_all_metrics, format_metrics
from app.models import available_model_types, get_model_class, transform_target
from app.temporal import last_complete_sunday
from app.csv_io import read_csv_checked
from app.pipeline import prepare_panel, feature_frame

COMPETITION_CUTOFF = pd.Timestamp("2026-07-26")


def make_cutoffs(panel, lookback_days=180, horizon=9, step_days=28):
    """Sunday origins; labels are restricted to complete target weeks later."""
    if step_days <= 0 or step_days % 7:
        raise ValueError("cutoff-step must be a positive multiple of seven days")
    first = pd.Timestamp(panel.date.min()) + pd.Timedelta(days=lookback_days - 1)
    first += pd.Timedelta(days=(6 - first.weekday()) % 7)
    last = last_complete_sunday(panel.date.max()) - pd.Timedelta(days=7)
    if first > last:
        raise ValueError("Not enough history for a complete target week")
    return list(pd.date_range(first, last, freq=f"{step_days}D"))


def holdout_cutoffs(panel, horizon=9):
    last = last_complete_sunday(panel.date.max())
    test = last - pd.Timedelta(weeks=horizon)
    selection = test - pd.Timedelta(weeks=horizon)
    return selection, test


def build_dataset(panel, cutoffs, horizon=9, lookback_days=180, stock_policy="window", model_type=None, model_params=None):
    """Use only horizons with seven available target days; never fake missing labels."""
    frames = []
    for i, cutoff in enumerate(cutoffs, 1):
        complete = [h for h in range(1, horizon + 1)
                    if forecast_end(cutoff, h) <= panel.date.max()]
        if not complete:
            continue
        frame = feature_frame(panel, cutoff, model_type, training=True, horizons=complete,
                              lookback_days=lookback_days, stock_policy=stock_policy, model_params=model_params)
        frame["origin"] = pd.Timestamp(cutoff)
        frames.append(frame)
        print(f"  [{i}/{len(cutoffs)}] {pd.Timestamp(cutoff).date()}: {len(frame):,} rows, {len(complete)} complete weeks", flush=True)
    if not frames:
        raise ValueError("No complete training weeks available")
    return frames


def evaluate_frame(model, frame, zero_threshold=0.0, round_int=False, calibration=None):
    truth = frame[["week", *KEY]].copy()
    if "origin" in frame:
        truth["origin"] = frame.origin
    truth["quantity"] = frame.target.to_numpy(float)
    pred = truth.drop(columns="quantity").copy()
    pred["quantity"] = predictions_for(model, frame, zero_threshold, round_int, calibration)
    return compute_all_metrics(truth, pred)


def predictions_for(model, frame, zero_threshold=0.0, round_int=False, calibration=None):
    from app.calibration import forecast_quantities
    return forecast_quantities(model, frame, zero_threshold, round_int, calibration)


def compact_params(estimators=800):
    """A bounded CPU recipe; selected iteration counts are reused by refit."""
    return {
        "weight_hurdle": 0.3,
        "hurdle": {"classifier": {"n_estimators": estimators},
                   "regressor": {"n_estimators": estimators}},
        "lgbm": {"n_estimators": estimators},
    }


def save_artifact(model, out_dir, training_end, selection_end, rows, cutoffs,
                  lookback=180, horizon=9, postprocess=None, validation=None,
                  calibration=None, provenance=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model{model.file_suffix}"
    model.save(model_path)
    spec = {**preprocessing_spec(stock_policy="window", lookback_days=lookback, horizon=horizon),
            "target_transform": model.target_transform}
    if model.name == "research":
        from app.enhancements import enhanced_spec, feature_columns
        spec.update(schema_version=6 if model.recipe.get('segments') else 4,
                    feature_columns=feature_columns(), enhancements=enhanced_spec())
        if model.params.get('expert_specs'):
            from app.sequence_features import sequence_spec, combined_feature_columns
            spec.update(schema_version=7, feature_columns=combined_feature_columns(), metric_experts=sequence_spec())
    if model.name == "ideas":
        from app.enhancements import enhanced_spec, feature_columns
        from app.ideas_features import GROUPS, ideas_columns, ideas_spec
        spec.update(schema_version=5,feature_columns=feature_columns()+ideas_columns(GROUPS),
                    enhancements=enhanced_spec(),ideas=ideas_spec())
    preprocessor = out_dir / "preprocessor.pkl"
    preprocessor.write_bytes(pickle.dumps(spec, protocol=4))
    config = {
        "model_type": model.name, "model_file": model_path.name,
        "preprocessor_file": preprocessor.name, "granularity": "week",
        "horizon": horizon, "forecast_weeks": horizon,
        "preprocessing_version": PREPROCESSING_VERSION,
        "stock_policy": "window", "stock_aggregation": "sum",
        "negative_stock_policy": "preserve", "forecast_alignment": "next_calendar_monday",
        "lookback_days": lookback, "target_transform": model.target_transform,
        "feature_columns": list(FEATURE_COLUMNS), "cat_features": list(CAT_FEATURES),
        "postprocess": postprocess or {"zero_threshold": 0.0, "round_int": False},
        "training_data_end": str(pd.Timestamp(training_end).date()),
        "selection_data_end": str(pd.Timestamp(selection_end).date()),
        "random_seed": int(model.params.get("seed", 42)),
        "thread_count": int(model.params.get("threads", 8)),
        "n_cutoffs": int(cutoffs), "train_rows": int(rows), "model_params": model.params,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage_policy": "requested_pairs_or_full_product_with_observed_only_group_features",
        "model_artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "preprocessor_sha256": hashlib.sha256(preprocessor.read_bytes()).hexdigest(),
        "validation": validation,
    }
    if calibration:
        config["calibration"] = calibration
    if model.name == "research":
        config.update(preprocessing_version=spec['schema_version'],feature_columns=spec["feature_columns"],enhancements=spec["enhancements"])
        if model.params.get('expert_specs'):
            config['metric_experts'] = spec['metric_experts']
    if model.name == "ideas":
        config.update(preprocessing_version=5,feature_columns=spec['feature_columns'],
                      enhancements=spec['enhancements'],ideas=spec['ideas'])
    if provenance:
        config["provenance"] = provenance
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train direct weekly forecasts, horizon nine weeks")
    ap.add_argument("--sales", type=Path, default=Path("data/sales_dataset_hakaton_train.csv"))
    ap.add_argument("--stock", type=Path, default=Path("data/stock_dataset_hakaton_train.csv"))
    ap.add_argument("--history-cutoff", default="2026-07-26")
    ap.add_argument("--out-dir", type=Path, default=Path("model_weekly"))
    ap.add_argument("--model-type", choices=available_model_types(), default="blend")
    ap.add_argument("--model-params", default="{}")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--cutoff-step", type=int, default=28)
    ap.add_argument("--max-cutoffs", type=int)
    ap.add_argument("--estimators", type=int, default=800)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--round-int", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--zero-threshold", type=float, default=0.0)
    ap.add_argument("--calibration-file", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.model_type in ("research", "ideas") and args.seed != 42:
        ap.error("research/ideas currently use a fixed seed of 42; choose --seed 42")
    if args.horizon != 9:
        ap.error("The submission horizon is exactly nine weeks")
    if args.lookback < 90 or args.cutoff_step <= 0 or args.cutoff_step % 7:
        ap.error("lookback>=90; cutoff-step must be a positive multiple of seven")
    if args.threads <= 0 or args.estimators <= 0 or (args.max_cutoffs is not None and args.max_cutoffs <= 0):
        ap.error("threads, estimators and max-cutoffs must be positive")
    cutoff = pd.Timestamp(args.history_cutoff)
    if cutoff > COMPETITION_CUTOFF:
        ap.error("Competition training cannot use data after 2026-07-26")
    try:
        supplied = json.loads(args.model_params)
    except json.JSONDecodeError as exc:
        ap.error(str(exc))
    if not isinstance(supplied, dict):
        ap.error("model-params must be a JSON object")
    if supplied.get('serving_profile') in ('all40_v1', 'joint_v1'):
        ap.error('Exported ensembles require their original component training and release exporter; '
                 'train.py cannot refit this serving graph. Existing bundled weights need no training.')
    if args.model_type == "blend":
        params = compact_params(args.estimators)
    elif args.model_type == "hurdle":
        params = {"classifier": {"n_estimators": args.estimators}, "regressor": {"n_estimators": args.estimators}}
    elif args.model_type == "catboost":
        params = {"iterations": args.estimators}
    elif args.model_type in ("research", "ideas"):
        params = {"estimators": args.estimators}
        if args.model_type == "ideas":
            params.update(specs=[{"kind": "blend", "groups": []}], weights=[1.0])
    else:
        params = {"n_estimators": args.estimators}
    params.update(supplied)
    args.model_params = {"seed": args.seed, "threads": args.threads, **params}
    return args


def main(argv=None):
    args = parse_args(argv)
    if (args.out_dir / "config.json").exists() and not args.overwrite:
        raise FileExistsError("Artifact exists; choose another --out-dir or pass --overwrite")
    started = time.perf_counter()
    sales = read_csv_checked(args.sales, parse_dates=["date"])
    stock = read_csv_checked(args.stock, parse_dates=["date"])
    sales, stock = truncate_history(sales, stock, args.history_cutoff)
    panel = prepare_panel(sales, stock, args.model_type, args.model_params)
    del sales, stock
    gc.collect()
    selection, test = holdout_cutoffs(panel, args.horizon)
    cutoffs = make_cutoffs(panel, args.lookback, args.horizon, args.cutoff_step)
    eligible = [c for c in cutoffs if forecast_end(c, args.horizon) <= selection]
    if args.max_cutoffs:
        eligible = eligible[-args.max_cutoffs:]
    if not eligible:
        raise ValueError("Not enough data for independent train/selection/test windows")
    print(f"Weekly selection origin {selection.date()}, confirmation origin {test.date()}", flush=True)
    frames = build_dataset(panel, eligible, args.horizon, args.lookback, model_type=args.model_type, model_params=args.model_params)
    train = pd.concat(frames, ignore_index=True)
    del frames
    initial_rows = len(train)
    if (train.week + pd.Timedelta(days=6)).max() > selection:
        raise RuntimeError("Training targets overlap selection window")
    valid = feature_frame(panel, selection, args.model_type, training=True,
                          horizons=range(1, args.horizon + 1), lookback_days=args.lookback, model_params=args.model_params)
    model = get_model_class(args.model_type)(args.model_params)
    model.fit(train, transform_target(train.target.to_numpy(float), model.target_transform),
              valid, transform_target(valid.target.to_numpy(float), model.target_transform),
              args.early_stopping_rounds)
    iterations = model.selected_iterations
    del train, valid
    gc.collect()
    validation_frame = feature_frame(panel, test, args.model_type, training=True,
                                     horizons=range(1, args.horizon + 1), lookback_days=args.lookback, model_params=args.model_params)
    calibration = None
    calibration_record = None
    if args.calibration_file:
        calibration_record = json.loads(args.calibration_file.read_text(encoding="utf-8"))
        calibration = calibration_record.get("final_spec")
        if calibration is None or model.name != "blend" or not calibration_record.get("confirmation_origin"):
            raise ValueError("calibration-file requires a confirmed blend calibration")
        calibration_end = forecast_end(pd.Timestamp(calibration_record["confirmation_origin"]), args.horizon)
        if calibration_end > pd.Timestamp(args.history_cutoff):
            raise ValueError("Calibration used facts beyond the authorized history cutoff")
    validation = {"scheme": "weekly_chronological_holdout", "selection_origin": str(selection.date()),
                  "confirmation_origin": str(test.date()), "target_end": str(forecast_end(test, 9).date()),
                  "iterations": iterations,
                  "metrics_kind": "out_of_sample_before_final_refit",
                  "metrics": evaluate_frame(model, validation_frame, args.zero_threshold, args.round_int)}
    if calibration_record:
        validation["calibrated_metrics"] = evaluate_frame(model, validation_frame, args.zero_threshold, args.round_int, calibration)
        validation["calibrated_metrics_kind"] = (
            "diagnostic_calibration_seen_truth" if calibration_end > test else "out_of_sample_before_final_refit")
        validation["independent_confirmation_source"] = str(args.calibration_file)
    print(format_metrics(validation["metrics"]), flush=True)
    del validation_frame
    refit_cutoffs = cutoffs[-args.max_cutoffs:] if args.max_cutoffs else cutoffs
    final_rows = initial_rows
    training_end = selection
    if args.refit:
        frames = build_dataset(panel, refit_cutoffs, args.horizon, args.lookback, model_type=args.model_type, model_params=args.model_params)
        refit = pd.concat(frames, ignore_index=True)
        del frames
        final_rows = len(refit)
        training_end = (refit.week + pd.Timedelta(days=6)).max()
        if training_end > pd.Timestamp(args.history_cutoff):
            raise RuntimeError("Final labels exceed the authorized history cutoff")
        model.fit(refit, transform_target(refit.target.to_numpy(float), model.target_transform), n_iterations=iterations)
        del refit
        gc.collect()
    # Conservative provenance includes every inspected validation fact, even
    # when the holdout metrics did not affect the fitted tree weights.
    selection_end = max(forecast_end(test, args.horizon), calibration_end) if calibration_record else forecast_end(test, args.horizon)
    config = save_artifact(model, args.out_dir, training_end, selection_end, final_rows,
                           len(refit_cutoffs) if args.refit else len(eligible), args.lookback,
                           args.horizon, {"zero_threshold": args.zero_threshold, "round_int": args.round_int},
                           validation, calibration,
                           {"history_cutoff": str(pd.Timestamp(args.history_cutoff).date()),
                            "recipe": "direct_weekly", "seconds": round(time.perf_counter() - started, 3),
                            "calibration_file": str(args.calibration_file) if args.calibration_file else None,
                            "calibration_sha256": hashlib.sha256(args.calibration_file.read_bytes()).hexdigest() if args.calibration_file else None})
    print(json.dumps({"model_dir": str(args.out_dir), "training_data_end": config["training_data_end"],
                      "seconds": round(time.perf_counter() - started, 1)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
