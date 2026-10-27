#!/usr/bin/env python3
"""Evaluate weekly artifacts without silently permitting in-sample evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import HORIZON, KEY, build_panel, make_features, truncate_history
from app.metrics import compute_all_metrics
from app.predictor import Predictor
from baseline import pairs_at, truth_for
from train import predictions_for

HORIZON_BUCKETS = ((1, 3), (4, 6), (7, 9))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate an artifact on nine complete future weeks")
    ap.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    ap.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    ap.add_argument("--history-cutoff", default="2026-07-26")
    ap.add_argument("--origins", "--cutoffs", nargs="+", required=True)
    ap.add_argument("--required-pairs", type=Path)
    ap.add_argument("--allow-in-sample", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../artifacts/weekly/evaluation.json"))
    args = ap.parse_args(argv)
    if pd.Timestamp(args.history_cutoff) > pd.Timestamp("2026-07-26"):
        ap.error("Competition evaluation cannot use facts after 2026-07-26")
    predictor = Predictor(args.model_dir).load()
    config, model = predictor.config, predictor.model
    boundaries = [pd.Timestamp(config[k]) for k in ("training_data_end", "selection_data_end") if config.get(k)]
    if not boundaries:
        raise ValueError("Artifact lacks temporal provenance")
    if not args.allow_in_sample and any(pd.Timestamp(o) < max(boundaries) for o in args.origins):
        raise ValueError("Artifact saw facts after the evaluation origin; refit a past-only copy or explicitly pass --allow-in-sample")
    sales, stock = truncate_history(pd.read_csv(args.sales, parse_dates=["date"]), pd.read_csv(args.stock, parse_dates=["date"]), args.history_cutoff)
    panel = build_panel(sales, stock)
    required = pd.read_csv(args.required_pairs, dtype=str) if args.required_pairs else None
    results = {}
    post = config.get("postprocess", {})
    for origin in args.origins:
        cut = pd.Timestamp(origin)
        pairs = pairs_at(panel, cut) if required is None else required
        f = make_features(panel, cut, range(1, HORIZON + 1), int(config.get("lookback_days", 180)), required_pairs=pairs)
        pred = f[["week", *KEY]].copy()
        pred["quantity"] = predictions_for(model, f, post.get("zero_threshold", 0.0), post.get("round_int", False), config.get("calibration"))
        truth = truth_for(panel, cut, HORIZON, pairs)
        record = {"metrics": compute_all_metrics(truth, pred), "horizon": {}}
        weeks = sorted(truth.week.unique())
        for lo, hi in HORIZON_BUCKETS:
            target_weeks = weeks[lo - 1:hi]
            record["horizon"][f"{lo}-{hi}"] = compute_all_metrics(truth[truth.week.isin(target_weeks)], pred[pred.week.isin(target_weeks)])
        record["volume_bias"] = float((pred.quantity.sum() / truth.quantity.sum() - 1) * 100) if truth.quantity.sum() else None
        results[origin] = record
    report = {"evaluation_kind": "diagnostic_in_sample" if args.allow_in_sample else "out_of_sample", "granularity": "week", "origins": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
