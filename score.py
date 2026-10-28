#!/usr/bin/env python3
"""Strict weekly submission scoring with rank or baseline normalization.

Ranks are local to the supplied candidates, not an official leaderboard score.
Invalid submissions are reported and skipped; an invalid truth/baseline fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.metrics import METRIC_WEIGHTS, compute_all_metrics
from app.submission import COLUMNS, FORECAST_WEEKS, KEY, _dates, validate_forecast


def normalize_minmax(metrics: dict, baseline_metrics: dict) -> dict:
    """Zero is perfect, one is baseline; worse values are capped at one."""
    result = {}
    for name, values in metrics.items():
        result[name] = {}
        for key in METRIC_WEIGHTS:
            value, base = float(values[key]), float(baseline_metrics[key])
            if not np.isfinite([value, base]).all() or min(value, base) < 0:
                raise ValueError("Metrics must be finite and nonnegative")
            result[name][key] = (0.0 if value == 0 else 1.0) if base == 0 else min(value / base, 1.0)
    return result


def normalize_rank(metrics: dict) -> dict:
    """Normalize average ranks for tied values to [0, 1]."""
    names = list(metrics)
    count = len(names)
    result = {name: {} for name in names}
    for key in METRIC_WEIGHTS:
        values = pd.Series([float(metrics[name][key]) for name in names], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Metrics must be finite and nonnegative")
        ranks = values.rank(method="average").to_numpy()
        for index, name in enumerate(names):
            result[name][key] = float((ranks[index] - 1) / (count - 1)) if count > 1 else 0.0
    return result


def validate_submission(df: pd.DataFrame, df_truth: pd.DataFrame, name: str = "submission") -> list[str]:
    """Compatibility helper returning validation messages instead of repair."""
    try:
        first_week = _dates(df_truth["week"]).min()
        validate_forecast(df, df_truth[KEY], first_week, FORECAST_WEEKS)
        # Also checks every truth key, useful if a caller supplied malformed truth.
        compute_all_metrics(df_truth, df)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return [f"{name}: {exc}"]
    return []


def make_leaderboard(metrics: dict, method: str = "rank", baseline_metrics: dict | None = None, weights: dict | None = None) -> pd.DataFrame:
    """Produce raw metrics, normalized metrics and weighted local score."""
    if not metrics:
        raise ValueError("No valid submissions to score")
    weights = METRIC_WEIGHTS if weights is None else weights
    if set(weights) != set(METRIC_WEIGHTS):
        raise ValueError("Weights must specify all five metric names")
    weight_values = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(weight_values).all() or (weight_values < 0).any() or not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Weights must be finite, nonnegative and sum to one")
    if method == "minmax":
        if baseline_metrics is None:
            raise ValueError("minmax requires a validated baseline")
        normalized = normalize_minmax(metrics, baseline_metrics)
    elif method == "rank":
        normalized = normalize_rank(metrics)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    rows = []
    for name, values in metrics.items():
        row = {"submission": name, "score": sum(float(weights[k]) * normalized[name][k] for k in weights)}
        for key in METRIC_WEIGHTS:
            row[f"{key}_raw"] = float(values[key])
            row[f"{key}_norm"] = normalized[name][key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["score", "submission"], kind="stable").reset_index(drop=True)


def _read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"warehouse_name": "string", "article": "string"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--method", choices=["rank", "minmax"], default="rank")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", help="JSON object with five nonnegative weights summing to 1")
    args = parser.parse_args(argv)
    if args.method == "minmax" and args.baseline is None:
        parser.error("--method minmax requires --baseline")
    names = [path.stem for path in args.submissions]
    if len(names) != len(set(names)):
        parser.error("Submission filenames must have distinct stems")
    try:
        weights = json.loads(args.weights) if args.weights else METRIC_WEIGHTS
        if not isinstance(weights, dict):
            raise ValueError("--weights must be a JSON object")
        truth = _read_csv(args.truth)
        if list(truth.columns) != COLUMNS:
            raise ValueError(f"Truth columns must be {COLUMNS}")
        first_week = _dates(truth["week"]).min()
        validate_forecast(truth, truth[KEY], first_week)
        baseline_metrics = None
        if args.baseline is not None:
            baseline = _read_csv(args.baseline)
            errors = validate_submission(baseline, truth, "baseline")
            if errors:
                raise ValueError("; ".join(errors))
            baseline_metrics = compute_all_metrics(truth, baseline)
        metrics = {}
        for name, path in zip(names, args.submissions):
            try:
                prediction = _read_csv(path)
                errors = validate_submission(prediction, truth, name)
                if errors:
                    raise ValueError("; ".join(errors))
                metrics[name] = compute_all_metrics(truth, prediction)
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                print(f"SKIPPED {name}: {exc}", file=sys.stderr)
        leaderboard = make_leaderboard(metrics, args.method, baseline_metrics, weights)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"Scoring failed: {exc}", file=sys.stderr)
        return 1
    print(f"Normalization: {args.method}. Local comparison, not the official leaderboard.")
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
