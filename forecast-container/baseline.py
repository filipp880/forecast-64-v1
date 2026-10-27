#!/usr/bin/env python3
"""Weekly naive forecasts on exactly the same complete target-week keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import KEY, HORIZON, build_panel, forecast_end, next_week_start, truncate_history
from app.metrics import compute_all_metrics
from app.temporal import last_complete_sunday

METHODS = ("zero", "last_week", "mean4", "mean8", "mean13")


def pairs_at(panel, cutoff):
    history = panel[panel.date <= pd.Timestamp(cutoff)]
    if "_first_sale_date" in history:
        history = history[history._first_sale_date <= pd.Timestamp(cutoff)]
    return history[KEY].drop_duplicates().reset_index(drop=True)


def weekly_keys(pairs, cutoff, horizon=9):
    weeks = pd.DataFrame({"week": pd.date_range(next_week_start(cutoff), periods=horizon, freq="7D")})
    return pairs[KEY].drop_duplicates().assign(_join=1).merge(weeks.assign(_join=1), on="_join").drop(columns="_join")[["week", *KEY]]


def truth_for(panel, cutoff, horizon=9, required_pairs=None):
    cutoff = pd.Timestamp(cutoff)
    if forecast_end(cutoff, horizon) > panel.date.max():
        raise ValueError("Truth requires nine complete weeks, including the final Sunday")
    pairs = pairs_at(panel, cutoff) if required_pairs is None else required_pairs
    keys = weekly_keys(pairs, cutoff, horizon)
    data = panel[(panel.date >= next_week_start(cutoff)) & (panel.date <= forecast_end(cutoff, horizon))][["date", *KEY, "quantity"]].copy()
    data["week"] = data.date - pd.to_timedelta(data.date.dt.dayofweek, unit="D")
    actual = data.groupby(["week", *KEY], observed=True, as_index=False).quantity.sum()
    truth = keys.merge(actual, on=["week", *KEY], how="left", validate="one_to_one")
    truth["quantity"] = truth.quantity.fillna(0.0)
    return truth.sort_values(["week", *KEY]).reset_index(drop=True)


def baseline_forecast(panel, cutoff, method="mean4", horizon=9, combos=None):
    cutoff = pd.Timestamp(cutoff)
    pairs = pairs_at(panel, cutoff) if combos is None else combos
    result = weekly_keys(pairs, cutoff, horizon)
    if method not in METHODS:
        raise ValueError(f"Unknown weekly baseline: {method}")
    if method == "zero":
        result["quantity"] = 0.0
    else:
        weeks = 1 if method == "last_week" else int(method.removeprefix("mean"))
        end = last_complete_sunday(cutoff)
        start = end - pd.Timedelta(days=7 * weeks - 1)
        history = panel[(panel.date >= start) & (panel.date <= end)]
        # The daily panel has explicit zeros. Divide by the number of full weeks,
        # including weeks where this series sold nothing.
        totals = history.groupby(KEY, observed=True, as_index=False).quantity.sum()
        totals["quantity"] /= weeks
        result = result.merge(totals, on=KEY, how="left", validate="many_to_one")
        result["quantity"] = result.quantity.fillna(0.0)
    return result.sort_values(["week", *KEY]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate weekly baselines with complete target weeks")
    ap.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    ap.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    ap.add_argument("--history-cutoff", default="2026-07-26")
    ap.add_argument("--origins", nargs="+", default=["2026-02-15", "2026-05-24"])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--out", type=Path, default=Path("../artifacts/weekly/baselines.json"))
    args = ap.parse_args(argv)
    if pd.Timestamp(args.history_cutoff) > pd.Timestamp("2026-07-26"):
        ap.error("Competition history must end no later than 2026-07-26")
    sales, stock = truncate_history(pd.read_csv(args.sales, parse_dates=["date"]), pd.read_csv(args.stock, parse_dates=["date"]), args.history_cutoff)
    panel = build_panel(sales, stock)
    report = {}
    for origin in args.origins:
        pairs = pairs_at(panel, origin)
        truth = truth_for(panel, origin, HORIZON, pairs)
        report[origin] = {method: compute_all_metrics(truth, baseline_forecast(panel, origin, method, HORIZON, pairs)) for method in args.methods}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
