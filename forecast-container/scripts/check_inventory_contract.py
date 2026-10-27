#!/usr/bin/env python3
"""Verify signed, summed daily stock against the panel used by weekly inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import KEY, build_panel, truncate_history


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify daily sum and reservation preservation")
    ap.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    ap.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    ap.add_argument("--origin", default="2026-07-26")
    ap.add_argument("--out", type=Path, default=Path("../artifacts/weekly/inventory_contract.json"))
    args = ap.parse_args(argv)
    if pd.Timestamp(args.origin) > pd.Timestamp("2026-07-26"):
        ap.error("Competition inventory cutoff cannot exceed 2026-07-26")
    sales, stock = truncate_history(pd.read_csv(args.sales, parse_dates=["date"]), pd.read_csv(args.stock, parse_dates=["date"]), args.origin)
    if "article" not in stock and "char_article" in stock:
        stock = stock.rename(columns={"char_article": "article"})
    panel = build_panel(sales, stock)
    expected = stock.groupby(["date", *KEY], as_index=False).stock.sum()
    compared = panel[["date", *KEY, "stock"]].merge(expected, on=["date", *KEY], how="inner", suffixes=("_panel", "_expected"), validate="one_to_one")
    if compared.empty or not np.allclose(compared.stock_panel, compared.stock_expected, equal_nan=True):
        raise AssertionError("Daily stock is not the signed sum of original records")
    record = {"origin": args.origin, "stock_aggregation": "sum", "negative_stock_policy": "preserve",
              "raw_stock_rows": len(stock), "negative_raw_rows": int((stock.stock < 0).sum()),
              "negative_aggregated_daily_keys": int((expected.stock < 0).sum()),
              "duplicate_rows_aggregated": int(len(stock) - len(expected)), "compared_keys": len(compared),
              "equal_daily_stock": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
