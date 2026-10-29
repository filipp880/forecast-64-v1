"""Frozen artificial leave-pair-out fallback diagnostic on historical cache.

This does not promote a fallback or estimate quality on truly unseen IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from app.metrics import compute_all_metrics
from app.submission import normalized_pairs
from scripts.research_decision import paired_bootstrap

KEY = ["warehouse_name", "article"]
ORIGINS = ["2026-01-18", "2026-03-22"]
ARCHIVE = ROOT / "artifacts/improvements-20261028-r1"
POLICIES = ["zero", "current_article_mean", "shrunk_article_warehouse"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_holdout(history, fraction=.10, seed=42):
    """Select by ID only; every held ID remains represented in source pairs."""
    pairs = normalized_pairs(history)
    chosen = np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}\0{warehouse}\0{article}".encode("utf-8")).digest()[:8], "big") / 2**64 < fraction
        for warehouse, article in pairs.itertuples(index=False, name=None)
    ])
    source = pairs.loc[~chosen]
    supported = pairs.warehouse_name.isin(source.warehouse_name) & pairs.article.isin(source.article)
    chosen &= supported.to_numpy()
    return pairs.loc[chosen].reset_index(drop=True)


def source_and_holdout(history, held):
    history = history[[*KEY, "weekly_mean"]].copy()
    if history.duplicated(KEY).any() or not np.isfinite(history.weekly_mean).all() or history.weekly_mean.lt(0).any():
        raise ValueError("Expected finite nonnegative pair history without duplicates")
    held = normalized_pairs(held)
    indices = pd.MultiIndex.from_frame(history[KEY])
    held_index = pd.MultiIndex.from_frame(held)
    if len(held_index.difference(indices)):
        raise ValueError("Held pairs are not in the origin-known catalogue")
    source = history.loc[~indices.isin(held_index)].copy()
    if source.empty or not held.warehouse_name.isin(source.warehouse_name).all() or not held.article.isin(source.article).all():
        raise ValueError("Every held article and warehouse must remain known through other source pairs")
    return source, held


def fallback_predictions(history, held):
    """Recompute all aggregates after censoring; never use cached group means."""
    source, held = source_and_holdout(history, held)
    by_article = source.groupby("article", observed=True).weekly_mean.agg(["sum", "size", "mean"])
    by_warehouse = source.groupby("warehouse_name", observed=True).weekly_mean.agg(["mean", "size"])
    global_mean = float(source.weekly_mean.mean())
    article = held.article.map(by_article["mean"]).to_numpy(float)
    count = held.article.map(by_article["size"]).to_numpy(float)
    shrink = (held.article.map(by_article["sum"]).to_numpy(float) + 5 * global_mean) / (count + 5)
    warehouse_count = held.warehouse_name.map(by_warehouse["size"]).to_numpy(float)
    raw_factor = held.warehouse_name.map(by_warehouse["mean"]).to_numpy(float) / max(global_mean, 1e-12)
    factor = 1 + warehouse_count / (warehouse_count + 50) * (np.clip(raw_factor, .5, 2.) - 1)
    predictions = {
        "zero": np.zeros(len(held)),
        "current_article_mean": np.rint(article),
        "shrunk_article_warehouse": np.rint(np.maximum(shrink * factor, 0.)),
    }
    support = {
        "source_pairs": len(source), "held_pairs": len(held),
        "held_warehouses": int(held.warehouse_name.nunique()), "held_articles": int(held.article.nunique()),
        "article_support_min": int(count.min()) if len(count) else 0,
        "article_support_median": float(np.median(count)) if len(count) else 0.,
        "warehouse_support_min": int(warehouse_count.min()) if len(warehouse_count) else 0,
        "source_weekly_mean": global_mean,
        "source_and_held_disjoint": True,
    }
    return {name: held.assign(quantity=value) for name, value in predictions.items()}, support


def initialize(out):
    files = [Path(__file__).resolve(), ROOT / "scripts/research_decision.py", ROOT / "app/metrics.py", ROOT / "app/submission.py"]
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    if (out / "protocol.json").exists():
        protocol = read(out / "protocol.json")
        if sha(out / "protocol.json") != (out / "protocol.sha256").read_text().strip() or protocol["source_hashes"] != sources:
            raise ValueError("Frozen protocol/sources changed; choose a new run directory")
        return protocol
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty cold diagnostic directory")
    inputs = {}
    for origin in ORIGINS:
        complete_path = ARCHIVE / origin / "complete.json"
        complete = read(complete_path)
        if complete["origin"] != origin or complete["history_end"] != origin:
            raise ValueError("Archive temporal provenance mismatch")
        inputs[origin] = {"complete_sha256": sha(complete_path),
                          "evaluation_sha256": complete["files"]["evaluation.parquet"]}
    protocol = {
        "experiment": "e4_artificial_leave_pair_out_v1", "date": "2026-10-29",
        "origins": ORIGINS, "inputs": inputs, "source_hashes": sources,
        "holdout": "first64bits SHA256(seed NUL warehouse NUL article)/2**64 < .10; unsupported IDs restored to source",
        "holdout_fraction": .10, "seed": 42,
        "history": "7*mean_28 per observed pair; recompute every group aggregate after removing all held pairs",
        "policies": POLICIES, "round_int": True,
        "shrink_prior_pairs": 5, "warehouse_factor_prior_pairs": 50,
        "warehouse_factor_clip_before_shrinkage": [.5, 2.],
        "bootstrap_replicates": 300,
        "confirmation_policy": "No May input or confirmation; diagnostic only; no production promotion",
        "limitations": ["Previously inspected historical data, no fresh independent holdout",
                        "Artificial missing pair with known warehouse/article, not truly unseen IDs",
                        "Only cached per-pair history is used; no raw-history retraining/model evaluation",
                        "Existing observed-history features are not evidence of a true product birth date",
                        "Pair bootstrap conditional on these origins; shared warehouse/week shocks omitted"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n", encoding="utf-8")
    print("COLD PROTOCOL FROZEN before metrics:", sha(out / "protocol.json"), flush=True)
    return protocol


def run(out, protocol):
    if (out / "results.json").exists():
        if sha(out / "results.json") != (out / "results.sha256").read_text().strip():
            raise ValueError("Frozen results changed")
        return read(out / "results.json")
    results, bootstrap_records = {}, {name: [] for name in POLICIES}
    for origin in protocol["origins"]:
        directory = ARCHIVE / origin
        expected = protocol["inputs"][origin]
        if sha(directory / "complete.json") != expected["complete_sha256"] or sha(directory / "evaluation.parquet") != expected["evaluation_sha256"]:
            raise ValueError("Cached origin input changed")
        # Deliberately do not load any cached cross-pair aggregate or May data.
        frame = pd.read_parquet(directory / "evaluation.parquet", columns=["week", *KEY, "h", "mean_28", "actual"])
        if frame.duplicated(["week", *KEY]).any() or not frame.groupby(KEY, observed=True).size().eq(9).all():
            raise ValueError("Incomplete archived pair horizons")
        distinct = frame.groupby(KEY, observed=True).mean_28.nunique()
        if not distinct.eq(1).all():
            raise ValueError("History changes across horizons")
        history = frame.drop_duplicates(KEY)[[*KEY, "mean_28"]].assign(weekly_mean=lambda f: 7 * f.mean_28)
        held = select_holdout(history, protocol["holdout_fraction"], protocol["seed"])
        if held.empty:
            raise ValueError("No supported held pairs")
        predictions, support = fallback_predictions(history, held)
        evaluated = frame.merge(held, on=KEY, how="inner", validate="many_to_one")
        truth = evaluated[["week", *KEY]].assign(quantity=evaluated.actual.to_numpy(float))
        baseline = evaluated.merge(predictions["current_article_mean"], on=KEY, how="left", validate="many_to_one").quantity.to_numpy(float)
        result = {"support": support, "rows": len(evaluated), "actual_volume": float(evaluated.actual.sum()), "policies": {}}
        article_support = history.loc[~pd.MultiIndex.from_frame(history[KEY]).isin(pd.MultiIndex.from_frame(held))].groupby("article", observed=True).size()
        for name, pair_prediction in predictions.items():
            forecast = evaluated[["week", *KEY]].merge(pair_prediction, on=KEY, how="left", validate="many_to_one")
            q = forecast.quantity.to_numpy(float)
            record = {"metrics": compute_all_metrics(truth, forecast), "predicted_volume": float(q.sum()),
                      "zero_share": float((q == 0).mean()), "support_segments": {}}
            for segment, mask in {
                "article_support_1_to_3": evaluated.article.map(article_support).le(3),
                "article_support_4_plus": evaluated.article.map(article_support).gt(3),
            }.items():
                record["support_segments"][segment] = {"rows": int(mask.sum()), "pairs": int(evaluated.loc[mask, KEY].drop_duplicates().shape[0])}
                if mask.any():
                    record["support_segments"][segment]["metrics"] = compute_all_metrics(truth.loc[mask], forecast.loc[mask])
            result["policies"][name] = record
            bootstrap_records[name].append((evaluated, q, baseline))
        results[origin] = result
        held.to_csv(out / f"held_pairs_{origin}.csv", index=False)
    report = {"protocol_sha256": sha(out / "protocol.json"), "origins": results,
              "conditional_paired_bootstrap_vs_current": {
                  name: paired_bootstrap(records, seed=42, replicates=protocol["bootstrap_replicates"])
                  for name, records in bootstrap_records.items() if name != "current_article_mean"},
              "confirmation_loaded": False, "production_changed": False, "booster_training_executed": False,
              "limitations": protocol["limitations"]}
    write(out / "results.json", report)
    (out / "results.sha256").write_text(sha(out / "results.json") + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/cold-pairs-20261029")
    args = parser.parse_args(argv)
    protocol = initialize(args.out)
    print(json.dumps(run(args.out, protocol), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
