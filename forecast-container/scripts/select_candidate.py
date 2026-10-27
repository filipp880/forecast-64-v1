#!/usr/bin/env python3
"""Freeze candidate choice on selection data, confirm that choice once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.calibration import combine
from app.metrics import METRIC_WEIGHTS
from verified_backtest import file_sha256, metrics, verify_protocol_fingerprints, verify_run_integrity


def candidate_grid():
    candidates = {"incumbent": {"round_int": False}}
    for w in (0.0, 0.6, 1.0):
        candidates[f"weight_{w}"] = {"weight_hurdle": w, "round_int": False}
    for gate in (0.3, 0.4, 0.6, 0.7):
        candidates[f"gate_{gate}"] = {"gate": gate, "round_int": False}
    for scale in (0.8, 1.1, 1.2):
        candidates[f"scale_{scale}"] = {"scale": scale, "round_int": False}
    for w in (0.3, 0.6, 1.0):
        candidates[f"expected_{w}"] = {"expected": True, "weight_hurdle": w, "round_int": False}
    candidates["rounded"] = {"round_int": True}
    for gate in (0.3, 0.5, 0.7):
        candidates[f"hurdle_gate_{gate}"] = {"weight_hurdle": 1.0, "gate": gate, "round_int": False}
    return candidates


def initialize_grid(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "candidate_grid.json"
    value = {"granularity": "week", "candidates": candidate_grid(),
             "selection_rule": "minimum weighted baseline-normalized minmax on selection; ties by name",
             "confirmation_rule": "selected candidate only; retain if no worse than incumbent score and SMAPE <= incumbent*1.01; never choose a different candidate on confirmation"}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("The frozen candidate registry differs; choose a new output directory")
    if not path.exists():
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def score(value, baseline):
    # Matches the specified ideal/baseline minmax normalization. This is local,
    # because the organizers' official comparison set is not available here.
    return float(sum(METRIC_WEIGHTS[k] * (min(value[k] / baseline[k], 1.0) if baseline[k] > 0 else float(value[k] != 0)) for k in METRIC_WEIGHTS))


def predict(df, spec):
    return combine(df.probability.to_numpy(float), df.magnitude.to_numpy(float),
                   df.lgbm_raw.to_numpy(float), 7 * df.mean_28.to_numpy(float),
                   df.n_sale_days_90.to_numpy(float), spec)


def summary(df, p):
    return metrics(df.actual.to_numpy(float), p, pd.MultiIndex.from_frame(df[["warehouse_name", "article"]]).to_numpy())


def segment_summary(df, p):
    frame = df.copy()
    frame["prediction"] = p
    frame["frequency"] = pd.cut(frame.n_sale_days_90, [-1, 0, 9, 44, 90], labels=["0/90", "1-9/90", "10-44/90", "45-90/90"])
    frame["horizon"] = pd.cut(frame.h, [0, 3, 6, 9], labels=["1-3", "4-6", "7-9"])
    frame["volume_band"] = pd.cut(7 * frame.mean_28, [-np.inf, 0, 1, 7, np.inf], labels=["zero", "0-1", "1-7", "7+"])
    return {col: {str(k): summary(g, g.prediction.to_numpy(float)) for k, g in frame.groupby(col, observed=True)}
            for col in ["warehouse_name", "frequency", "category_lvl_1", "horizon", "volume_band"]}


def baseline_metrics(df):
    return {col.removeprefix("baseline_"): summary(df, df[col].to_numpy(float)) for col in df if col.startswith("baseline_")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Select and independently confirm a weekly calibration")
    ap.add_argument("--out", type=Path, default=Path("../artifacts/weekly"))
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args(argv)
    protocol = json.loads((args.out / "protocol.json").read_text(encoding="utf-8"))
    artifacts = verify_run_integrity(args.out, protocol)
    origins = protocol["origins"]
    evidence = {"protocol_sha256": file_sha256(args.out / "protocol.json"),
                "candidate_grid_sha256": protocol["candidate_grid_sha256"],
                "selection_predictions_sha256": artifacts[origins[0]]["predictions_sha256"],
                "confirmation_predictions_sha256": artifacts[origins[1]]["predictions_sha256"]}
    if not args.confirm:
        destination = args.out / "selection.json"
        if destination.exists():
            raise FileExistsError("Selection is already frozen")
        df = pd.read_csv(args.out / origins[0] / "predictions.csv.gz")
        grid = json.loads((args.out / "candidate_grid.json").read_text(encoding="utf-8"))["candidates"]
        baseline = summary(df, df.baseline_last_week.to_numpy(float))
        results = {}
        for name, spec in grid.items():
            value = summary(df, predict(df, spec))
            results[name] = {"spec": spec, "metrics": value, "local_score": score(value, baseline)}
        selected = min(results, key=lambda name: (results[name]["local_score"], name))
        record = {"granularity": "week", "forecast_weeks": 9, "history_cutoff": protocol["history_cutoff"],
                  "selection_origin": origins[0], "selected_name": selected,
                  "selected_spec": results[selected]["spec"], "results": results,
                  "baselines": baseline_metrics(df), "normalization": protocol["normalization"],
                  "segments_selected": segment_summary(df, predict(df, results[selected]["spec"]))}
    else:
        destination = args.out / "decision.json"
        if destination.exists():
            raise FileExistsError("Confirmation has already been completed")
        record = json.loads((args.out / "selection.json").read_text(encoding="utf-8"))
        lock = args.out / "selection.sha256"
        if not lock.is_file() or lock.read_text(encoding="ascii").strip() != file_sha256(args.out / "selection.json"):
            raise ValueError("Frozen selection SHA-256 mismatch")
        if record.get("evidence") != evidence:
            raise ValueError("Frozen selection belongs to different inputs or OOF artifacts")
        # Only the previously fixed candidate and incumbent are examined here.
        df = pd.read_csv(args.out / origins[1] / "predictions.csv.gz")
        grid = json.loads((args.out / "candidate_grid.json").read_text(encoding="utf-8"))["candidates"]
        if record.get("selected_name") not in grid or record.get("selected_spec") != grid[record["selected_name"]]:
            raise ValueError("Frozen selection does not match the candidate registry")
        incumbent = summary(df, predict(df, grid["incumbent"]))
        chosen = summary(df, predict(df, record["selected_spec"]))
        baseline = summary(df, df.baseline_last_week.to_numpy(float))
        accepted = score(chosen, baseline) <= score(incumbent, baseline) and chosen["smape"] <= incumbent["smape"] * 1.01
        record = {"granularity": "week", "forecast_weeks": 9, "history_cutoff": protocol["history_cutoff"],
                  "selection_origin": origins[0], "selection_name": record["selected_name"], "confirmation_origin": origins[1],
                  "promote": bool(accepted), "final_name": record["selected_name"] if accepted else "incumbent",
                  "final_spec": record["selected_spec"] if accepted else grid["incumbent"],
                  "confirmation_before": incumbent, "confirmation_candidate": chosen,
                  "confirmation_final": chosen if accepted else incumbent,
                  "confirmation_local_score": score(chosen, baseline),
                  "baselines": baseline_metrics(df),
                  "segments_final": segment_summary(df, predict(df, record["selected_spec"] if accepted else grid["incumbent"])),
                  "normalization": protocol["normalization"],
                  "reason": "Frozen candidate confirmed" if accepted else "Retained incumbent; no second selection on confirmation"}
    verify_protocol_fingerprints(protocol)
    current_artifacts = verify_run_integrity(args.out, protocol)
    if file_sha256(args.out / "protocol.json") != evidence["protocol_sha256"] or any(
            current_artifacts[origin]["predictions_sha256"] != artifacts[origin]["predictions_sha256"] for origin in origins):
        raise ValueError("Frozen protocol or OOF artifacts changed during selection")
    record["evidence"] = evidence
    if args.confirm:
        record["selection_sha256"] = file_sha256(args.out / "selection.json")
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.confirm:
        (args.out / "selection.sha256").write_text(file_sha256(destination) + "\n", encoding="ascii")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
