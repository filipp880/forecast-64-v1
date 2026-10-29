"""Frozen E1 experiment on archived past-only raw predictions, not final weights.

Development uses January/March; October is calibration warm-up. Confirmation is
reserved until a parent selection across E1/E2 explicitly chooses one recipe.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from app.metrics import compute_all_metrics
from app.models.research_model import recipe_prediction
from app.smape_decision import (DecisionSpec, METRICS, SmapeDecision,
                                admissible_oof_origins, development_gate)

ARCHIVE = ROOT / "artifacts/improvements-20261028-r1"
ORIGINS = ["2025-10-19", "2026-01-18", "2026-03-22", "2026-05-24"]
KEY = ["warehouse_name", "article"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def candidates():
    result = {}
    for calibration in ("identity", "platt"):
        for action_mode in ("magnitude", "mixed"):
            result[f"{calibration}_{action_mode}"] = {
                "calibration": calibration, "action_mode": action_mode,
                "magnitude_scales": [.85, 1., 1.15],
                "probability_bins": [.2, .4, .6, .8],
                "quantity_bins": [.5, 1., 2., 4., 8., 16., 32.],
                "positive_loss_shrinkage": 100.,
            }
    return result


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "app/smape_decision.py", ROOT / "app/calibration.py",
             ROOT / "app/models/research_model.py", ROOT / "app/metrics.py", ROOT / "app/features.py",
             ROOT / "model/config.json"]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def initialize(out):
    if (out / "protocol.json").exists():
        protocol = read(out / "protocol.json")
        if sha(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
            raise ValueError("Frozen protocol changed")
        if protocol["source_hashes"] != source_hashes() or protocol["candidates"] != candidates():
            raise ValueError("Frozen sources/grid changed; choose a new run directory")
        return protocol
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty E1 run directory")
    config = read(ROOT / "model/config.json")
    recipe = config["model_params"]["recipe"]
    if config["model_type"] != "research" or recipe["source"] != "dynamics":
        raise ValueError("This experiment requires the frozen dynamics champion")
    inputs = {}
    # Complete records contain provenance/checksums, not truth/prediction arrays.
    # May arrays are not read until the separately requested confirm stage.
    for origin in ORIGINS:
        complete_path = ARCHIVE / origin / "complete.json"
        complete = read(complete_path)
        if complete["origin"] != origin or complete["history_end"] != origin:
            raise ValueError("Archive origin provenance mismatch")
        inputs[origin] = {
            "complete_sha256": sha(complete_path),
            "files": {name: complete["files"][name] for name in ("evaluation.parquet", "dynamics-values.pkl")},
        }
    protocol = {
        "experiment": "e1_expected_smape_decoder_v1", "created_date": "2026-10-29",
        "warmup_origin": ORIGINS[0], "development_origins": ORIGINS[1:3],
        "confirmation_origin": ORIGINS[3], "forecast_weeks": 9,
        "confirmation_policy": "reserved_for_one_joint_parent_selection_across_E1_E2; never_auto_confirm",
        "calibration_policy": "only earlier archived OOF whose complete target_end <= evaluated origin",
        "champion_recipe": recipe, "candidates": candidates(), "inputs": inputs,
        "source_hashes": source_hashes(), "seed": 42,
        "gate": {"mean_weighted_ratio_max": .995, "mean_smape_ratio_max": 1.,
                 "minimum_improved_metrics": 3, "mean_other_metric_ratio_max": 1.01,
                 "per_origin_smape_ratio_max": 1.03, "per_origin_other_ratio_max": 1.05},
        "bootstrap": {"replicates": 300, "seed": 42, "unit": "paired complete pair across development origins"},
        "limitations": [
            "Previously inspected historical periods, not a fresh independent holdout",
            "Archived past-only dynamics boosters are not re-fitted by this experiment",
            "Comparison isolates the decision layer on archived observed-pair feature/prediction arrays",
            "Serving changes/cold coverage are verified separately; no full-product score is inferred",
            "Two development origins only: October is calibration warm-up",
            "Pair bootstrap is conditional on these origins and misses shared warehouse/week shocks",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    write(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n", encoding="utf-8")
    print("PROTOCOL FROZEN before metrics:", sha(out / "protocol.json"), flush=True)
    return protocol


def load_origin(origin, protocol):
    expected = protocol["inputs"][origin]
    directory = ARCHIVE / origin
    if sha(directory / "complete.json") != expected["complete_sha256"]:
        raise ValueError("Archived completion record changed")
    for name, digest in expected["files"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"Archived input changed: {origin}/{name}")
    frame = pd.read_parquet(directory / "evaluation.parquet")
    values = pickle.loads((directory / "dynamics-values.pkl").read_bytes())
    if any(len(values[name]) != len(frame) for name in ("probability", "magnitude", "lgbm_raw")):
        raise ValueError("Archived prediction alignment mismatch")
    if not frame.h.between(1, 9).all() or not frame.groupby(KEY, observed=True).size().eq(9).all():
        raise ValueError("Archive does not contain complete pair horizons")
    if frame.duplicated(["week", *KEY]).any():
        raise ValueError("Duplicate archive keys")
    first = pd.Timestamp(origin) + pd.Timedelta(days=1)
    if not pd.to_datetime(frame.week).eq(first + pd.to_timedelta(7 * (frame.h - 1), unit="D")).all():
        raise ValueError("Archive horizon alignment mismatch")
    continuous_recipe = deepcopy(protocol["champion_recipe"])
    continuous_recipe["post"]["round_int"] = False
    continuous = recipe_prediction({"dynamics": values}, frame, continuous_recipe)
    champion = recipe_prediction({"dynamics": values}, frame, protocol["champion_recipe"])
    return frame, values, continuous, champion


def metrics(frame, prediction):
    truth = frame[["week", *KEY]].assign(quantity=frame.actual.to_numpy(float))
    pred = frame[["week", *KEY]].assign(quantity=prediction)
    return compute_all_metrics(truth, pred)


def diagnostics(frame, prediction, raw_probability=None):
    y, q = frame.actual.to_numpy(float), np.asarray(prediction, float)
    false_zero, false_positive = (y > 0) & (q == 0), (y == 0) & (q > 0)
    positive = (y > 0) & (q > 0)
    loss = np.divide(200 * np.abs(y - q), y + q, out=np.zeros(len(y)), where=y + q > 0)
    result = {
        "false_zero_pp": float(200 * false_zero.mean()),
        "false_positive_pp": float(200 * false_positive.mean()),
        "positive_quantity_pp": float(loss[positive].sum() / len(y)),
        "event_mismatch_rate": float((false_zero | false_positive).mean()),
        "precision": float(positive.sum() / (q > 0).sum()) if (q > 0).any() else None,
        "recall": float(positive.sum() / (y > 0).sum()) if (y > 0).any() else None,
        "volume_bias": float((q.sum() / y.sum() - 1) * 100) if y.sum() else None,
        "zero_share": float((q == 0).mean()),
    }
    if raw_probability is not None:
        p = np.clip(np.asarray(raw_probability, float), 1e-12, 1 - 1e-12)
        event = y > 0
        result["brier"] = float(np.mean((p - event) ** 2))
        result["logloss"] = float(-np.mean(event * np.log(p) + (~event) * np.log1p(-p)))
    return result


def fit_past_decoder(spec, evaluation_origin, protocol):
    prior = admissible_oof_origins(ORIGINS, evaluation_origin)
    if not prior:
        raise ValueError("No fully completed prior OOF origin available")
    values, continuous, actual, weight = [], [], [], []
    for origin in prior:
        frame, raw, raw_champion, _ = load_origin(origin, protocol)
        values.append(raw)
        continuous.append(raw_champion)
        actual.append(frame.actual.to_numpy(float))
        weight.append(np.full(len(frame), 1 / len(frame)))
    model = SmapeDecision(spec).fit(
        np.concatenate([v["probability"] for v in values]), np.concatenate([v["magnitude"] for v in values]),
        np.concatenate(continuous), np.concatenate(actual), np.concatenate(weight),
    )
    return model, prior


def paired_bootstrap(records, seed=42, replicates=300):
    """Paired, full-horizon pair weights shared across development origins."""
    all_pairs = pd.MultiIndex.from_frame(pd.concat([f[KEY] for f, _, _ in records]).drop_duplicates())
    sufficient = []
    for frame, candidate, champion in records:
        codes = pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))[0]
        order = frame[KEY].drop_duplicates()
        positions = all_pairs.get_indexer(pd.MultiIndex.from_frame(order))
        y = frame.actual.to_numpy(float)
        count = np.bincount(codes)
        ysum = np.bincount(codes, weights=y)
        parts = []
        for q in (candidate, champion):
            loss = np.divide(200 * np.abs(y - q), y + q, out=np.zeros(len(y)), where=y + q > 0)
            parts.append({
                "smape": np.bincount(codes, weights=loss) / count,
                "abs": np.bincount(codes, weights=np.abs(y - q)),
                "square": np.bincount(codes, weights=(y - q) ** 2),
                "logsquare": np.bincount(codes, weights=(np.log1p(y) - np.log1p(q)) ** 2),
                "bias": np.divide(100 * np.abs(np.bincount(codes, weights=q) - ysum), ysum,
                                  out=np.zeros(len(count)), where=ysum > 0),
            })
        sufficient.append((positions, count, ysum, parts))
    rng = np.random.default_rng(seed)
    ratios, smape_differences = [], []
    weights = np.array([.5, .2, .15, .1, .05])
    for _ in range(replicates):
        draw = np.bincount(rng.integers(len(all_pairs), size=len(all_pairs)), minlength=len(all_pairs))
        origin_ratios, origin_smape_differences = [], []
        for positions, count, ysum, parts in sufficient:
            w = draw[positions]
            computed = []
            for part in parts:
                computed.append(np.array([
                    np.average(part["smape"], weights=w),
                    100 * (part["abs"] @ w) / max(ysum @ w, 1e-12),
                    np.sqrt((part["square"] @ w) / (count @ w)),
                    np.average(part["bias"][ysum > 0], weights=w[ysum > 0]),
                    np.sqrt((part["logsquare"] @ w) / (count @ w)),
                ]))
            origin_ratios.append(computed[0] / np.maximum(computed[1], 1e-12))
            origin_smape_differences.append(computed[0][0] - computed[1][0])
        ratios.append(float(np.mean(origin_ratios, axis=0) @ weights))
        smape_differences.append(float(np.mean(origin_smape_differences)))
    return {"replicates": replicates, "weighted_ratio_interval_95pct": np.quantile(ratios, [.025, .975]).tolist(),
            "mean_smape_difference_pp_interval_95pct": np.quantile(smape_differences, [.025, .975]).tolist(),
            "fraction_ratio_below_one": float(np.mean(np.asarray(ratios) < 1)),
            "note": "Conditional paired-pair bootstrap; shared warehouse/week shocks not captured"}


def development(out, protocol):
    if (out / "development.json").exists():
        if sha(out / "development.json") != (out / "development.sha256").read_text().strip():
            raise ValueError("Frozen development result changed")
        return read(out / "development.json")
    champion_metrics, results, predictions = {}, {}, {}
    for candidate_id, spec in protocol["candidates"].items():
        result, records = {}, []
        for origin in protocol["development_origins"]:
            decoder, prior = fit_past_decoder(spec, origin, protocol)
            frame, raw, continuous, champion = load_origin(origin, protocol)
            prediction = decoder.predict(raw["probability"], raw["magnitude"], continuous)
            champion_metrics[origin] = metrics(frame, champion)
            result[origin] = {
                "metrics": metrics(frame, prediction), "prior_oof_origins": prior,
                "diagnostics": diagnostics(frame, prediction, decoder.probability(raw["probability"])),
                "champion_diagnostics": diagnostics(frame, champion, raw["probability"]),
                "horizon_metrics": {str(h): metrics(frame.loc[frame.h == h], prediction[frame.h == h]) for h in range(1, 10)},
            }
            medium = (frame.mean_28 * 7 >= .5) & (frame.mean_28 * 7 < 2)
            if medium.any():
                result[origin]["segment_0.5_to_2"] = {
                    "rows": int(medium.sum()), "metrics": metrics(frame.loc[medium], prediction[medium]),
                    "champion": metrics(frame.loc[medium], champion[medium]),
                    "diagnostics": diagnostics(frame.loc[medium], prediction[medium]),
                }
            write(out / "calibrators" / f"{candidate_id}-{origin}.json", decoder.to_dict())
            records.append((frame, prediction, champion))
        gate = development_gate({origin: record["metrics"] for origin, record in result.items()}, champion_metrics)
        results[candidate_id] = {"spec": spec, "origins": result, "gate": gate}
        predictions[candidate_id] = records
        print(candidate_id, json.dumps(gate), flush=True)
    eligible = [name for name, record in results.items() if record["gate"]["eligible"]]
    selected = min(eligible, key=lambda name: (results[name]["gate"]["mean_weighted_ratio"], name)) if eligible else None
    report = {"protocol_sha256": sha(out / "protocol.json"), "development_origins": protocol["development_origins"],
              "champion_metrics": champion_metrics, "results": results, "selected_e1_candidate": selected,
              "confirmation_labels_loaded": False, "booster_training_executed": False,
              "calibrator_training_executed": True, "promotion_executed": False,
              "confirmation_status": "reserved_for_parent_joint_selection" if selected else "skipped_no_eligible_development_candidate",
              "limitations": protocol["limitations"]}
    if selected:
        report["selected_paired_bootstrap"] = paired_bootstrap(predictions[selected], **{
            "seed": protocol["bootstrap"]["seed"], "replicates": protocol["bootstrap"]["replicates"]})
    write(out / "development.json", report)
    (out / "development.sha256").write_text(sha(out / "development.json") + "\n", encoding="utf-8")
    write(out / "decision.json", {"status": report["confirmation_status"], "selected_e1_candidate": selected,
                                  "confirmation_executed": False, "promote": False,
                                  "development_sha256": sha(out / "development.json")})
    return report


def confirmation(out, protocol, candidate_id):
    report = development(out, protocol)
    if candidate_id not in report["results"] or not report["results"][candidate_id]["gate"]["eligible"]:
        raise ValueError("Explicitly selected candidate must have passed development")
    lock = out / "confirmation_selection.json"
    if lock.exists() and read(lock)["candidate_id"] != candidate_id:
        raise ValueError("Confirmation already reserved for a different frozen candidate")
    if (out / "confirmation.json").exists():
        return read(out / "confirmation.json")
    write(lock, {"candidate_id": candidate_id, "protocol_sha256": sha(out / "protocol.json"),
                 "development_sha256": sha(out / "development.json")})
    origin = protocol["confirmation_origin"]
    decoder, prior = fit_past_decoder(protocol["candidates"][candidate_id], origin, protocol)
    frame, raw, continuous, champion = load_origin(origin, protocol)
    prediction = decoder.predict(raw["probability"], raw["magnitude"], continuous)
    candidate_metrics, baseline_metrics = metrics(frame, prediction), metrics(frame, champion)
    gate = development_gate({origin: candidate_metrics}, {origin: baseline_metrics})
    result = {"candidate_id": candidate_id, "candidate_metrics": candidate_metrics, "champion_metrics": baseline_metrics,
              "gate": gate, "prior_oof_origins": prior, "diagnostics": diagnostics(frame, prediction),
              "confirmation_selection_sha256": sha(lock), "promotion_executed": False,
              "note": "Previously inspected historical period; no production artifact replaced"}
    write(out / "confirmation.json", result)
    write(out / "confirmed_decoder.json", decoder.to_dict())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/decision-20261029")
    parser.add_argument("--stage", choices=["development", "confirm"], default="development")
    parser.add_argument("--candidate-id", choices=list(candidates()))
    args = parser.parse_args(argv)
    if args.stage == "confirm" and not args.candidate_id:
        parser.error("confirm requires a parent-selected --candidate-id; it is never automatic")
    protocol = initialize(args.out)
    result = development(args.out, protocol) if args.stage == "development" else confirmation(args.out, protocol, args.candidate_id)
    print(json.dumps({key: value for key, value in result.items() if key not in ("results", "champion_metrics")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
