"""Verify a bundled weekly CSV, catalogue, weights and their recorded hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.csv_io import read_csv_checked
from app.predictor import Predictor
from app.submission import validate_forecast
from app.features import next_week_start
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(forecast, pairs, manifest, model_dir):
    manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    hashes = {"forecast_sha256": digest(forecast), "required_pairs_sha256": digest(pairs)}
    for key, actual in hashes.items():
        if manifest.get(key) != actual:
            raise ValueError(f"Release hash mismatch: {key}")
    predictor = Predictor(model_dir).load()
    cutoff, first_week = manifest["history_cutoff"], manifest["first_week"]
    if pd.Timestamp(first_week) != next_week_start(cutoff):
        raise ValueError("Release first_week must follow the history cutoff's calendar week")
    predictor.assert_history_compatible(cutoff)
    current_model = {p.name: digest(p) for p in Path(model_dir).iterdir() if p.is_file()}
    if "model_files" in manifest:
        if manifest["model_files"] != current_model:
            raise ValueError("Release model bundle differs from manifest")
    elif manifest.get("model_config") != predictor.config:
        raise ValueError("Release model config differs from manifest")
    summary = validate_forecast(read_csv_checked(forecast), read_csv_checked(pairs), first_week)
    return {"valid": True, **summary, **hashes, "model_files": current_model,
            "official_pair_coverage_verified": bool(manifest.get("official_pair_coverage_verified", False)),
            "external_key_equality_asserted_by_this_check": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=ROOT / "forecast.csv")
    parser.add_argument("--pairs", type=Path, default=ROOT / "required_pairs.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "forecast.manifest.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.forecast, args.pairs, args.manifest, args.model_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
