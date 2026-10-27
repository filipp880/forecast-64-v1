"""Create a reproducible delivery archive without raw training CSV or legacy models."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    container = Path(__file__).resolve().parents[1]
    root = container.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecast", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--validation-dir", type=Path, default=root / "artifacts/weekly-verified-20261027")
    args = ap.parse_args(argv)
    manifest_path = args.forecast.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(container))
    sys.path.insert(0, str(container / "scripts"))
    from app.predictor import Predictor
    from verified_backtest import verify_run_integrity
    verify_run_integrity(args.validation_dir)
    model = Predictor(container / "model").load()
    if model.config != manifest["model_config"]:
        raise ValueError("The working model does not match the generated submission manifest")
    if manifest.get("forecast_sha256") != sha256(args.forecast):
        raise ValueError("Forecast SHA-256 differs from its manifest")
    catalogue = args.forecast.with_name("required_pairs.csv" if manifest["official_pair_coverage_verified"] else "observed_pairs.csv")
    expected_catalogue_hash = manifest["sha256"].get(str(catalogue.resolve()))
    if not expected_catalogue_hash or sha256(catalogue) != expected_catalogue_hash:
        raise ValueError("Pair catalogue SHA-256 differs from its manifest")
    archive = args.out or args.forecast.with_name("forecast_weekly_submission.zip")
    if archive.exists():
        raise FileExistsError(f"Delivery already exists: {archive}")
    paths = [root / "README.md", root / "docs/README.md"]
    paths += sorted((root / "docs/requirements").glob("*.md"))
    paths += sorted((root / "docs/reports").glob("*.md"))
    paths += [container / name for name in ("Dockerfile", ".dockerignore", "docker-compose.yml", "README.md",
              "requirements.txt", "requirements.lock.txt", "requirements-dev.txt", "train.py", "evaluate.py",
              "baseline.py", "score.py", "generate_submission.py")]
    paths += list((container / "app").rglob("*.py"))
    paths += [container / "scripts" / name for name in ("verified_backtest.py", "select_candidate.py",
              "make_samples.py", "check_inventory_contract.py", "check_api_weekdays.py", "check_offline.py", "package_weekly.py", "verify_runtime.py", "verify_docker.py")]
    paths += list((container / "tests").glob("test_*.py")) + [container / "tests/test_api.sh"]
    paths += list((container / "data").glob("sample*.csv"))
    paths += [container / "model" / name for name in ("config.json", "preprocessor.pkl", manifest["model_config"]["model_file"])]
    paths += [container / "logs/README.md", container / "logs/weekly_summary.json", container / "logs/weekly_runtime.json", container / "logs/weekly_docker.json"]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required delivery files are missing: {missing}")
    records = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}
    delivery = {"granularity": "week", "history_cutoff": "2026-07-26", "forecast_weeks": 9,
                "official_pair_coverage_verified": manifest["official_pair_coverage_verified"],
                "coverage_policy": manifest["coverage_policy"], "files": records,
                "forecast_sha256": sha256(args.forecast), "forecast_manifest_sha256": sha256(manifest_path),
                "excludes": ["raw training CSV", "legacy artifacts", "working directories", "credentials"]}
    evidence = [args.validation_dir / name for name in ("protocol.json", "candidate_grid.json", "selection.json", "selection.sha256", "decision.json", "api_weekdays.json", "inventory_contract.json", "offline.json")]
    evidence += list(args.validation_dir.glob("*/complete.json"))
    missing = [str(p) for p in evidence if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required validation evidence is missing: {missing}")
    delivery["validation_evidence"] = {p.relative_to(args.validation_dir).as_posix(): sha256(p) for p in evidence}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for p in sorted(set(paths)):
            handle.write(p, p.relative_to(root).as_posix())
        for p in evidence:
            handle.write(p, "validation/" + p.relative_to(args.validation_dir).as_posix())
        handle.write(args.forecast, "submission/forecast.csv")
        handle.write(manifest_path, "submission/forecast.manifest.json")
        handle.write(catalogue, f"submission/{catalogue.name}")
        handle.writestr("delivery_manifest.json", json.dumps(delivery, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive, "r") as checked:
        bad = checked.testzip()
        if bad is not None:
            raise ValueError(f"Archive verification failed: {bad}")
    delivery["archive_sha256"] = sha256(archive)
    delivery["archive_bytes"] = archive.stat().st_size
    archive.with_suffix(".manifest.json").write_text(json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": delivery["archive_sha256"], "official_pair_coverage_verified": delivery["official_pair_coverage_verified"]}, indent=2))


if __name__ == "__main__":
    main()
