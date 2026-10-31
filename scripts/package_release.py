"""Package the working release without raw training data or local artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_release import verify


def release_files(root):
    names = {"README.md", "Dockerfile", "docker-compose.yml", ".dockerignore", ".gitignore",
             ".gitattributes", "pytest.ini", "train.py", "evaluate.py", "score.py", "baseline.py",
             "generate_submission.py", "forecast.csv", "forecast.manifest.json", "required_pairs.csv",
             "observed_pairs.csv"}
    names.update(p.name for p in root.glob("requirements*.txt"))
    names.update(p.name for p in root.glob("train_*.py"))
    for directory in ("app", "tests", ".github"):
        names.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".log"))
    for directory, pattern in (("data", "sample*.csv"), ("model", "*")):
        names.update(p.relative_to(root).as_posix() for p in (root / directory).glob(pattern) if p.is_file())
    names.update(p.relative_to(root).as_posix() for p in (root / "scripts").glob("*.py")
                 if p.name != "analyze_smape.py")  # pre-existing local diagnostic, not a release dependency
    paths = [root / name for name in sorted(names)]
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid release path: {path}")
    return paths


def package(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    check = verify(ROOT / "forecast.csv", ROOT / "required_pairs.csv", ROOT / "forecast.manifest.json", ROOT / "model")
    paths = release_files(ROOT)
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    manifest = {"release_validation": check, "files": {name: hashlib.sha256(data).hexdigest()
                for name, data in contents.items()}, "raw_training_data_included": False,
                "local_artifacts_included": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".release-", dir=output.parent) as directory:
        staged = Path(directory) / "release.zip"
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
            archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(staged) as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt release archive")
            for name, expected in manifest["files"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Archive hash mismatch: {name}")
        # Same-filesystem publication is atomic and fails if a release appeared
        # after the initial check; the temporary link is removed by its context.
        os.link(staged, output)
    return {"archive": str(output.resolve()), "files": len(contents), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "validation": check}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = package(args.out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
