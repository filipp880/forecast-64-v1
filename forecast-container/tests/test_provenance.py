"""Frozen runs must reject changed input, code and OOF before any training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CONTAINER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CONTAINER), str(CONTAINER / "scripts")]

import select_candidate
import verified_backtest as backtest


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sources = ["train.py", "baseline.py", "evaluate.py", "scripts/verified_backtest.py", "scripts/select_candidate.py",
               "app/features.py", "app/calibration.py", "app/metrics.py", "app/models/base.py"]
    for name in sources:
        path = repo / "forecast-container" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source\n", encoding="utf-8")
    data = repo / "data"
    data.mkdir()
    sales, stock = data / "sales.csv", data / "stock.csv"
    sales.write_text("date,quantity\n2026-01-01,1\n", encoding="utf-8")
    stock.write_text("date,stock\n2026-01-01,-1\n", encoding="utf-8")
    out = repo / "artifacts/run"
    select_candidate.initialize_grid(out)
    protocol = {"origins": ["2026-02-15", "2026-05-24"], "history_cutoff": "2026-07-26",
                "normalization": "local_baseline_minmax", "frozen_at": "2026-10-27T00:00:00+00:00",
                "fingerprints": backtest.freeze_fingerprints(sales, stock, repo),
                "candidate_grid_sha256": backtest.file_sha256(out / "candidate_grid.json")}
    protocol_path = out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    loads = []

    class StubPredictor:
        def __init__(self, directory):
            self.directory = directory

        def load(self):
            loads.append(self.directory)
            return self

    import app.predictor
    monkeypatch.setattr(app.predictor, "Predictor", StubPredictor)
    monkeypatch.setattr(backtest, "PROJECT_ROOT", repo)
    for origin in protocol["origins"]:
        fold = out / origin
        fold.mkdir()
        (fold / "model.pkl").write_bytes(b"stub weights")
        (fold / "preprocessor.pkl").write_bytes(b"stub preprocessing")
        (fold / "predictions.csv.gz").write_bytes(b"stub compressed OOF")
        config = {"model_file": "model.pkl", "preprocessor_file": "preprocessor.pkl",
                  "training_data_end": origin, "selection_data_end": origin,
                  "model_artifact_sha256": backtest.file_sha256(fold / "model.pkl"),
                  "preprocessor_sha256": backtest.file_sha256(fold / "preprocessor.pkl"),
                  "provenance": {"protocol_sha256": backtest.file_sha256(protocol_path)}}
        (fold / "config.json").write_text(json.dumps(config), encoding="utf-8")
        record = {"origin": origin, "protocol_sha256": backtest.file_sha256(protocol_path),
                  "model_sha256": backtest.file_sha256(fold / "model.pkl"),
                  "preprocessor_sha256": backtest.file_sha256(fold / "preprocessor.pkl"),
                  "config_sha256": backtest.file_sha256(fold / "config.json"),
                  "predictions_sha256": backtest.file_sha256(fold / "predictions.csv.gz")}
        (fold / "complete.json").write_text(json.dumps(record), encoding="utf-8")
    return repo, out, protocol, loads


def test_unchanged_frozen_run_verifies_and_loads_both_models(frozen_run):
    _, out, protocol, loads = frozen_run
    records = backtest.verify_run_integrity(out, protocol)
    assert set(records) == set(protocol["origins"])
    assert len(loads) == 2


@pytest.mark.parametrize("source", ["sales", "stock"])
def test_changed_input_rejects_resume_and_selection(frozen_run, source):
    repo, out, protocol, loads = frozen_run
    (repo / "data" / f"{source}.csv").write_text("modified input\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"Input SHA-256 changed: {source}"):
        backtest.verify_protocol_fingerprints(protocol)
    with pytest.raises(ValueError, match="Input SHA-256 changed"):
        select_candidate.main(["--out", str(out)])
    assert not loads and not (out / "selection.json").exists()


def test_changed_ml_source_rejects_resume(frozen_run):
    repo, _, protocol, _ = frozen_run
    (repo / "forecast-container/app/features.py").write_text("# modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ML source SHA-256 changed"):
        backtest.verify_protocol_fingerprints(protocol)


def test_added_ml_source_rejects_resume(frozen_run):
    repo, _, protocol, _ = frozen_run
    (repo / "forecast-container/app/models/new_model.py").write_text("# new model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ML source SHA-256 changed"):
        backtest.verify_protocol_fingerprints(protocol)


@pytest.mark.parametrize("filename", ["predictions.csv.gz", "model.pkl", "preprocessor.pkl", "config.json"])
def test_changed_fold_artifact_rejects_before_model_load(frozen_run, filename):
    _, out, protocol, loads = frozen_run
    fold = out / protocol["origins"][0]
    path = fold / filename
    if filename == "config.json":
        config = json.loads(path.read_text(encoding="utf-8"))
        config["training_data_end"] = "2026-07-31"
        path.write_text(json.dumps(config), encoding="utf-8")
    else:
        path.write_bytes(b"modified artifact")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        backtest.verify_fold_artifacts(fold, out / "protocol.json")
    assert not loads


def test_changed_oof_rejects_both_selection_and_confirmation(frozen_run):
    _, out, protocol, loads = frozen_run
    (out / protocol["origins"][0] / "predictions.csv.gz").write_bytes(b"modified OOF")
    for extra in ([], ["--confirm"]):
        with pytest.raises(ValueError, match="predictions.csv.gz"):
            select_candidate.main(["--out", str(out), *extra])
    assert not loads


def test_missing_fold_hash_is_not_silently_resumed(frozen_run):
    _, out, protocol, loads = frozen_run
    fold = out / protocol["origins"][0]
    path = fold / "complete.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["predictions_sha256"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks frozen artifact SHA-256"):
        backtest.verify_fold_artifacts(fold, out / "protocol.json")
    assert not loads


def test_legacy_protocol_without_input_hashes_rejected():
    with pytest.raises(ValueError, match="lacks frozen SHA-256"):
        backtest.verify_protocol_fingerprints({"origins": ["2026-02-15"]})


def test_changed_grid_rejected_before_ooof_read(frozen_run):
    _, out, protocol, loads = frozen_run
    (out / "candidate_grid.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate grid SHA-256 mismatch"):
        backtest.verify_run_integrity(out, protocol)
    assert not loads


def test_fold_temporal_provenance_checked_even_with_matching_hashes(frozen_run):
    _, out, protocol, loads = frozen_run
    fold = out / protocol["origins"][0]
    path = fold / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["selection_data_end"] = "2026-07-31"
    path.write_text(json.dumps(config), encoding="utf-8")
    marker = fold / "complete.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["config_sha256"] = backtest.file_sha256(path)
    marker.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="temporal provenance"):
        backtest.verify_fold_artifacts(fold, out / "protocol.json")
    assert not loads
