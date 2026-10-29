"""Submission orchestration regressions; inference math is tested separately."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import generate_submission as generator
from app.coverage import cartesian_pairs
from app.submission import normalized_pairs


@pytest.fixture
def generation_inputs(tmp_path, monkeypatch):
    dates = pd.date_range("2026-01-28", "2026-07-26")
    sales = pd.DataFrame([
        {"date": date.strftime("%Y-%m-%d"), "warehouse_name": warehouse, "article": article,
         "quantity": .25, "composite_division": "d", "cities_name": "c",
         "category_lvl_1": "a", "category_lvl_2": "b", "category_lvl_3": "c",
         "characteristic": "x", "sales_sum": 1., "discount_sum": float("nan")}
        for warehouse, article in [("001", "002"), ("003", "004")]
        for date in dates
    ])
    stock = sales[["date", "warehouse_name", "article"]].assign(stock=10.)
    sales_path, stock_path = tmp_path / "sales.csv", tmp_path / "stock.csv"
    sales.to_csv(sales_path, index=False)
    stock.to_csv(stock_path, index=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"fixture": true}', encoding="utf-8")
    calls = []

    class Predictor:
        config = {"postprocess": {"round_int": False}}

        def __init__(self, model_dir):
            pass

        def load(self):
            return self

        def assert_history_compatible(self, cutoff):
            pass

        def predict(self, sales, stock, *, history_cutoff, first_week, required_pairs):
            calls.append(required_pairs)
            pairs = cartesian_pairs(sales) if required_pairs is None else normalized_pairs(required_pairs)
            weeks = pd.DataFrame({"week": pd.date_range(first_week, periods=9, freq="7D").strftime("%Y-%m-%d")})
            return weeks.merge(pairs, how="cross").assign(quantity=.25)

    monkeypatch.setattr(generator, "Predictor", Predictor)
    args = ["--sales", str(sales_path), "--stock", str(stock_path), "--model-dir", str(model_dir)]
    return tmp_path, args, calls, sales


def test_default_uses_shared_predictor_coverage_without_rounding(generation_inputs):
    root, args, calls, _ = generation_inputs
    output = root / "fractional.csv"
    assert generator.main([*args, "--out", str(output)]) == 0
    forecast = pd.read_csv(output, dtype={"warehouse_name": str, "article": str})
    assert len(forecast) == 36
    assert forecast.quantity.eq(.25).all()
    assert set(forecast.warehouse_name) == {"001", "003"}
    assert calls == [None, None]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["fallback_pairs"] == 2
    assert manifest["forecast_sha256"] == generator.file_sha256(output)
    assert manifest["required_pairs_sha256"] == generator.file_sha256(root / "fractional.pairs.csv")


def test_supplied_catalogue_never_implies_official_verification(generation_inputs):
    root, args, calls, sales = generation_inputs
    source = root / "required_pairs.csv"
    pairs = cartesian_pairs(sales).assign(source_note="reconstructed locally")
    pairs.to_csv(source, index=False)
    original = source.read_bytes()
    output = root / "new.csv"
    generator.main([*args, "--required-pairs", str(source), "--out", str(output)])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert source.read_bytes() == original
    assert manifest["catalogue_provided"] is True
    assert manifest["local_key_match"] is True
    assert manifest["official_pair_coverage_verified"] is False
    assert manifest["external_verification"] is None
    assert manifest["input_sha256"][str(source.resolve())] == generator.file_sha256(source)
    assert manifest["coverage_policy"] == "supplied_catalogue"
    assert len(calls) == 2 and all(len(frame) == 4 for frame in calls)


def test_separate_runs_do_not_touch_shared_or_each_others_catalogues(generation_inputs):
    root, args, _, _ = generation_inputs
    shared = root / "required_pairs.csv"
    shared.write_bytes(b"local source catalogue\n")
    generator.main([*args, "--out", str(root / "first.csv")])
    first_hashes = {path.name: generator.file_sha256(path) for path in root.glob("first.*")}
    generator.main([*args, "--out", str(root / "second.csv")])
    assert shared.read_bytes() == b"local source catalogue\n"
    assert first_hashes == {path.name: generator.file_sha256(path) for path in root.glob("first.*")}


@pytest.mark.parametrize("sidecar", ["new.pairs.csv", "new.manifest.json"])
def test_existing_sidecar_is_protected_before_inference(generation_inputs, sidecar):
    root, args, calls, _ = generation_inputs
    target = root / sidecar
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        generator.main([*args, "--out", str(root / "new.csv")])
    assert not calls
    assert target.read_bytes() == b"keep"
    assert not (root / "new.csv").exists()


def test_input_catalogue_cannot_be_overwritten_even_with_overwrite(generation_inputs):
    root, args, calls, sales = generation_inputs
    source = root / "new.pairs.csv"
    cartesian_pairs(sales).to_csv(source, index=False)
    previous = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        generator.main([*args, "--required-pairs", str(source), "--out", str(root / "new.csv"), "--overwrite"])
    assert not calls and source.read_bytes() == previous


def test_output_cannot_alias_sales_input(generation_inputs):
    root, args, calls, _ = generation_inputs
    previous = (root / "sales.csv").read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        generator.main([*args, "--out", str(root / "sales.csv"), "--overwrite"])
    assert not calls and (root / "sales.csv").read_bytes() == previous


def test_observed_only_remains_an_explicit_mode(generation_inputs):
    root, args, calls, _ = generation_inputs
    output = root / "observed.csv"
    generator.main([*args, "--allow-observed-pairs", "--out", str(output)])
    assert len(pd.read_csv(output)) == 18
    assert all(len(frame) == 2 for frame in calls)


def test_publish_failure_restores_the_entire_previous_bundle(tmp_path, monkeypatch):
    outputs = generator.bundle_paths(tmp_path / "forecast.csv")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    staged = {name: stage_dir / name for name in outputs}
    for name, path in outputs.items():
        path.write_bytes(f"old {name}".encode())
        staged[name].write_bytes(f"new {name}".encode())
    replace = generator.os.replace

    def fail_manifest(source, destination):
        if Path(source) == staged["manifest"]:
            raise OSError("simulated manifest publication failure")
        return replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        generator.publish_bundle(staged, outputs, [], overwrite=True)
    assert {name: path.read_bytes() for name, path in outputs.items()} == {
        name: f"old {name}".encode() for name in outputs
    }
    assert not list(tmp_path.glob(".forecast-backup-*"))


def test_invalid_forecast_never_publishes_partial_files(generation_inputs, monkeypatch):
    root, args, _, _ = generation_inputs
    original = generator.Predictor.predict

    def negative(self, *values, **kwargs):
        return original(self, *values, **kwargs).assign(quantity=-1)

    monkeypatch.setattr(generator.Predictor, "predict", negative)
    with pytest.raises(ValueError, match="nonnegative"):
        generator.main([*args, "--out", str(root / "bad.csv")])
    assert not list(root.glob("bad.*"))


def test_failed_rollback_keeps_original_backup_for_recovery(tmp_path, monkeypatch):
    outputs = generator.bundle_paths(tmp_path / "forecast.csv")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    staged = {name: stage_dir / name for name in outputs}
    for name, path in outputs.items():
        path.write_bytes(f"old {name}".encode())
        staged[name].write_bytes(f"new {name}".encode())
    replace = generator.os.replace

    def fail_publication_and_rollback(source, destination):
        source = Path(source)
        if source == staged["manifest"] or source.name == "forecast" and source.parent != stage_dir:
            raise OSError("simulated file lock")
        return replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_publication_and_rollback)
    with pytest.raises(RuntimeError, match="backups kept"):
        generator.publish_bundle(staged, outputs, [], overwrite=True)
    backups = list(tmp_path.glob(".forecast-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "forecast").read_bytes() == b"old forecast"


def test_input_mutation_during_inference_aborts_before_publication(generation_inputs, monkeypatch):
    root, args, _, _ = generation_inputs
    original = generator.Predictor.predict

    def mutate_source(self, *values, **kwargs):
        result = original(self, *values, **kwargs)
        (root / "model" / "config.json").write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(generator.Predictor, "predict", mutate_source)
    with pytest.raises(ValueError, match="Input changed"):
        generator.main([*args, "--out", str(root / "mutated.csv")])
    assert not list(root.glob("mutated.*"))
    assert not list(root.glob(".forecast-stage-*"))
