"""Generate a checked weekly forecast and its run-specific catalogue/manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import pandas as pd

from app.coverage import cartesian_pairs
from app.csv_io import read_csv_checked
from app.features import COMPETITION_CUTOFF, COMPETITION_FIRST_WEEK, truncate_history
from app.predictor import Predictor
from app.submission import normalized_pairs, validate_forecast
from app.validators import validate_sales_csv, validate_stock_csv


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_paths(output):
    """Every run owns its sidecars; never overwrite a shared required_pairs.csv."""
    output = Path(output)
    return {
        "forecast": output,
        "catalogue": output.with_name(f"{output.stem}.pairs.csv"),
        "manifest": output.with_suffix(".manifest.json"),
    }


def _same_path(first, second):
    first, second = Path(first), Path(second)
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and os.path.samefile(first, second)


def preflight_outputs(outputs, inputs, overwrite=False):
    """Check all destinations, including aliases, before fitting or writing."""
    destinations = list(outputs.values())
    for index, destination in enumerate(destinations):
        if any(_same_path(destination, other) for other in destinations[:index]):
            raise ValueError(f"Output paths collide: {destination}")
        if any(_same_path(destination, source) for source in inputs):
            raise ValueError(f"Output would overwrite an input: {destination}")
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Output is not a regular file: {destination}")
            if not overwrite:
                raise FileExistsError(f"File exists: {destination}; use --overwrite for the whole bundle")


def publish_bundle(staged, outputs, inputs, overwrite=False):
    """Publish the manifest last, with rollback if a replace fails.

    Individual replacements are atomic on the same filesystem. The manifest is
    the completion marker; readers must validate its hashes. A multi-file bundle
    cannot be made crash-atomic by three separate file replacements.
    """
    preflight_outputs(outputs, inputs, overwrite)
    backup_dir = Path(tempfile.mkdtemp(prefix=".forecast-backup-", dir=outputs["forecast"].parent))
    backups = {}
    published = []
    keep_backups = False
    try:
        for name, target in outputs.items():
            if target.exists():
                backup = backup_dir / name
                shutil.copy2(target, backup)
                backups[name] = backup
        for name in ("forecast", "catalogue", "manifest"):
            target = outputs[name]
            # Hard-link publication fails rather than overwriting a file that
            # appeared after preflight without explicit --overwrite.
            if overwrite:
                os.replace(staged[name], target)
                published.append(name)
            else:
                os.link(staged[name], target)
                published.append(name)
                staged[name].unlink()
    except BaseException as exc:
        rollback_errors = []
        for name in reversed(published):
            target = outputs[name]
            try:
                if name in backups:
                    os.replace(backups[name], target)
                else:
                    target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        if rollback_errors:
            keep_backups = True
            raise RuntimeError(
                f"Publication failed and rollback was incomplete; backups kept at {backup_dir}: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        if not keep_backups:
            shutil.rmtree(backup_dir)


def main(argv=None):
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sales", type=Path, default=root / "data/sales_dataset_hakaton_train.csv")
    ap.add_argument("--stock", type=Path, default=root / "data/stock_dataset_hakaton_train.csv")
    ap.add_argument("--model-dir", type=Path, default=root / "model")
    ap.add_argument("--required-pairs", type=Path, help="Supplied catalogue/template; its official origin is not inferred")
    ap.add_argument("--allow-observed-pairs", action="store_true", help="Explicitly generate a PROVISIONAL observed-pair forecast")
    ap.add_argument("--all-combinations", action="store_true", help="Full warehouse x article product; default without a catalogue")
    ap.add_argument("--out", type=Path, default=root / "forecast.csv")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing all existing files belonging to this output bundle")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.required_pairs, a.allow_observed_pairs, a.all_combinations)) > 1:
        ap.error("Choose only one pair-coverage option")
    if not a.required_pairs and not a.allow_observed_pairs:
        a.all_combinations = True

    outputs = bundle_paths(a.out)
    input_paths = [a.sales, a.stock, *sorted(p for p in a.model_dir.glob("*") if p.is_file())]
    if a.required_pairs:
        input_paths.append(a.required_pairs)
    preflight_outputs(outputs, input_paths, a.overwrite)
    started = time.perf_counter()
    # Freeze input provenance before any output directory/file is created.
    input_hashes = {str(p.resolve()): file_sha256(p) for p in input_paths}
    sales, stock = read_csv_checked(a.sales), read_csv_checked(a.stock)
    original_counts = len(sales), len(stock)
    sales, stock = truncate_history(sales, stock, COMPETITION_CUTOFF)
    errors = validate_sales_csv(sales) + validate_stock_csv(stock)
    if errors:
        raise ValueError("; ".join(errors))
    supplied_pairs = normalized_pairs(read_csv_checked(a.required_pairs)) if a.required_pairs else None
    if a.allow_observed_pairs:
        supplied_pairs = normalized_pairs(sales)
    expected_pairs = supplied_pairs if supplied_pairs is not None else cartesian_pairs(sales)
    predictor = Predictor(a.model_dir).load()
    predictor.assert_history_compatible(COMPETITION_CUTOFF)
    begin = time.perf_counter()
    forecast = predictor.predict(
        sales, stock, history_cutoff=COMPETITION_CUTOFF,
        first_week=COMPETITION_FIRST_WEEK, required_pairs=supplied_pairs,
    )
    seconds = time.perf_counter() - begin
    repeat_started = time.perf_counter()
    repeated = predictor.predict(
        sales, stock, history_cutoff=COMPETITION_CUTOFF,
        first_week=COMPETITION_FIRST_WEEK, required_pairs=supplied_pairs,
    )
    if not forecast.equals(repeated):
        raise ValueError("Forecast is not deterministic")
    repeat_seconds = time.perf_counter() - repeat_started
    summary = validate_forecast(forecast, expected_pairs, COMPETITION_FIRST_WEEK)
    observed = pd.MultiIndex.from_frame(normalized_pairs(sales))
    fallback = len(pd.MultiIndex.from_frame(expected_pairs).difference(observed))
    policy = "supplied_catalogue" if a.required_pairs else (
        "full_warehouse_article_product" if a.all_combinations else "provisional_observed_sales_pairs"
    )
    summary.update(
        history_cutoff="2026-07-26", first_week="2026-07-27", coverage_policy=policy,
        catalogue_provided=bool(a.required_pairs),
        catalogue_source=str(a.required_pairs.resolve()) if a.required_pairs else "derived_from_permitted_sales",
        local_key_match=True, official_pair_coverage_verified=False,
        external_verification=None, fallback_pairs=fallback,
        excluded_future_rows={"sales": original_counts[0] - len(sales), "stock": original_counts[1] - len(stock)},
        predict_seconds=seconds, deterministic_repeat_seconds=repeat_seconds,
        model_config=predictor.config,
        coverage_note="Keys match supplied catalogue; official origin/external coverage not verified" if a.required_pairs else (
            "Full Cartesian product; exact external key-set equality pending scorer confirmation" if a.all_combinations else "Observed pairs only"
        ),
        input_sha256=input_hashes,
        output_files={name: str(path.resolve()) for name, path in outputs.items()},
    )
    try:
        import psutil
        mem = psutil.Process().memory_info()
        summary["process_memory_gib"] = getattr(mem, "peak_wset", mem.rss) / 2**30
        summary["memory_kind"] = "peak_working_set" if hasattr(mem, "peak_wset") else "current_rss"
    except ImportError:
        pass

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forecast-stage-", dir=a.out.parent) as directory:
        stage = Path(directory)
        staged = {name: stage / name for name in outputs}
        forecast.to_csv(staged["forecast"], index=False)
        expected_pairs.to_csv(staged["catalogue"], index=False)
        # Validate the actual staged bytes before exposing final pathnames.
        validate_forecast(read_csv_checked(staged["forecast"]), read_csv_checked(staged["catalogue"]), COMPETITION_FIRST_WEEK)
        if any(file_sha256(path) != digest for path, digest in input_hashes.items()):
            raise ValueError("Input changed during forecast generation")
        summary["sha256"] = {
            **input_hashes,
            **{str(outputs[name].resolve()): file_sha256(staged[name]) for name in ("forecast", "catalogue")},
        }
        summary["forecast_sha256"] = file_sha256(staged["forecast"])
        summary["required_pairs_sha256"] = file_sha256(staged["catalogue"])
        summary["total_seconds"] = time.perf_counter() - started
        staged["manifest"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        publish_bundle(staged, outputs, input_paths, a.overwrite)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("model_config", "sha256", "input_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
