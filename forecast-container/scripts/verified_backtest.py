#!/usr/bin/env python3
"""Two new, past-only nine-week windows; selection never reads confirmation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import HORIZON, KEY, build_panel, forecast_end, make_features, make_training_frame, truncate_history
from app.metrics import smape, wape, rmse, bias_abs, rmsle
from app.models import get_model_class
from baseline import METHODS, baseline_forecast
from train import build_dataset, compact_params, make_cutoffs, save_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    """Hash large CSVs in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ml_source_paths(project_root=None):
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    container = root / "forecast-container"
    fixed = [container / name for name in (
        "train.py", "baseline.py", "evaluate.py", "scripts/verified_backtest.py", "scripts/select_candidate.py")]
    for path in fixed:
        if not path.is_file():
            raise ValueError(f"Missing ML source: {path}")
    return sorted(set(fixed + list((container / "app").glob("*.py")) +
                      list((container / "app/models").glob("*.py"))))


def _record_path(path, root):
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_record(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def freeze_fingerprints(sales, stock, project_root=None):
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return {
        "algorithm": "sha256",
        "inputs": {name: {"path": _record_path(path, root), "sha256": file_sha256(path)}
                   for name, path in (("sales", sales), ("stock", stock))},
        "sources": {_record_path(path, root): file_sha256(path) for path in ml_source_paths(root)},
    }


def verify_protocol_fingerprints(protocol, project_root=None):
    """Legacy protocols and changed inputs/code must never resume silently."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    frozen = protocol.get("fingerprints")
    if not isinstance(frozen, dict) or frozen.get("algorithm") != "sha256":
        raise ValueError("Protocol lacks frozen SHA-256 fingerprints; use a new output directory")
    inputs = frozen.get("inputs", {})
    if set(inputs) != {"sales", "stock"} or not frozen.get("sources"):
        raise ValueError("Protocol has incomplete input/source fingerprints")
    try:
        current = freeze_fingerprints(*[_resolve_record(inputs[k]["path"], root) for k in ("sales", "stock")], root)
    except (KeyError, OSError) as exc:
        raise ValueError(f"Cannot verify frozen inputs/sources: {exc}") from exc
    for name in ("sales", "stock"):
        if current["inputs"][name]["sha256"] != inputs[name].get("sha256"):
            raise ValueError(f"Input SHA-256 changed: {name}")
    if current["sources"] != frozen["sources"]:
        changed = sorted(set(current["sources"]) ^ set(frozen["sources"]) |
                         {p for p in current["sources"] if p in frozen["sources"] and current["sources"][p] != frozen["sources"][p]})
        raise ValueError(f"ML source SHA-256 changed: {changed}")
    return True


def verify_fold_artifacts(fold, protocol_path):
    """A completion marker is trusted only with intact OOF and loadable weights."""
    fold, protocol_path = Path(fold), Path(protocol_path)
    try:
        record = json.loads((fold / "complete.json").read_text(encoding="utf-8"))
        config = json.loads((fold / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete fold {fold}: {exc}") from exc
    required = ("predictions_sha256", "model_sha256", "config_sha256", "preprocessor_sha256", "protocol_sha256")
    if any(not record.get(key) for key in required):
        raise ValueError(f"Fold lacks frozen artifact SHA-256 values: {fold}")
    if record["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError(f"Fold protocol SHA-256 mismatch: {fold}")
    declared = {"predictions_sha256": "predictions.csv.gz", "config_sha256": "config.json",
                "model_sha256": config.get("model_file", "model.pkl"),
                "preprocessor_sha256": config.get("preprocessor_file", "preprocessor.pkl")}
    for key, filename in declared.items():
        path = (fold / filename).resolve()
        if path.parent != fold.resolve():
            raise ValueError("Artifact filenames must refer to the same fold directory")
        try:
            digest = file_sha256(path)
        except OSError as exc:
            raise ValueError(f"Missing artifact {path}") from exc
        if record[key] != digest:
            raise ValueError(f"Fold artifact SHA-256 mismatch: {filename}")
    if config.get("model_artifact_sha256") != record["model_sha256"] or config.get("preprocessor_sha256") != record["preprocessor_sha256"]:
        raise ValueError("Artifact hashes do not agree with model config")
    if config.get("provenance", {}).get("protocol_sha256") != record["protocol_sha256"]:
        raise ValueError("Model config belongs to a different frozen protocol")
    if record.get("origin") != fold.name:
        raise ValueError("Fold origin does not agree with its directory")
    origin = pd.Timestamp(record["origin"])
    if pd.isna(origin):
        raise ValueError("Fold origin is not a valid date")
    for name in ("training_data_end", "selection_data_end"):
        if not config.get(name) or pd.isna(pd.Timestamp(config[name])) or pd.Timestamp(config[name]) > origin:
            raise ValueError(f"Fold has incompatible temporal provenance: {name}")
    from app.predictor import Predictor
    Predictor(fold).load()
    return record


def verify_run_integrity(out, protocol=None):
    out = Path(out)
    protocol_path = out / "protocol.json"
    if protocol is None:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol_fingerprints(protocol)
    if not protocol.get("candidate_grid_sha256") or protocol["candidate_grid_sha256"] != file_sha256(out / "candidate_grid.json"):
        raise ValueError("Frozen candidate grid SHA-256 mismatch")
    return {origin: verify_fold_artifacts(out / origin, protocol_path) for origin in protocol["origins"]}


def metrics(y, p, ids):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError("Metrics require finite nonnegative values")
    return dict(smape=smape(y, p, ids), wape=wape(y, p), rmse=rmse(y, p),
                bias=bias_abs(y, p, ids), rmsle=rmsle(y, p),
                zero_share=float((p == 0).mean()),
                volume_bias=float((p.sum() / y.sum() - 1) * 100) if y.sum() > 0 else None)


def peak_memory_gib():
    try:
        import psutil
        mem = psutil.Process().memory_info()
        return float(getattr(mem, "peak_wset", mem.rss) / 2 ** 30)
    except ImportError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Past-only weekly blend/log1p/hurdle candidate backtest")
    ap.add_argument("--sales", type=Path, default=Path("../data/sales_dataset_hakaton_train.csv"))
    ap.add_argument("--stock", type=Path, default=Path("../data/stock_dataset_hakaton_train.csv"))
    ap.add_argument("--out", type=Path, default=Path("../artifacts/weekly"))
    ap.add_argument("--history-cutoff", default="2026-07-26")
    ap.add_argument("--origins", nargs=2, default=["2026-02-15", "2026-05-24"])
    ap.add_argument("--max-cutoffs", type=int, default=16)
    ap.add_argument("--estimators", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    args = ap.parse_args(argv)
    if pd.Timestamp(args.history_cutoff) > pd.Timestamp("2026-07-26"):
        ap.error("Competition history cannot exceed 2026-07-26")
    if args.max_cutoffs <= 0 or args.estimators <= 0 or args.threads <= 0:
        ap.error("Positive max-cutoffs, estimators and threads are required")
    origins = [pd.Timestamp(o) for o in args.origins]
    if any(o.weekday() != 6 for o in origins) or origins[0] >= origins[1]:
        ap.error("Origins must be increasing Sundays")
    if forecast_end(origins[0], HORIZON) > origins[1]:
        ap.error("Selection and confirmation target windows must not overlap")
    params = {"seed": args.seed, "threads": args.threads, **compact_params(args.estimators)}
    protocol = {
        "schema_version": 3, "granularity": "week", "forecast_weeks": 9,
        "origins": args.origins, "history_cutoff": args.history_cutoff,
        "selection": "first_origin_only", "confirmation": "selected_candidate_only_second_origin",
        "stock_aggregation": "sum", "negative_stock_policy": "preserve",
        "recipe": params, "max_cutoffs": args.max_cutoffs,
        "early_stopping_rounds": args.early_stopping_rounds,
        "normalization": "local_baseline_minmax; not the official competition leaderboard",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    from select_candidate import initialize_grid
    initialize_grid(args.out)
    protocol_path = args.out / "protocol.json"
    previous = None
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        verify_protocol_fingerprints(previous)
    protocol.update(
        fingerprints=freeze_fingerprints(args.sales, args.stock),
        candidate_grid_sha256=file_sha256(args.out / "candidate_grid.json"),
        frozen_at=previous["frozen_at"] if previous else datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if previous is not None and previous != protocol:
        raise ValueError("Output directory contains a different frozen protocol")
    if previous is None:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    frozen_protocol_sha = file_sha256(protocol_path)
    completed = [args.out / str(cut.date()) for cut in origins if (args.out / str(cut.date()) / "complete.json").exists()]
    for fold in completed:
        verify_fold_artifacts(fold, protocol_path)
    if len(completed) == len(origins):
        print("All frozen folds verified; no training or CSV reload required", flush=True)
        return 0
    sales, stock = truncate_history(pd.read_csv(args.sales, parse_dates=["date"]),
                                    pd.read_csv(args.stock, parse_dates=["date"]), args.history_cutoff)
    if any(forecast_end(o, 9) > sales.date.max() for o in origins):
        raise ValueError("Data do not contain all nine complete truth weeks")
    daily = sales.groupby(["date", *KEY], as_index=False).quantity.sum()
    daily["week"] = daily.date - pd.to_timedelta(daily.date.dt.weekday, unit="D")
    weekly = daily.groupby(["week", *KEY], as_index=False).quantity.sum()
    del daily
    for cut in origins:
        fold = args.out / str(cut.date())
        fold.mkdir(exist_ok=True)
        verify_protocol_fingerprints(protocol)
        if file_sha256(protocol_path) != frozen_protocol_sha:
            raise ValueError("Frozen protocol file changed during the run")
        if (fold / "complete.json").exists():
            verify_fold_artifacts(fold, protocol_path)
            print(f"Saved fold: {fold}", flush=True)
            continue
        started = time.perf_counter()
        # Slice before constructing attributes, the assortment and the stock panel.
        past_sales, past_stock = truncate_history(sales, stock, cut)
        panel = build_panel(past_sales, past_stock)
        del past_sales, past_stock
        gc.collect()
        selection = cut - pd.Timedelta(weeks=9)
        cutoffs = make_cutoffs(panel, 180, 9, 28)
        train_cuts = [c for c in cutoffs if forecast_end(c, 9) <= selection][-args.max_cutoffs:]
        frames = build_dataset(panel, train_cuts, 9, 180)
        train = pd.concat(frames, ignore_index=True)
        del frames
        valid = make_training_frame(panel, selection, range(1, 10))
        if (train.week + pd.Timedelta(days=6)).max() > selection:
            raise RuntimeError("Training labels overlap early stopping labels")
        model = get_model_class("blend")(params)
        print(f"Early stopping {cut.date()}: {len(train):,} rows", flush=True)
        model.fit(train, train.target.to_numpy(float), valid, valid.target.to_numpy(float), args.early_stopping_rounds)
        iterations = model.selected_iterations
        del train, valid
        gc.collect()
        refit_cuts = cutoffs[-args.max_cutoffs:]
        frames = build_dataset(panel, refit_cuts, 9, 180)
        refit = pd.concat(frames, ignore_index=True)
        del frames
        rows = len(refit)
        label_end = (refit.week + pd.Timedelta(days=6)).max()
        if label_end > cut:
            raise RuntimeError("Refit saw future truth")
        print(f"Past-only refit {cut.date()}: {rows:,} rows", flush=True)
        model.fit(refit, refit.target.to_numpy(float), n_iterations=iterations)
        del refit
        gc.collect()
        save_artifact(model, fold, cut, cut, rows, len(refit_cuts),
                      validation={"scheme": "past_only_weekly", "early_stopping_origin": str(selection.date()), "iterations": iterations},
                      provenance={"protocol": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
                                  "history_cutoff": str(cut.date()), "label_end": str(label_end.date())})
        features = make_features(panel, cut, range(1, 10))
        encoded = model.hurdle._encode(features, False)
        kwargs = {"num_threads": args.threads}
        probability = model.hurdle.classifier.predict(encoded, num_iteration=model.hurdle._best_iterations["classifier"], **kwargs)
        magnitude = model.hurdle.regressor.predict(encoded, num_iteration=model.hurdle._best_iterations["regressor"], **kwargs)
        lgbm_raw = np.maximum(np.expm1(model.lgbm.predict(features)), 0.0)
        result = features[["week", *KEY, "h", "mean_28", "n_sale_days_90", "category_lvl_1"]].copy()
        result["origin"] = str(cut.date())
        result["probability"], result["magnitude"], result["lgbm_raw"] = probability, magnitude, lgbm_raw
        result = result.merge(weekly, on=["week", *KEY], how="left", validate="one_to_one")
        # No transaction means zero actual sales, unlike a missing submitted prediction.
        result["actual"] = result.pop("quantity").fillna(0.0)
        for method in METHODS:
            base = baseline_forecast(panel, cut, method, 9, result[KEY].drop_duplicates())
            result = result.merge(base.rename(columns={"quantity": f"baseline_{method}"}), on=["week", *KEY], how="left", validate="one_to_one")
        if result.filter(like="baseline_").isna().any().any():
            raise RuntimeError("Baseline keys do not align")
        result.to_csv(fold / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        ids = pd.MultiIndex.from_frame(result[KEY]).to_numpy()
        raw = 0.3 * np.where(probability >= 0.5, magnitude, 0.0) + 0.7 * lgbm_raw
        summary = {"origin": str(cut.date()), "rows": len(result), "pairs": len(result[KEY].drop_duplicates()),
                   "evaluation_kind": "out_of_sample", "iterations": iterations,
                   "seconds": round(time.perf_counter() - started, 3), "peak_process_gib": peak_memory_gib(),
                   "incumbent": metrics(result.actual.to_numpy(float), raw, ids),
                   "model_sha256": file_sha256(fold / "model.pkl"),
                   "predictions_sha256": file_sha256(fold / "predictions.csv.gz"),
                   "config_sha256": file_sha256(fold / "config.json"),
                   "preprocessor_sha256": file_sha256(fold / "preprocessor.pkl"),
                   "protocol_sha256": file_sha256(protocol_path),
                   "past_only_assertions": True}
        verify_protocol_fingerprints(protocol)
        if file_sha256(protocol_path) != frozen_protocol_sha:
            raise ValueError("Frozen protocol file changed during training")
        (fold / "complete.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        del panel, features, result, model, encoded
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
