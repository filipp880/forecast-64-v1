"""Generate the fixed competition horizon, with explicit reference/provisional coverage."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
from app.features import COMPETITION_CUTOFF, COMPETITION_FIRST_WEEK, KEY, truncate_history
from app.predictor import Predictor
from app.submission import normalized_pairs, validate_forecast
from app.validators import validate_sales_csv, validate_stock_csv
from app.coverage import complete_cartesian_forecast


def main(argv=None):
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sales", type=Path, default=root / "data/sales_dataset_hakaton_train.csv")
    ap.add_argument("--stock", type=Path, default=root / "data/stock_dataset_hakaton_train.csv")
    ap.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    ap.add_argument("--required-pairs", type=Path, help="Official catalogue or submission template with pair columns")
    ap.add_argument("--allow-observed-pairs", action="store_true", help="Explicitly generate a PROVISIONAL observed-pair forecast")
    ap.add_argument("--all-combinations", action="store_true", help="Full warehouse x article product; default without a catalogue")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.required_pairs,a.allow_observed_pairs,a.all_combinations))>1:
        ap.error("Choose only one pair-coverage option")
    if not a.required_pairs and not a.allow_observed_pairs:a.all_combinations=True
    a.out = a.out or root / "forecast.csv"
    if a.out.exists() and not a.overwrite:
        raise FileExistsError(f"File exists: {a.out}")
    started = time.perf_counter()
    dtypes = {"warehouse_name": "string", "article": "string", "char_article": "string"}
    sales, stock = pd.read_csv(a.sales, dtype=dtypes), pd.read_csv(a.stock, dtype=dtypes)
    original_counts = len(sales), len(stock)
    sales, stock = truncate_history(sales, stock, COMPETITION_CUTOFF)
    errors = validate_sales_csv(sales) + validate_stock_csv(stock)
    if errors:
        raise ValueError("; ".join(errors))
    pairs = normalized_pairs(pd.read_csv(a.required_pairs, dtype=dtypes) if a.required_pairs else sales)
    predictor = Predictor(a.model_dir).load()
    predictor.assert_history_compatible(COMPETITION_CUTOFF)
    begin = time.perf_counter()
    forecast = predictor.predict(sales, stock, history_cutoff=COMPETITION_CUTOFF,
                                 first_week=COMPETITION_FIRST_WEEK, required_pairs=pairs)
    seconds = time.perf_counter() - begin
    if not forecast.equals(predictor.predict(sales, stock, history_cutoff=COMPETITION_CUTOFF,
                                            first_week=COMPETITION_FIRST_WEEK, required_pairs=pairs)):
        raise ValueError("Forecast is not deterministic")
    if a.all_combinations:
        forecast,pairs,_=complete_cartesian_forecast(forecast,sales,COMPETITION_CUTOFF,COMPETITION_FIRST_WEEK)
    summary = validate_forecast(forecast, pairs, COMPETITION_FIRST_WEEK)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(a.out, index=False)
    catalogue = a.out.with_name("required_pairs.csv" if a.required_pairs or a.all_combinations else "observed_pairs.csv")
    pairs.to_csv(catalogue, index=False)
    observed = pd.MultiIndex.from_frame(normalized_pairs(sales))
    fallback = len(pd.MultiIndex.from_frame(pairs).difference(observed))
    summary.update(history_cutoff="2026-07-26", first_week="2026-07-27",
                   coverage_policy="official_reference_pairs" if a.required_pairs else ("full_warehouse_article_product" if a.all_combinations else "provisional_observed_sales_pairs"),
                   official_pair_coverage_verified=bool(a.required_pairs), fallback_pairs=fallback,
                   excluded_future_rows={"sales": original_counts[0] - len(sales), "stock": original_counts[1] - len(stock)},
                   predict_seconds=seconds, total_seconds=time.perf_counter() - started,
                   model_config=predictor.config,
                   coverage_note="Official catalogue supplied" if a.required_pairs else
                   ("Full Cartesian product; exact external key-set equality pending scorer confirmation" if a.all_combinations else "Observed pairs only"))
    try:
        import psutil
        mem = psutil.Process().memory_info()
        summary["process_memory_gib"] = getattr(mem, "peak_wset", mem.rss) / 2**30
        summary["memory_kind"] = "peak_working_set" if hasattr(mem, "peak_wset") else "current_rss"
    except ImportError:
        pass
    paths = [a.sales, a.stock, a.out, catalogue, *a.model_dir.glob("*")]
    if a.required_pairs:
        paths.append(a.required_pairs)
    summary["sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    summary["forecast_sha256"] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    a.out.with_suffix(".manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("model_config", "sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
