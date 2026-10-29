"""Cross-entry-point regressions on real bundled weights and small fresh fits."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.coverage import cartesian_pairs
from app.csv_io import read_csv_checked
from app.features import CAT_FEATURES, KEY
from app.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]


def histories(start="2026-01-28", end="2026-07-26"):
    dates = pd.date_range(start, end)
    frames = []
    for i in range(2):
        f = pd.DataFrame({"date": dates, "quantity": np.where(np.arange(len(dates)) % 3 == 0, .5, 0.) if i == 0 else 3.})
        for c in ["composite_division", "cities_name", *CAT_FEATURES]:
            f[c] = "x"
        f["warehouse_name"] = f"warehouse_name_{i}"
        f["article"] = f"char_article_{i}"
        f["sales_sum"], f["discount_sum"] = 1., np.nan
        frames.append(f)
    sales = pd.concat(frames, ignore_index=True)
    return sales, sales[["date", *KEY]].assign(stock=100.)


def test_real_model_warm_predictions_independent_of_requested_catalogue():
    sales, stock = histories()
    predictor = Predictor(ROOT / "model").load()
    default = predictor.predict(sales, stock)
    explicit = predictor.predict(sales, stock, required_pairs=cartesian_pairs(sales))
    pd.testing.assert_frame_equal(default, explicit)
    subset = sales[KEY].drop_duplicates().iloc[:1]
    partial = predictor.predict(sales, stock, required_pairs=subset)
    pd.testing.assert_frame_equal(partial, default.merge(subset, on=KEY).reset_index(drop=True))


def test_unused_money_does_not_break_current_dynamics_model():
    sales, stock = histories()
    predictor = Predictor(ROOT / "model").load()
    expected = predictor.predict(sales, stock)
    sales["sales_sum"] = 1000.
    pd.testing.assert_frame_equal(expected, predictor.predict(sales, stock))


def test_shared_loader_preserves_leading_zero_ids():
    data = read_csv_checked(io.StringIO("warehouse_name,article,quantity\n001,002,1\n"))
    assert data.warehouse_name.iloc[0] == "001"
    assert data.article.iloc[0] == "002"


def test_explicit_diagnostic_origin_does_not_move_to_last_sparse_sale():
    sales, stock = histories()
    predictor = Predictor(ROOT / "model").load()
    sales = sales[sales.date <= "2026-07-12"]
    origin = "2026-07-19"
    with pytest.raises(ValueError, match="training_data_end"):
        predictor.predict(sales, stock, history_cutoff=origin)
    result = predictor.predict(sales, stock, history_cutoff=origin, allow_in_sample=True)
    assert result.week.min() == "2026-07-20"


def test_full_coverage_ignores_future_only_identifiers():
    from app.coverage import complete_cartesian_forecast
    sales, stock = histories()
    predictor = Predictor(ROOT / "model").load()
    observed = sales[KEY].drop_duplicates()
    forecast = predictor.predict(sales, stock, required_pairs=observed)
    future = sales.iloc[:1].assign(date=pd.Timestamp("2026-07-27"), article="future", warehouse_name="future")
    extended = pd.concat([sales, future], ignore_index=True)
    result, pairs, _ = complete_cartesian_forecast(forecast, extended, "2026-07-26", "2026-07-27")
    assert len(pairs) == 4 and len(result) == 36
    assert "future" not in set(pairs.article)


def test_extended_model_seed_is_not_silently_misreported():
    from train import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--model-type", "research", "--seed", "123"])


@pytest.mark.parametrize("model_type", ["research", "ideas"])
def test_extended_training_cli_roundtrip_and_evaluation(tmp_path, model_type):
    from train import main as train_main
    from evaluate import main as evaluate_main
    sales, stock = histories("2024-10-01")
    sales_path, stock_path = tmp_path / "sales.csv", tmp_path / "stock.csv"
    sales.to_csv(sales_path, index=False)
    stock.to_csv(stock_path, index=False)
    target = tmp_path / "model"
    assert train_main(["--sales", str(sales_path), "--stock", str(stock_path),
                       "--out-dir", str(target), "--model-type", model_type,
                       "--estimators", "2", "--threads", "1", "--max-cutoffs", "2",
                       "--cutoff-step", "84"]) == 0
    loaded = Predictor(target).load()
    assert len(loaded.predict(sales, stock)) == 36
    # Deliberately diagnostic; the saved final artifact has seen these dates.
    assert evaluate_main(["--sales", str(sales_path), "--stock", str(stock_path),
                          "--model-dir", str(target), "--origins", "2026-05-24",
                          "--allow-in-sample", "--out", str(tmp_path / "evaluation.json")]) == 0
