"""Small deterministic checks for the descriptive weekday transfer script."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.features import CAT_FEATURES, preprocessing_spec
from app.predictor import Predictor
from check_api_weekdays import evaluate_weekdays, normalize_origins


ORIGINS = pd.date_range("2026-02-16", periods=7, freq="D").strftime("%Y-%m-%d").tolist()


class FrozenMeanModel:
    target_transform = "none"

    def __init__(self):
        self.seen = []

    def fit(self, *args, **kwargs):
        raise AssertionError("Weekday checking must not fit or select a model")

    def predict(self, features):
        self.seen.append(features.copy())
        return 7 * features.mean_28.to_numpy(float)


@pytest.fixture
def setup(tmp_path):
    dates = pd.date_range("2025-08-10", "2026-04-26", freq="D")
    sales = pd.DataFrame({c: "x" for c in ["composite_division", "cities_name", *CAT_FEATURES]}, index=range(len(dates)))
    sales["date"], sales["quantity"] = dates, 1.0
    sales["sales_sum"], sales["discount_sum"] = 1.0, np.nan
    stock = pd.DataFrame({"date": dates, "warehouse_name": "x", "article": "x", "stock": 3.0})
    predictor = Predictor(tmp_path)
    predictor.model = FrozenMeanModel()
    predictor.config = {
        **preprocessing_spec(), "target_transform": "none",
        "training_data_end": "2026-02-15", "selection_data_end": "2026-02-15",
        "postprocess": {"zero_threshold": 0.0, "round_int": False},
    }
    return predictor, sales, stock


def test_weekday_report_uses_same_truth_and_frozen_model(setup):
    predictor, sales, stock = setup
    before_sales, before_stock = sales.copy(deep=True), stock.copy(deep=True)
    before_config = json.dumps(predictor.config, sort_keys=True)
    report = evaluate_weekdays(predictor, sales, stock, ORIGINS)
    assert report["common_first_week"] == "2026-02-23"
    assert report["common_last_day"] == "2026-04-26"
    assert report["truth_windows_overlap"]
    assert not any(report[key] for key in ("calibration_used", "refit_performed", "selection_performed", "promotion_performed"))
    assert [record["weekday"] for record in report["by_origin"].values()] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for record in report["by_origin"].values():
        assert record["rows"] == 9 and record["pairs"] == 1
        assert record["actual_total"] == record["prediction_total"] == 63
        assert all(value == 0 for value in record["metrics"].values())
    assert len(predictor.model.seen) == 7
    pd.testing.assert_frame_equal(sales, before_sales)
    pd.testing.assert_frame_equal(stock, before_stock)
    assert json.dumps(predictor.config, sort_keys=True) == before_config


def test_each_weekday_features_use_only_that_input_prefix(setup):
    predictor, sales, stock = setup
    sales.loc[sales.date == "2026-02-17", "quantity"] = 9
    stock.loc[stock.date == "2026-02-20", "stock"] = -6
    report = evaluate_weekdays(predictor, sales, stock, ORIGINS)
    first, second, friday = predictor.model.seen[0], predictor.model.seen[1], predictor.model.seen[4]
    assert first.last_value.eq(1).all()
    assert second.last_value.eq(9).all()
    assert first.stock_last.eq(3).all()
    assert friday.stock_last.eq(-6).all()
    assert report["by_origin"]["2026-02-16"]["metrics"]["wape"] == 0
    assert report["by_origin"]["2026-02-17"]["metrics"]["wape"] > 0


def test_pair_catalogue_is_fixed_before_later_origins(setup):
    predictor, sales, stock = setup
    future_pair = sales.loc[sales.date == "2026-02-18"].copy()
    future_pair["article"] = "new_on_wednesday"
    sales = pd.concat([sales, future_pair], ignore_index=True)
    report = evaluate_weekdays(predictor, sales, stock, ORIGINS)
    assert report["pairs"] == 1
    assert all(set(features.article) == {"x"} for features in predictor.model.seen)


@pytest.mark.parametrize("field", ["training_data_end", "selection_data_end"])
def test_artifact_must_not_see_facts_after_earliest_origin(setup, field):
    predictor, sales, stock = setup
    predictor.config[field] = "2026-02-17"
    with pytest.raises(ValueError, match=field):
        evaluate_weekdays(predictor, sales, stock, ORIGINS)
    assert not predictor.model.seen


def test_selected_calibration_is_not_allowed_in_descriptive_raw_check(setup):
    predictor, sales, stock = setup
    predictor.config["calibration"] = {"scale": 1.1}
    with pytest.raises(ValueError, match="raw artifact"):
        evaluate_weekdays(predictor, sales, stock, ORIGINS)


def test_incomplete_ninth_truth_week_is_rejected_before_inference(setup):
    predictor, sales, stock = setup
    with pytest.raises(ValueError, match="final Sunday"):
        evaluate_weekdays(predictor, sales, stock, ORIGINS, history_cutoff="2026-04-25")
    assert not predictor.model.seen


@pytest.mark.parametrize("origins", [ORIGINS[:6], ORIGINS[::-1], [*ORIGINS[:6], ORIGINS[5]], [str((pd.Timestamp(origin) + pd.Timedelta(days=1)).date()) for origin in ORIGINS]])
def test_only_one_complete_calendar_week_of_origins_is_accepted(origins):
    with pytest.raises(ValueError):
        normalize_origins(origins)


def test_missing_asof_sales_day_does_not_silently_change_api_origin(setup):
    predictor, sales, stock = setup
    sales = sales.loc[sales.date != "2026-02-16"]
    with pytest.raises(ValueError, match="different origin"):
        evaluate_weekdays(predictor, sales, stock, ORIGINS)


def test_cutoff_later_than_competition_is_rejected(setup):
    predictor, sales, stock = setup
    with pytest.raises(ValueError, match="2026-07-26"):
        evaluate_weekdays(predictor, sales, stock, ORIGINS, history_cutoff="2026-07-31")
