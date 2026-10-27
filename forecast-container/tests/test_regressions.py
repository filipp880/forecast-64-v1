"""Weekly pipeline regressions using synthetic histories and fresh artifacts."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.features import (
    CAT_FEATURES, FEATURE_COLUMNS, KEY, build_panel, make_features,
    make_training_frame, monetary_weekly_features, next_week_start,
    preprocessing_spec, truncate_history,
)
from app.models import get_model_class, inverse_target, transform_target
from app.predictor import ForecastNotReadyError, ModelNotReadyError, Predictor
from app.temporal import research_split
from app.validators import validate_sales_csv, validate_stock_csv


def make_inputs(n=180, start="2025-01-01"):
    dates = pd.date_range(start, periods=n)
    sales = pd.DataFrame({c: "x" for c in ["composite_division", "cities_name", *CAT_FEATURES]}, index=range(n))
    sales["date"] = dates.strftime("%Y-%m-%d")
    sales["quantity"] = 1.0
    sales["sales_sum"] = 1.0
    sales["discount_sum"] = np.nan
    stock = pd.DataFrame({"date": sales.date, "warehouse_name": "x", "article": "x", "stock": 3.0})
    return sales, stock


@pytest.fixture
def inputs():
    return make_inputs()


class FakeWeeklyModel:
    target_transform = "none"

    def predict(self, features):
        # Transparent output makes date/aggregation/cutoff mistakes observable.
        return features.mean_week_4.to_numpy(dtype=float)


@pytest.fixture
def predictor(tmp_path):
    instance = Predictor(tmp_path)
    instance.model = FakeWeeklyModel()
    instance.config = {
        **preprocessing_spec(), "target_transform": "none",
        "training_data_end": "2024-12-31", "selection_data_end": "2024-12-31",
        "postprocess": {"zero_threshold": 0.0, "round_int": False},
    }
    return instance


@pytest.mark.parametrize("value", ["bad", np.nan, np.inf, -np.inf, -1])
def test_bad_quantity_rejected(inputs, value):
    sales, _ = inputs
    sales["quantity"] = sales.quantity.astype(object)
    sales.loc[0, "quantity"] = value
    assert validate_sales_csv(sales)


@pytest.mark.parametrize("column", ["date", "warehouse_name", "article"])
def test_empty_sales_key_rejected(inputs, column):
    sales, _ = inputs
    sales.loc[0, column] = None
    assert validate_sales_csv(sales)


def test_null_discount_and_missing_old_optional_fields_are_accepted(inputs):
    sales, _ = inputs
    assert sales.discount_sum.isna().all()
    assert "cost_price" not in sales and "volume_weight" not in sales
    assert validate_sales_csv(sales) == []


def test_exact_history_boundary_and_sparse_dates(inputs):
    sales, stock = inputs
    assert not validate_sales_csv(sales)
    assert not validate_stock_csv(stock)
    assert validate_sales_csv(sales.iloc[1:])
    assert validate_stock_csv(stock.iloc[1:])
    assert not validate_sales_csv(sales.iloc[[0, -1]])


@pytest.mark.parametrize("value", ["bad", np.nan, np.inf, -np.inf])
def test_bad_stock_rejected(inputs, value):
    _, stock = inputs
    stock["stock"] = stock.stock.astype(object)
    stock.loc[0, "stock"] = value
    assert validate_stock_csv(stock)


def test_stock_alias_supported_and_conflicts_rejected(inputs):
    _, stock = inputs
    assert not validate_stock_csv(stock.rename(columns={"article": "char_article"}))
    stock["char_article"] = "different"
    assert validate_stock_csv(stock)


def test_signed_stock_duplicates_are_summed_before_features(inputs):
    sales, stock = inputs
    stock.loc[0, "stock"] = -4.0
    stock.loc[1, "stock"] = 5.0
    duplicate = stock.iloc[[1]].copy()
    duplicate["stock"] = -2.0
    stock = pd.concat([stock, duplicate], ignore_index=True)
    assert validate_stock_csv(stock) == []
    panel = build_panel(sales, stock)
    assert panel.loc[panel.date == pd.Timestamp(sales.date.iloc[0]), "stock"].iloc[0] == -4.0
    assert panel.loc[panel.date == pd.Timestamp(sales.date.iloc[1]), "stock"].iloc[0] == 3.0
    assert panel.quantity.sum() == 180.0  # reservations do not invalidate sales
    cut = pd.Timestamp(sales.date.iloc[0])
    features = make_features(panel, cut, [1])
    assert features.stock_last.iloc[0] == -4
    assert features.stock_reserved.iloc[0] == 4
    assert features.days_of_cover.iloc[0] == 0


def test_department_and_city_sales_aggregate_to_pair_day(inputs):
    sales, stock = inputs
    extra = sales.iloc[:1].copy()
    extra["composite_division"] = "another_department"
    extra["cities_name"] = "another_city"
    extra["quantity"] = 2.5
    panel = build_panel(pd.concat([sales, extra], ignore_index=True), stock)
    assert panel.loc[panel.date == pd.Timestamp(sales.date.iloc[0]), "quantity"].iloc[0] == 3.5


@pytest.mark.parametrize("weekday", range(7))
def test_api_next_monday_for_every_last_weekday(predictor, weekday):
    last_date = pd.Timestamp("2026-07-27") + pd.Timedelta(days=weekday)
    sales, stock = make_inputs(start=last_date - pd.Timedelta(days=179))
    result = predictor.predict(sales, stock)
    expected_start = pd.Timestamp("2026-08-03")
    assert next_week_start(last_date) == expected_start
    assert list(result.columns) == ["week", *KEY, "quantity"]
    assert len(result) == 9
    assert result.week.tolist() == pd.date_range(expected_start, periods=9, freq="7D").strftime("%Y-%m-%d").tolist()
    assert result.quantity.tolist() == [7.0] * 9


def test_competition_cutoff_excludes_both_sources_and_future_attributes(predictor):
    sales, stock = make_inputs(n=210, start="2026-01-03")
    before = predictor.predict(sales, stock, history_cutoff="2026-07-26", first_week="2026-07-27")
    future_sales = sales.iloc[:1].copy()
    future_sales["date"] = "2026-07-31"
    future_sales["quantity"] = 10_000
    future_sales["category_lvl_1"] = "future_category"
    future_stock = stock.iloc[:1].copy()
    future_stock["date"] = "2026-07-31"
    future_stock["stock"] = -10_000
    future_pair = future_sales.copy()
    future_pair["article"] = "future_article"
    after = predictor.predict(
        pd.concat([sales, future_sales, future_pair], ignore_index=True),
        pd.concat([stock, future_stock], ignore_index=True),
        history_cutoff="2026-07-26", first_week="2026-07-27",
    )
    pd.testing.assert_frame_equal(before, after)
    assert after.week.iloc[0] == "2026-07-27"
    assert set(after.article) == {"x"}


def test_explicit_cutoff_extends_sparse_history_to_origin(predictor):
    sales, stock = make_inputs(n=180, start="2026-01-01")
    # Last recorded sale is 29 June; remaining calendar days to origin are zeros.
    result = predictor.predict(sales, stock, history_cutoff="2026-07-26", first_week="2026-07-27")
    assert result.week.iloc[0] == "2026-07-27"
    assert result.quantity.iloc[0] == pytest.approx(0.25)


def test_late_model_provenance_is_rejected_for_competition(predictor, inputs):
    predictor.config["training_data_end"] = "2026-07-31"
    with pytest.raises(ValueError, match="training_data_end"):
        predictor.predict(*inputs, history_cutoff="2026-07-26", first_week="2026-07-27")
    predictor.config["training_data_end"] = "2026-07-26"
    predictor.config["selection_data_end"] = "2026-07-31"
    with pytest.raises(ValueError, match="selection_data_end"):
        predictor.predict(*inputs, history_cutoff="2026-07-26", first_week="2026-07-27")


def test_weekly_target_is_seven_day_sum_and_partial_week_is_rejected():
    sales, stock = make_inputs(n=300)
    dates = pd.to_datetime(sales.date)
    sales["quantity"] = dates.dt.weekday + 1.0
    panel = build_panel(sales, stock)
    frame = make_training_frame(panel, pd.Timestamp("2025-07-20"), [1, 2, 9])
    assert frame.week.tolist() == list(pd.to_datetime(["2025-07-21", "2025-07-28", "2025-09-15"]))
    assert frame.target.tolist() == [28.0] * 3
    with pytest.raises(ValueError, match="полных"):
        make_training_frame(panel.loc[panel.date < "2025-09-21"], pd.Timestamp("2025-07-20"), [9])


def test_future_sales_stock_attributes_and_catalog_do_not_change_snapshot(inputs):
    sales, stock = inputs
    cut = pd.Timestamp(sales.date.iloc[-5])
    before = make_features(build_panel(sales, stock), cut)
    future = pd.to_datetime(sales.date) > cut
    sales.loc[future, "quantity"] = 999
    sales.loc[future, "category_lvl_1"] = "future_category"
    sales.loc[future, "article"] = "future_product"
    stock.loc[pd.to_datetime(stock.date) > cut, "stock"] = -999
    after = make_features(build_panel(sales, stock), cut)
    pd.testing.assert_frame_equal(before, after)
    assert "future_product" not in set(after.article)


def test_old_inventory_is_not_carried_into_missing_api_window():
    sales, stock = make_inputs(n=220)
    cut = pd.Timestamp(sales.date.iloc[-1])
    stock = stock.iloc[:1].copy()
    full = make_features(build_panel(sales, stock), cut, [1])
    recent = make_features(build_panel(sales.iloc[-180:], stock.iloc[:0]), cut, [1])
    assert np.isnan(full.stock_last.iloc[0])
    assert np.isnan(recent.stock_last.iloc[0])


def test_all_available_history_affects_long_context_only():
    sales, stock = make_inputs(n=400)
    sales.loc[:219, "quantity"] = 10
    cut = pd.Timestamp(sales.date.iloc[-1])
    full = make_features(build_panel(sales, stock), cut, [1])
    recent = make_features(build_panel(sales.iloc[-180:], stock.iloc[-180:]), cut, [1])
    assert full.mean_all.iloc[0] == pytest.approx(5.95)
    assert recent.mean_all.iloc[0] == 1
    assert full.mean_28.iloc[0] == recent.mean_28.iloc[0] == 1


def test_required_pair_without_history_gets_explicit_fallback(predictor, inputs):
    sales, stock = inputs
    pairs = pd.DataFrame({"warehouse_name": ["x", "new_warehouse"], "article": ["x", "x"]})
    features = make_features(build_panel(sales, stock), pd.Timestamp(sales.date.iloc[-1]), required_pairs=pairs)
    missing = features.loc[features.warehouse_name == "new_warehouse"]
    assert len(missing) == 9
    assert missing.no_history.eq(1).all()
    assert missing.mean_week_4.eq(7).all()
    result = predictor.predict(sales, stock, required_pairs=pairs)
    assert len(result) == 18
    assert result.loc[result.warehouse_name == "new_warehouse", "quantity"].eq(7).all()


def test_evaluation_and_api_share_no_history_fallback(predictor, inputs):
    from train import predictions_for

    class ConstantModel:
        target_transform = "none"

        def predict(self, features):
            return np.full(len(features), 123.0)

    predictor.model = ConstantModel()
    sales, stock = inputs
    pairs = pd.DataFrame({"warehouse_name": ["x", "new_warehouse"], "article": ["x", "x"]})
    features = make_features(build_panel(sales, stock), pd.Timestamp(sales.date.iloc[-1]), required_pairs=pairs)
    evaluated = features[["week", *KEY]].copy()
    evaluated["quantity"] = predictions_for(predictor.model, features)
    evaluated["week"] = evaluated.week.dt.strftime("%Y-%m-%d")
    evaluated = evaluated.sort_values(["week", *KEY]).reset_index(drop=True)
    served = predictor.predict(sales, stock, required_pairs=pairs)
    pd.testing.assert_frame_equal(evaluated, served)
    assert served.loc[served.warehouse_name == "x", "quantity"].eq(123).all()
    assert served.loc[served.warehouse_name == "new_warehouse", "quantity"].eq(7).all()


def test_money_is_exponentiated_before_weekly_sum_with_null_discount():
    sales, _ = make_inputs(n=3, start="2026-07-27")
    sales["sales_sum"] = np.log([2.0, 3.0, 5.0])
    sales["discount_sum"] = [np.nan, np.log(4.0), 0.0]
    result = monetary_weekly_features(sales)
    assert result.sales_sum.iloc[0] == pytest.approx(10.0)
    assert result.discount_sum.iloc[0] == pytest.approx(5.0)  # null=0, logged numeric zero=1
    assert sales.discount_sum.isna().iloc[0]


def test_research_targets_do_not_cross_selection_or_test():
    train, valid = research_split([pd.Timestamp("2026-05-24")], horizon=9, step=28, n_train=4)
    assert max(train) + pd.Timedelta(weeks=9) <= valid
    assert valid + pd.Timedelta(weeks=9) <= pd.Timestamp("2026-05-24")
    assert all(origin.weekday() == 6 for origin in [*train, valid])


def test_inference_is_deterministic_offline_and_cache_is_defensive(predictor, inputs, monkeypatch):
    import socket

    def denied(*args, **kwargs):
        raise AssertionError("Network access during inference")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    first = predictor.predict(*inputs)
    second = predictor.predict(*inputs)
    pd.testing.assert_frame_equal(first, second)
    cache = predictor.get_cached_forecast()
    cache.loc[0, "quantity"] = -1
    assert predictor.get_cached_forecast().quantity.ge(0).all()


def test_api_weekly_post_get_json_errors_and_cache(predictor, inputs, monkeypatch):
    import app.main as main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "predictor", predictor)
    client = TestClient(main.app)
    assert client.get("/health").json()["model_loaded"]
    assert client.get("/predict/all").status_code == 400
    assert client.get("/predict").status_code == 400
    assert client.post("/predict").status_code == 422
    sales, stock = inputs

    def post(a, b):
        return client.post("/predict", files={"sales": ("sales.csv", a.to_csv(index=False).encode()), "stock": ("stock.csv", b.to_csv(index=False).encode())})

    empty = client.post("/predict", files={"sales": ("empty.csv", b""), "stock": ("stock.csv", b"")})
    assert empty.status_code == 400
    assert empty.json()["code"] == "VALIDATION_ERROR"
    assert post(sales.iloc[1:], stock).status_code == 422
    bad = sales.copy()
    bad["quantity"] = "broken"
    assert post(bad, stock).status_code == 422
    stock["stock"] = -2
    response = post(sales, stock)
    assert response.status_code == 200
    forecast = pd.read_csv(io.BytesIO(response.content))
    assert len(forecast) == 9
    assert forecast.columns.tolist() == ["week", *KEY, "quantity"]
    for url in ["/predict?article=x", "/predict?warehouse=x", "/predict?article=x&warehouse=x", "/predict/all"]:
        result = client.get(url)
        assert result.status_code == 200
        pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(result.content)), forecast)
    pd.testing.assert_frame_equal(pd.DataFrame(client.get("/predict/all?format=json").json()), forecast)
    assert pd.read_csv(io.BytesIO(client.get("/predict?article=unknown").content)).empty
    assert client.get("/predict/all?format=invalid").status_code == 400
    # Failed validation must not destroy the last successful forecast.
    post(bad, stock)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(client.get("/predict/all").content)), forecast)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    from train import save_artifact

    path = tmp_path_factory.mktemp("fresh_weekly_model")
    sales, stock = make_inputs(n=300, start="2024-01-01")
    sales["quantity"] = np.arange(len(sales)) % 4
    panel = build_panel(sales, stock)
    frame = make_training_frame(panel, pd.Timestamp("2024-07-07"))
    model = get_model_class("lgbm")({"n_estimators": 8, "min_child_samples": 2, "num_leaves": 7, "threads": 1, "seed": 42, "log_period": 0})
    model.fit(frame, transform_target(frame.target.to_numpy(float), model.target_transform))
    config = save_artifact(model, path, "2024-09-08", "2024-09-08", len(frame), 1)
    return path, model, frame, config


def test_fresh_model_serialization_preserves_predictions(artifact):
    path, model, frame, _ = artifact
    loaded = Predictor(path).load()
    np.testing.assert_allclose(loaded.model.predict(frame), model.predict(frame), rtol=0, atol=0)


def test_old_daily_artifact_is_rejected_before_loading_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"horizon": 64, "model_type": "blend"}), encoding="utf-8")
    predictor = Predictor(tmp_path)
    with pytest.raises(ModelNotReadyError, match="granularity"):
        predictor.load()
    assert not predictor.is_loaded and predictor.config == {}


def test_semantically_incompatible_preprocessor_rejected_even_with_valid_hash(artifact, tmp_path):
    path, _, _, _ = artifact
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    spec_path = tmp_path / "preprocessor.pkl"
    spec = pickle.loads(spec_path.read_bytes())
    spec["stock_aggregation"] = "max"
    payload = pickle.dumps(spec)
    spec_path.write_bytes(payload)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["preprocessor_sha256"] = hashlib.sha256(payload).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelNotReadyError, match="препроцессора"):
        Predictor(tmp_path).load()


def test_failed_reload_clears_previous_model_and_cache(artifact, tmp_path, inputs):
    path, _, _, config = artifact
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    predictor = Predictor(tmp_path).load()
    predictor.predict(*inputs)
    (tmp_path / config["model_file"]).write_bytes(b"corrupt")
    with pytest.raises(ModelNotReadyError, match="SHA-256"):
        predictor.load()
    assert not predictor.is_loaded and predictor.config == {}
    with pytest.raises(ForecastNotReadyError):
        predictor.get_cached_forecast()


def test_sample_generator_supports_char_article_and_preserves_negative_stock(tmp_path):
    import subprocess

    sales, stock = make_inputs(n=184)
    stock = stock.rename(columns={"article": "char_article"})
    stock.loc[len(stock) - 10, "stock"] = -3
    future = stock.iloc[-1:].copy()
    future["date"] = "2026-12-31"
    stock = pd.concat([stock, future], ignore_index=True)
    sales_path, stock_path = tmp_path / "sales.csv", tmp_path / "stock.csv"
    sales.to_csv(sales_path, index=False)
    stock.to_csv(stock_path, index=False)
    out = tmp_path / "samples"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_samples.py"), "--sales", str(sales_path),
         "--stock", str(stock_path), "--out-dir", str(out), "--days", "180", "--articles", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    sample = pd.read_csv(out / "sample_stock_6mo.csv")
    assert "char_article" in sample
    assert sample.stock.min() == -3
    assert sample.date.max() == sales.date.max()
    assert len(sample) == 180
