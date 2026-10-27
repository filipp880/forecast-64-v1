"""Weekly output and scoring checks independent of any trained artifact."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.metrics import METRIC_WEIGHTS, compute_all_metrics, smape_series, wape
from app.submission import COLUMNS, validate_forecast
from score import main as score_main, make_leaderboard, normalize_minmax, normalize_rank


@pytest.fixture
def forecast():
    weeks = pd.date_range("2026-07-27", periods=9, freq="7D").strftime("%Y-%m-%d")
    return pd.DataFrame(
        [(week, "w", article, 0.5) for week in weeks for article in ("a", "b")],
        columns=COLUMNS,
    )


def test_complete_weekly_forecast_keeps_fractional_values(forecast):
    before = forecast.copy(deep=True)
    summary = validate_forecast(forecast.sample(frac=1, random_state=1), forecast, "2026-07-27")
    assert summary["rows"] == 18
    assert summary["weeks"] == 9
    assert summary["last_day"] == "2026-09-27"
    assert summary["total"] == 9.0
    pd.testing.assert_frame_equal(forecast, before)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.1, "bad"])
def test_submission_bad_quantity_rejected(forecast, value):
    forecast["quantity"] = forecast.quantity.astype(object)
    forecast.loc[0, "quantity"] = value
    with pytest.raises(ValueError):
        validate_forecast(forecast, forecast, "2026-07-27")


@pytest.mark.parametrize("column", ["week", "warehouse_name", "article"])
def test_submission_empty_key_rejected(forecast, column):
    forecast.loc[0, column] = None
    with pytest.raises(ValueError):
        validate_forecast(forecast, forecast, "2026-07-27")


@pytest.mark.parametrize("week", ["2026-07-28", "2026-07-20", "2026-09-28", "2026-7-27", "2026-07-27T00:00:00"])
def test_wrong_week_rejected(forecast, week):
    forecast.loc[0, "week"] = week
    with pytest.raises(ValueError):
        validate_forecast(forecast, forecast, "2026-07-27")


def test_pair_week_holes_are_not_hidden_by_global_week_coverage(forecast):
    with pytest.raises(ValueError):
        validate_forecast(forecast.iloc[1:], forecast, "2026-07-27")
    duplicate = pd.concat([forecast, forecast.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_forecast(duplicate, forecast, "2026-07-27")


def test_expected_catalogue_controls_pairs_without_cartesian_expansion(forecast):
    pairs = pd.DataFrame({"warehouse_name": ["w", "other"], "article": ["a", "b"]})
    with pytest.raises(ValueError, match="catalogue mismatch"):
        validate_forecast(forecast, pairs, "2026-07-27")
    pairs = forecast[["warehouse_name", "article"]].drop_duplicates()
    pairs.loc[len(pairs)] = ["w", "unseen"]
    with pytest.raises(ValueError, match="missing=1"):
        validate_forecast(forecast, pairs, "2026-07-27")


def test_contract_rejects_old_columns_and_horizon(forecast):
    with pytest.raises(ValueError):
        validate_forecast(forecast.rename(columns={"week": "date"}), forecast, "2026-07-27")
    with pytest.raises(ValueError):
        validate_forecast(forecast, forecast, "2026-07-27", weeks=8)
    with pytest.raises(ValueError):
        validate_forecast(forecast, forecast, "2026-07-28")


def test_wape_all_zero_truth_edge_cases():
    assert wape(np.zeros(2), np.zeros(2)) == 0
    assert wape(np.zeros(2), np.array([0.0, 2.0])) == 100
    assert smape_series(np.array([0.0, 1.0]), np.zeros(2)) == 100


def test_metrics_match_weekly_keys_independent_of_order(forecast):
    prediction = forecast.sample(frac=1, random_state=2)
    prediction["week"] = pd.to_datetime(prediction.week)
    assert compute_all_metrics(forecast, prediction) == {key: 0.0 for key in METRIC_WEIGHTS}


@pytest.mark.parametrize("fault", ["missing", "extra", "negative", "nan", "infinite", "duplicate", "blank"])
def test_metrics_never_repair_invalid_predictions(forecast, fault):
    prediction = forecast.copy()
    if fault == "missing":
        prediction = prediction.iloc[1:]
    elif fault == "extra":
        row = prediction.iloc[:1].copy()
        row["article"] = "extra"
        prediction = pd.concat([prediction, row], ignore_index=True)
    elif fault == "duplicate":
        prediction = pd.concat([prediction, prediction.iloc[:1]], ignore_index=True)
    elif fault == "blank":
        prediction.loc[0, "article"] = " "
    else:
        prediction.loc[0, "quantity"] = {"negative": -1, "nan": np.nan, "infinite": np.inf}[fault]
    with pytest.raises(ValueError):
        compute_all_metrics(forecast, prediction)


def test_metrics_origin_must_align_and_does_not_collapse_repeated_weeks(forecast):
    first = forecast.assign(origin="2026-07-26")
    second = forecast.assign(origin="2026-07-19")
    truth = pd.concat([first, second], ignore_index=True)
    prediction = truth.sample(frac=1, random_state=3)
    assert compute_all_metrics(truth, prediction)["rmse"] == 0
    with pytest.raises(ValueError):
        compute_all_metrics(truth, prediction.drop(columns="origin"))
    prediction.loc[prediction.index[0], "origin"] = "2026-07-12"
    with pytest.raises(ValueError):
        compute_all_metrics(truth, prediction)


def test_series_have_equal_smape_weight_and_zero_bias_series_are_excluded():
    truth = pd.DataFrame({
        "week": ["2026-07-27", "2026-08-03", "2026-07-27"],
        "warehouse_name": "w", "article": ["a", "a", "b"], "quantity": [0.0, 0.0, 2.0],
    })
    prediction = truth.copy()
    prediction["quantity"] = [0.0, 0.0, 0.0]
    metrics = compute_all_metrics(truth, prediction)
    assert metrics["smape"] == 100  # each series gets half of total weight
    assert metrics["bias"] == 100   # all-zero series a is excluded


def test_rank_ties_and_single_candidate():
    values = {name: {key: value for key in METRIC_WEIGHTS} for name, value in [("a", 1), ("b", 1), ("c", 3)]}
    normalized = normalize_rank(values)
    assert normalized["a"]["smape"] == normalized["b"]["smape"] == 0.25
    assert normalized["c"]["smape"] == 1
    assert normalize_rank({"a": values["a"]})["a"]["smape"] == 0
    assert make_leaderboard(values).score.tolist() == [0.25, 0.25, 1.0]


def test_minmax_zero_baseline_and_clipping():
    baseline = {key: 0.0 for key in METRIC_WEIGHTS}
    values = {"perfect": baseline.copy(), "bad": {key: 2.0 for key in METRIC_WEIGHTS}}
    result = normalize_minmax(values, baseline)
    assert result["perfect"]["smape"] == 0
    assert result["bad"]["smape"] == 1
    assert make_leaderboard(values, "minmax", baseline).score.tolist() == [0.0, 1.0]


def test_score_cli_skips_invalid_submission_without_repair(tmp_path, forecast, capsys):
    truth_path, good_path, bad_path = [tmp_path / f"{name}.csv" for name in ("truth", "good", "bad")]
    forecast.to_csv(truth_path, index=False)
    forecast.to_csv(good_path, index=False)
    forecast.iloc[1:].to_csv(bad_path, index=False)
    output = tmp_path / "leaderboard.csv"
    assert score_main(["--truth", str(truth_path), "--submissions", str(good_path), str(bad_path), "--output", str(output)]) == 0
    leaderboard = pd.read_csv(output)
    assert leaderboard.submission.tolist() == ["good"]
    assert "SKIPPED bad" in capsys.readouterr().err
    assert score_main(["--truth", str(truth_path), "--submissions", str(bad_path)]) == 1


def test_score_cli_rejects_invalid_baseline(tmp_path, forecast):
    truth = tmp_path / "truth.csv"
    baseline = tmp_path / "baseline.csv"
    forecast.to_csv(truth, index=False)
    forecast.iloc[:-1].to_csv(baseline, index=False)
    assert score_main(["--truth", str(truth), "--submissions", str(truth), "--baseline", str(baseline), "--method", "minmax"]) == 1


def test_score_script_executes_from_outside_project(tmp_path, forecast):
    path = tmp_path / "truth.csv"
    forecast.to_csv(path, index=False)
    process = subprocess.run(
        [sys.executable, str(ROOT / "score.py"), "--truth", str(path), "--submissions", str(path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "Normalization: rank" in process.stdout
