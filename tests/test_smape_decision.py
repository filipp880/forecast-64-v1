"""Expected-loss math, serialization and temporal OOF boundaries."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from app.smape_decision import (SmapeDecision, admissible_oof_origins,
                                development_gate, expected_smape_risks)
from scripts import research_decision as runner


def test_expected_smape_threshold_accounts_for_quantity_error():
    losses = np.full((2, 1), 44.)
    probabilities = np.array([.55, .57])
    risks = expected_smape_risks(probabilities, losses)
    assert risks.argmin(axis=1).tolist() == [0, 1]
    assert np.allclose(risks[:, 0], 200 * probabilities)
    assert np.allclose(risks[:, 1], 200 * (1 - probabilities) + 44 * probabilities)


def test_zero_zero_is_zero_and_uncertain_positive_can_choose_zero():
    decoder = SmapeDecision({"calibration": "identity", "action_mode": "magnitude", "magnitude_scales": [1.]})
    decoder.fit([.1, .9], [1., 1.], [1., 1.], [0., 1.])
    q = decoder.predict([0., .1, 1.], [1., 1., 1.], [1., 1., 1.])
    assert q.tolist() == [0., 0., 1.]


@pytest.mark.parametrize("calibration", ["identity", "platt"])
def test_json_roundtrip_preserves_deterministic_fractional_actions(calibration):
    probability = np.tile([.1, .3, .7, .9], 20)
    magnitude = np.tile([.25, .5, 1., 2.], 20)
    actual = np.tile([0., 0., .5, 2.], 20)
    decoder = SmapeDecision({"calibration": calibration}).fit(probability, magnitude, magnitude, actual)
    expected = decoder.predict(probability, magnitude, magnitude)
    restored = SmapeDecision.from_dict(json.loads(json.dumps(decoder.to_dict())))
    assert np.array_equal(expected, restored.predict(probability, magnitude, magnitude))
    assert np.isfinite(expected).all() and (expected >= 0).all()


def test_admissibility_requires_the_last_target_sunday_not_just_an_earlier_origin():
    assert admissible_oof_origins(runner.ORIGINS, "2026-01-18") == ["2025-10-19"]
    assert admissible_oof_origins(runner.ORIGINS, "2026-03-22") == ["2025-10-19", "2026-01-18"]
    assert "2026-01-18" not in admissible_oof_origins(runner.ORIGINS, "2026-03-21")
    # Wednesday origin starts next Monday; nine weeks end later than origin+63d.
    assert not admissible_oof_origins(["2026-01-21"], "2026-03-25")
    with pytest.raises(ValueError):
        admissible_oof_origins(runner.ORIGINS, "2026-03-22", horizon=-1)


def test_gate_rejects_hidden_metric_tradeoff_even_when_weighted_ratio_improves():
    champion = {origin: dict.fromkeys(["smape", "wape", "rmse", "bias", "rmsle"], 1.) for origin in ("jan", "mar")}
    candidate = {origin: {"smape": .95, "wape": .95, "rmse": 1.10, "bias": .95, "rmsle": .8} for origin in champion}
    gate = development_gate(candidate, champion)
    assert gate["mean_weighted_ratio"] < .995
    assert not gate["eligible"]
    assert "mean_other_metric_worse_than_1pct" in gate["reasons"]
    assert "per_origin_guard_failed" in gate["reasons"]


def test_gate_accepts_uniform_small_improvement():
    champion = {"jan": dict.fromkeys(["smape", "wape", "rmse", "bias", "rmsle"], 1.)}
    candidate = {"jan": dict.fromkeys(champion["jan"], .99)}
    assert development_gate(candidate, champion)["eligible"]


def test_development_never_loads_confirmation_labels(tmp_path, monkeypatch):
    loaded = []
    specs = {"identity_magnitude": {"calibration": "identity", "action_mode": "magnitude"}}
    protocol = {"candidates": specs, "development_origins": runner.ORIGINS[1:3], "limitations": [],
                "bootstrap": {"seed": 42, "replicates": 10}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")

    def load(origin, frozen):
        assert origin != "2026-05-24", "Development attempted to inspect confirmation"
        loaded.append(origin)
        h = np.tile(np.arange(1, 10), 2)
        actual = np.tile([0., 1., 0., 2., 0., 1., 0., 2., 1.], 2)
        frame = pd.DataFrame({"h": h, "week": pd.Timestamp(origin) + pd.to_timedelta(h * 7 - 6, unit="D"),
                              "warehouse_name": "w", "article": np.repeat(["a", "b"], 9),
                              "actual": actual, "mean_28": 1 / 7})
        raw = {"probability": np.full(len(h), .6), "magnitude": np.ones(len(h))}
        return frame, raw, np.ones(len(h)), np.ones(len(h))

    monkeypatch.setattr(runner, "load_origin", load)
    report = runner.development(tmp_path, protocol)
    assert report["confirmation_labels_loaded"] is False
    assert set(loaded) == {"2025-10-19", "2026-01-18", "2026-03-22"}
    assert not (tmp_path / "confirmation.json").exists()
    assert report["results"]["identity_magnitude"]["origins"]["2026-01-18"]["prior_oof_origins"] == ["2025-10-19"]


def test_confirmation_refuses_candidate_that_failed_development(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "development", lambda *_: {"results": {"x": {"gate": {"eligible": False}}}})
    monkeypatch.setattr(runner, "load_origin", lambda *_: pytest.fail("Should not load truth"))
    with pytest.raises(ValueError, match="passed development"):
        runner.confirmation(tmp_path, {}, "x")
