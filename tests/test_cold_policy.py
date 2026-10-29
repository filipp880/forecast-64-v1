"""Cold-pair decisions must follow the effective model postprocessing contract."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.features import CAT_FEATURES, KEY
from app.predictor import Predictor
from app.models.research_model import quantize


def history(weekly_quantity):
    dates = pd.date_range("2026-01-28", "2026-07-26")
    sources = []
    for i in range(2):
        frame = pd.DataFrame({"date": dates, "quantity": weekly_quantity / 7})
        for column in CAT_FEATURES:
            frame[column] = "x"
        frame["warehouse_name"], frame["article"] = f"w{i}", f"a{i}"
        sources.append(frame)
    sales = pd.concat(sources, ignore_index=True)
    return sales, sales[["date", *KEY]].assign(stock=1.)


class TransparentModel:
    target_transform = "none"
    name = "test"
    params = {}

    def predict(self, frame):
        return frame.mean_week_4.to_numpy(float)


class TransparentResearch(TransparentModel):
    name = "research"

    def __init__(self, post):
        self.recipe = {"source": "dynamics", "post": post}
        self.params = {"recipe": self.recipe}

    def predict(self, frame):
        return quantize(super().predict(frame), frame, self.recipe["post"])


class TransparentIdeas(TransparentModel):
    name = "ideas"

    def predict(self, frame):
        return np.rint(np.maximum(super().predict(frame), 0.))


class CalibratedBlend(TransparentModel):
    """Keep real calibration/combine code, with transparent positive heads."""
    name = "blend"

    def __init__(self):
        self.hurdle = SimpleNamespace(
            _encode=lambda frame, fit: frame,
            classifier=SimpleNamespace(predict=lambda frame, **kwargs: np.ones(len(frame))),
            regressor=SimpleNamespace(predict=lambda frame, **kwargs: frame.mean_week_4.to_numpy(float)),
            _best_iterations={"classifier": 1, "regressor": 1},
        )
        self.lgbm = SimpleNamespace(
            predict=lambda frame: np.log1p(frame.mean_week_4.to_numpy(float)),
            target_transform="log1p",
        )


def predict(model, weekly_quantity, post=None, calibration=None):
    predictor = Predictor("unused-test-artifact")
    predictor.model = model
    predictor.config = {"postprocess": post or {"round_int": False}}
    if calibration is not None:
        predictor.config["calibration"] = calibration
    result = predictor.predict(*history(weekly_quantity))
    warm = result.loc[((result.warehouse_name == "w0") & (result.article == "a0"))]
    cold = result.loc[((result.warehouse_name == "w1") & (result.article == "a0"))]
    assert len(warm) == len(cold) == 9
    return warm.quantity.to_numpy(), cold.quantity.to_numpy()


@pytest.mark.parametrize("calibrated_round,outer_round", [(True, False), (False, True)])
def test_calibrated_rounding_overrides_outer_rounding_for_cold_pairs(calibrated_round, outer_round):
    warm, cold = predict(CalibratedBlend(), .25, {"round_int": outer_round}, {"round_int": calibrated_round})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 0. if calibrated_round else .25)


def test_zero_threshold_applies_to_unrounded_cold_fallback():
    warm, cold = predict(TransparentModel(), .25, {"zero_threshold": .5, "round_int": False})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 0.)


def test_calibration_owns_postprocessing_including_nonstandard_round_threshold():
    warm, cold = predict(CalibratedBlend(), .4, {"round_int": False},
                         {"round_int": True, "round_threshold": .35})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 1.)


def test_calibrated_branch_preserves_established_outer_zero_threshold_semantics():
    warm, cold = predict(CalibratedBlend(), .6, {"zero_threshold": .8, "round_int": True},
                         {"round_int": False})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, .6)


def test_research_threshold_is_applied_after_recipe_quantization():
    warm, cold = predict(TransparentResearch({"round_int": True}), .6,
                         {"zero_threshold": .8, "round_int": False})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 1.)


def test_research_nonstandard_quantization_threshold_reaches_cold_pairs():
    warm, cold = predict(TransparentResearch({"round_int": True, "round_threshold": .35}), .4)
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 1.)


def test_research_nine_week_apportionment_reaches_cold_pairs():
    warm, cold = predict(TransparentResearch({"apportion": True}), .4)
    np.testing.assert_allclose(warm, cold)
    assert warm.sum() == cold.sum() == 4.


def test_outer_rounding_also_applies_after_fractional_research_recipe():
    warm, cold = predict(TransparentResearch({"round_int": False}), .6, {"round_int": True})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 1.)


def test_ideas_quantizes_before_outer_zero_threshold():
    warm, cold = predict(TransparentIdeas(), .6, {"zero_threshold": .8, "round_int": False})
    np.testing.assert_allclose(warm, cold)
    np.testing.assert_allclose(cold, 1.)
