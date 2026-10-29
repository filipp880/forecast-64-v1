"""Shared model-aware feature preparation, without a dependency on serving."""
from __future__ import annotations

import pandas as pd

from .features import make_features, make_training_frame


def needs_money(model_type, params=None):
    params = params or {}
    if model_type != "research":
        return False
    recipe = params.get("recipe", {"source": "all"})
    from .models.research_model import recipe_member_names
    # These standalone members use the complete enhancement profile.
    return bool(recipe_member_names(recipe) &
                {"monetary", "all", "poisson", "tweedie", "mae", "quantile", "catboost", "grouped"})


def prepare_panel(sales, stock, model_type, params=None, history_end=None):
    from .features import build_panel
    panel = build_panel(sales, stock, history_end=history_end)
    if needs_money(model_type, params):
        from .enhancements import add_money
        panel = add_money(panel, sales)
    return panel


def feature_frame(panel, cutoff, model_type, *, training=False, horizons=range(1, 10),
                  lookback_days=180, stock_policy="window", first_week=None, model_params=None):
    """Every row uses its own origin; only observed pairs enter group aggregates."""
    builder = make_training_frame if training else make_features
    frame = builder(panel, cutoff, horizons, lookback_days, stock_policy, first_week)
    if model_type in ("research", "ideas"):
        from .enhancements import augment_features
        frame = augment_features(panel, frame, cutoff)
    if model_type == "ideas":
        from .ideas_features import augment_ideas
        frame = augment_ideas(panel, frame, cutoff).copy()
        frame["feature_origin"] = pd.Timestamp(cutoff)
    if model_type == 'research' and (model_params or {}).get('expert_specs'):
        from .sequence_features import add_sequence_features
        frame = add_sequence_features(panel, frame, cutoff)
        frame['feature_origin'] = pd.Timestamp(cutoff)
    return frame


def fallback_rounding(model, config):
    if getattr(model, "name", None) == "research":
        return bool(model.recipe.get("post", {}).get("round_int", True))
    if getattr(model, "name", None) == "ideas":
        return True
    calibration = config.get("calibration")
    if calibration:
        return bool(calibration.get("round_int", False))
    return bool(config.get("postprocess", {}).get("round_int", False))


def postprocess_fallback(model, config, frame):
    """Apply the same zero/quantization policy to external cold-pair values."""
    import numpy as np
    from .models import postprocess_forecast
    post = config.get("postprocess", {})
    raw = frame.quantity.to_numpy(float)
    model_type = getattr(model, "name", None)
    if model_type == "research":
        from .models.research_model import quantize
        raw = quantize(raw, frame, model.recipe.get("post", {}))
    elif model_type == "ideas":
        raw = np.rint(np.maximum(raw, 0.))
    elif config.get("calibration"):
        # forecast_quantities' calibrated branch delegates the complete
        # decision policy to combine(), ignoring the outer postprocess flags.
        calibration = config["calibration"]
        raw = np.maximum(raw, 0.)
        if calibration.get("round_int", False):
            threshold = float(calibration.get("round_threshold", .5))
            raw = np.rint(raw) if threshold == .5 else np.floor(raw + 1 - threshold)
        return raw
    return postprocess_forecast(raw, float(post.get("zero_threshold", 0)),
                                bool(post.get("round_int", False)))
