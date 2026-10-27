"""Прозрачная постобработка ансамбля; параметры выбираются только на OOF."""
from __future__ import annotations
import numpy as np

def combine(probability, magnitude, lgbm_raw, mean28, n_sale_days, spec=None):
    spec = spec or {}
    allowed = {"weight_hurdle", "gate", "medium_gate", "expected", "mean_weight", "scale", "regular_scale", "round_int", "round_threshold"}
    if set(spec) - allowed:
        raise ValueError("Неизвестные параметры калибровки")
    for key in ("weight_hurdle", "gate", "medium_gate", "mean_weight", "round_threshold"):
        if key in spec and not 0 <= float(spec[key]) <= 1:
            raise ValueError(f"Параметр {key} должен быть в [0,1]")
    for key in ("scale", "regular_scale"):
        if key in spec and (not np.isfinite(float(spec[key])) or float(spec[key]) <= 0):
            raise ValueError(f"Параметр {key} должен быть положительным конечным")
    w = float(spec.get('weight_hurdle', 0.3))
    gate = float(spec.get('gate', 0.5))
    if not (0 <= w <= 1 and 0 <= gate <= 1):
        raise ValueError('Некорректные параметры ансамбля')
    effective_gate = np.where((n_sale_days >= 10) & (n_sale_days < 45), float(spec.get('medium_gate', gate)), gate)
    hurdle = probability * magnitude if spec.get('expected', False) else np.where(probability >= effective_gate, magnitude, 0.0)
    raw = w * hurdle + (1 - w) * np.maximum(lgbm_raw, 0.0)
    b = float(spec.get('mean_weight', 0.0))
    raw = (1 - b) * raw + b * np.maximum(mean28, 0.0)
    raw = raw * float(spec.get('scale', 1.0)) * np.where(n_sale_days >= 45, float(spec.get('regular_scale', 1.0)), 1.0)
    if not np.isfinite(raw).all():
        raise ValueError('Некорректный результат ансамбля')
    if spec.get('round_int', False):
        threshold = float(spec.get('round_threshold', 0.5))
        return np.rint(np.maximum(raw, 0.0)) if threshold == 0.5 else np.floor(np.maximum(raw, 0.0) + 1 - threshold)
    return np.maximum(raw, 0.0)

def predict_calibrated(model, features, spec):
    if model.name != 'blend':
        raise ValueError('Калибровка поддерживает только blend')
    x = model.hurdle._encode(features, False)
    p = model.hurdle.classifier.predict(x, num_iteration=model.hurdle._best_iterations['classifier'], num_threads=int(model.params.get('threads', 8)))
    m = model.hurdle.regressor.predict(x, num_iteration=model.hurdle._best_iterations['regressor'], num_threads=int(model.params.get('threads', 8)))
    from .models import inverse_target
    lg = inverse_target(model.lgbm.predict(features), model.lgbm.target_transform)
    return combine(p, m, lg, 7 * features.mean_28.to_numpy(float), features.n_sale_days_90.to_numpy(float), spec)


def forecast_quantities(model, features, zero_threshold=0.0, round_int=False, calibration=None):
    """One serving/evaluation path, including the explicit no-history fallback."""
    from .models import inverse_target, postprocess_forecast
    if calibration:
        predictions = predict_calibrated(model, features, calibration)
    else:
        predictions = postprocess_forecast(inverse_target(model.predict(features), model.target_transform),
                                           zero_threshold, round_int)
    if "no_history" in features:
        fallback = postprocess_forecast(features.mean_week_4.to_numpy(float), zero_threshold,
                                       bool(calibration.get("round_int", False)) if calibration else round_int)
        predictions = np.where(features.no_history.to_numpy() > 0, fallback, predictions)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Model returned nonfinite or negative quantity")
    return predictions
