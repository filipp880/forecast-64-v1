"""Expected-SMAPE actions learned from strictly earlier out-of-fold labels.

This experimental decoder is deliberately separate from the serving model until
its frozen temporal comparison has passed the release gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


METRICS = ("smape", "wape", "rmse", "bias", "rmsle")
WEIGHTS = np.array([.50, .20, .15, .10, .05])


def admissible_oof_origins(origins, evaluation_origin, horizon=9):
    """An OOF label is usable only after its entire target horizon ended."""
    evaluation_origin = pd.Timestamp(evaluation_origin).normalize()
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon <= 0:
        raise ValueError("Positive integer horizon required")
    return [str(pd.Timestamp(origin).date()) for origin in origins
            if pd.Timestamp(origin).normalize() + pd.Timedelta(
                days=7 - pd.Timestamp(origin).weekday() + 7 * horizon - 1) <= evaluation_origin
            and pd.Timestamp(origin).normalize() < evaluation_origin]


def expected_smape_risks(probability, positive_loss):
    p = np.asarray(probability, float)
    losses = np.asarray(positive_loss, float)
    if losses.ndim != 2 or losses.shape[0] != len(p):
        raise ValueError("Expected one conditional positive loss row per probability")
    if not np.isfinite(p).all() or not np.isfinite(losses).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities or conditional losses")
    if ((losses < 0) | (losses > 200)).any():
        raise ValueError("Conditional SMAPE must be in [0, 200]")
    return np.column_stack((200 * p, 200 * (1 - p[:, None]) + p[:, None] * losses))


@dataclass(frozen=True)
class DecisionSpec:
    calibration: str = "platt"
    action_mode: str = "mixed"
    magnitude_scales: tuple = (.85, 1., 1.15)
    probability_bins: tuple = (.2, .4, .6, .8)
    quantity_bins: tuple = (.5, 1., 2., 4., 8., 16., 32.)
    positive_loss_shrinkage: float = 100.


class SmapeDecision:
    """Small Platt calibrator plus shrunk conditional-loss lookup tables."""

    def __init__(self, spec=None):
        self.spec = spec if isinstance(spec, DecisionSpec) else DecisionSpec(**(spec or {}))
        if self.spec.calibration not in ("identity", "platt") or self.spec.action_mode not in ("mixed", "magnitude"):
            raise ValueError("Unknown decision calibration/action mode")
        if self.spec.positive_loss_shrinkage <= 0 or any(scale <= 0 for scale in self.spec.magnitude_scales):
            raise ValueError("Positive shrinkage and action scales required")
        self.slope = 1.
        self.intercept = 0.
        self.loss_tables = None

    def actions(self, magnitude, continuous_champion):
        magnitude, continuous = np.asarray(magnitude, float), np.asarray(continuous_champion, float)
        if magnitude.ndim != 1 or continuous.shape != magnitude.shape:
            raise ValueError("Mismatched action inputs")
        if not np.isfinite(magnitude).all() or not np.isfinite(continuous).all() or (magnitude < 0).any() or (continuous < 0).any():
            raise ValueError("Actions must be finite and nonnegative")
        actions = [magnitude * scale for scale in self.spec.magnitude_scales]
        if self.spec.action_mode == "mixed":
            actions.append(continuous)
        return np.column_stack(actions)

    def probability(self, raw):
        raw = np.asarray(raw, float)
        if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError("Invalid raw probability")
        if self.spec.calibration == "identity":
            return raw
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logits = self.intercept + self.slope * (np.log(clipped) - np.log1p(-clipped))
        return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))

    def _bins(self, probabilities, quantities):
        n_quantity = len(self.spec.quantity_bins) + 1
        p_bin = np.digitize(probabilities, self.spec.probability_bins)
        q_bin = np.digitize(quantities, self.spec.quantity_bins)
        return p_bin * n_quantity + q_bin

    def fit(self, raw_probability, magnitude, continuous_champion, actual, sample_weight=None):
        raw, y = np.asarray(raw_probability, float), np.asarray(actual, float)
        actions = self.actions(magnitude, continuous_champion)
        self.probability(raw)  # validate before fitting
        if y.shape != raw.shape or len(y) != len(actions) or not len(y):
            raise ValueError("Mismatched/empty calibration rows")
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid calibration labels")
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if weight.shape != y.shape or not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("Calibration weights must be positive and finite")
        weight = weight / weight.mean()
        positive = y > 0
        if self.spec.calibration == "platt":
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            logits = (np.log(clipped) - np.log1p(-clipped))[:, None]
            if len(np.unique(positive)) == 2:
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression(C=1., solver="lbfgs", max_iter=200, random_state=42)
                calibrator.fit(logits, positive, sample_weight=weight)
                self.slope, self.intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
            else:
                self.slope = 0.
                self.intercept = float(np.log((weight[positive].sum() + .5) / (weight[~positive].sum() + .5)))
        p = self.probability(raw)
        bins = (len(self.spec.probability_bins) + 1) * (len(self.spec.quantity_bins) + 1)
        self.loss_tables = []
        for quantity in actions.T:
            labels = y[positive]
            q = quantity[positive]
            loss = 200 * np.abs(labels - q) / (labels + q)
            global_loss = float(np.average(loss, weights=weight[positive])) if positive.any() else 200.
            codes = self._bins(p[positive], q)
            counts = np.bincount(codes, weights=weight[positive], minlength=bins)
            totals = np.bincount(codes, weights=weight[positive] * loss, minlength=bins)
            shrink = self.spec.positive_loss_shrinkage
            self.loss_tables.append((totals + shrink * global_loss) / (counts + shrink))
        self.loss_tables = np.asarray(self.loss_tables)
        return self

    def predict(self, raw_probability, magnitude, continuous_champion):
        if self.loss_tables is None:
            raise ValueError("Decision layer is not fitted")
        p = self.probability(raw_probability)
        actions = self.actions(magnitude, continuous_champion)
        losses = np.column_stack([table[self._bins(p, quantity)]
                                  for table, quantity in zip(self.loss_tables, actions.T)])
        selected = expected_smape_risks(p, losses).argmin(axis=1)
        quantities = np.column_stack((np.zeros(len(p)), actions))
        return quantities[np.arange(len(p)), selected]

    def to_dict(self):
        if self.loss_tables is None:
            raise ValueError("Decision layer is not fitted")
        return {"spec": asdict(self.spec), "slope": self.slope, "intercept": self.intercept,
                "loss_tables": self.loss_tables.tolist()}

    @classmethod
    def from_dict(cls, payload):
        result = cls(payload["spec"])
        result.slope, result.intercept = payload["slope"], payload["intercept"]
        result.loss_tables = np.asarray(payload["loss_tables"], float)
        return result


def development_gate(candidate, champion):
    """Frozen project gate, not the competition's official normalized score."""
    if set(candidate) != set(champion) or not candidate:
        raise ValueError("Candidate and champion need identical nonempty origins")
    origins = sorted(candidate)
    ratios = np.asarray([[candidate[origin][metric] / max(champion[origin][metric], 1e-12)
                          for metric in METRICS] for origin in origins])
    if not np.isfinite(ratios).all() or (ratios < 0).any():
        raise ValueError("Invalid metric ratios")
    mean = ratios.mean(axis=0)
    reasons = []
    if float(mean @ WEIGHTS) > .995:
        reasons.append("mean_weighted_ratio_above_0.995")
    if mean[0] > 1:
        reasons.append("mean_smape_worse")
    if (mean < 1).sum() < 3:
        reasons.append("fewer_than_three_metrics_improved")
    if (mean[1:] > 1.01).any():
        reasons.append("mean_other_metric_worse_than_1pct")
    if (ratios[:, 0] > 1.03).any() or (ratios[:, 1:] > 1.05).any():
        reasons.append("per_origin_guard_failed")
    return {"eligible": not reasons, "reasons": reasons,
            "mean_weighted_ratio": float(mean @ WEIGHTS),
            "mean_metric_ratios": dict(zip(METRICS, mean.tolist())),
            "per_origin_ratios": {origin: dict(zip(METRICS, row.tolist())) for origin, row in zip(origins, ratios)}}
