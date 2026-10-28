"""Five weekly metrics with exact key alignment and no submission repair.

SMAPE and absolute Bias average equally over warehouse/article series.
Zero/zero points remain in SMAPE. WAPE, RMSE and RMSLE are global.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

METRIC_WEIGHTS: Dict[str, float] = {
    "smape": 0.50,
    "wape": 0.20,
    "rmse": 0.15,
    "bias": 0.10,
    "rmsle": 0.05,
}

KEY_COLUMNS = ["week", "warehouse_name", "article"]
METRIC_NAMES = ("smape", "wape", "rmse", "bias", "rmsle")


def smape_series(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE одной серии, в процентах (0..200); недели «0 против 0» дают 0%."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.zeros_like(numerator, dtype="float64")
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    return float(np.mean(ratio) * 100.0)


def _series_codes(series_ids: np.ndarray) -> np.ndarray:
    """Целочисленные коды серий (hash-таблица) — для векторного groupby."""
    codes, _ = pd.factorize(series_ids, sort=False)
    return codes


def smape(y_true: np.ndarray, y_pred: np.ndarray, series_ids: np.ndarray) -> float:
    """SMAPE по сериям (склад × товар), усреднённый.

    Недели, где факт=0 и прогноз=0, входят в среднее серии как 0%.
    """
    if len(y_true) == 0:
        return 0.0
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.zeros(len(y_true), dtype="float64")
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    means = pd.Series(ratio).groupby(_series_codes(series_ids)).mean()
    return float(means.mean() * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE — глобально, в процентах."""
    denominator = np.sum(np.abs(y_true))
    numerator = np.sum(np.abs(y_true - y_pred))
    if denominator == 0:
        return 0.0 if numerator == 0 else 100.0
    return float(numerator / denominator * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE — глобально."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias_abs_series(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """|Bias| одной серии, в процентах; None если ``mean(y_true) == 0``."""
    mean_y = np.mean(y_true)
    if mean_y == 0:
        return None
    return float(abs(np.mean(y_pred) - mean_y) / mean_y * 100.0)


def bias_abs(y_true: np.ndarray, y_pred: np.ndarray, series_ids: np.ndarray) -> float:
    """|Bias| по сериям, усреднённый; серии с ``mean(y)=0`` исключаются.

    Векторный эквивалент цикла из ``score.py``: средние ``y``/``ŷ`` по серии
    считаются ``groupby``, затем усредняется ``|mean(ŷ)-mean(y)|/mean(y)·100``.
    """
    if len(y_true) == 0:
        return 0.0
    frame = pd.DataFrame({"series": _series_codes(series_ids), "y": y_true, "p": y_pred})
    means = frame.groupby("series", sort=False)[["y", "p"]].mean()
    means = means[means["y"] != 0]
    if means.empty:
        return 0.0
    values = np.abs(means["p"] - means["y"]) / means["y"] * 100.0
    return float(values.mean())


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE — глобально, через ``log1p``."""
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def compute_all_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> Dict[str, float]:
    """Compute metrics only after strict weekly key and value validation.

    ``origin`` is an optional extra key and must occur in both frames.
    Partial horizons are accepted for diagnostics; the submission validator
    separately enforces the complete nine-week calendar.
    """
    from .submission import _dates

    df_true, df_pred = df_true.copy(), df_pred.copy()
    if ("origin" in df_true) != ("origin" in df_pred):
        raise ValueError("origin должен присутствовать и в факте, и в прогнозе")
    keys = KEY_COLUMNS + (["origin"] if "origin" in df_true else [])
    for name, frame in (("truth", df_true), ("prediction", df_pred)):
        missing = set([*keys, "quantity"]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Empty {name}")
        frame["week"] = _dates(frame["week"])
        if not frame.week.dt.weekday.eq(0).all():
            raise ValueError(f"Weeks must start on Monday: {name}")
        frame["quantity"] = pd.to_numeric(frame["quantity"], errors="raise").astype(float)
        if "origin" in frame:
            frame["origin"] = _dates(frame["origin"], "origin")
        for column in ("warehouse_name", "article"):
            frame[column] = frame[column].astype("string")
            if frame[column].isna().any() or frame[column].str.strip().eq("").any():
                raise ValueError(f"Empty {column}: {name}")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"Пустые или повторяющиеся ключи {name}: {keys}")
        quantity = frame["quantity"].to_numpy(float)
        if not np.isfinite(quantity).all() or (quantity < 0).any():
            raise ValueError(f"Некорректные quantity: {name}")
    truth_keys = pd.MultiIndex.from_frame(df_true[keys])
    pred_keys = pd.MultiIndex.from_frame(df_pred[keys])
    if len(truth_keys.difference(pred_keys)) or len(pred_keys.difference(truth_keys)):
        raise ValueError("Truth and prediction keys must match exactly")
    df = df_true.merge(
        df_pred[[*keys, "quantity"]],
        on=keys,
        how="left",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    y_true = df["quantity_true"].to_numpy(dtype="float64")
    y_pred = df["quantity_pred"].to_numpy(dtype="float64")
    series_ids = pd.MultiIndex.from_frame(df[["warehouse_name", "article"]]).to_numpy()

    result = {
        "smape": smape(y_true, y_pred, series_ids),
        "wape": wape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias_abs(y_true, y_pred, series_ids),
        "rmsle": rmsle(y_true, y_pred),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite metric: quantity values overflow the metric calculation")
    return result


def format_metrics(metrics: Dict[str, float]) -> str:
    """Строка для консоли: ``SMAPE=…% WAPE=…% RMSE=… |Bias|=…% RMSLE=…``."""
    return (
        f"SMAPE={metrics['smape']:.2f}%  WAPE={metrics['wape']:.2f}%  RMSE={metrics['rmse']:.4f}  "
        f"|Bias|={metrics['bias']:.2f}%  RMSLE={metrics['rmsle']:.4f}"
    )
