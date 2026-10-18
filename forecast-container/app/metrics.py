"""Пять метрик ТЗ (п. 5) — зеркало официального ``score.py`` из задания.

Используются в ``train.py`` (метрики на отложенном срезе) и ``evaluate.py``
(бэктест по нескольким срезам). Семантика 1-в-1 как в шаблоне организаторов:

- SMAPE — **по сериям** (склад × товар), затем среднее; дни с нулевым
  знаменателем ``(|y|+|ŷ|)/2`` исключаются; если весь знаменатель серии нулевой
  (факт=0 и прогноз=0) — серия даёт 0;
- WAPE / RMSE / RMSLE — **глобально**; RMSLE через ``log1p``;
- |Bias| — **по сериям** (серии с ``mean(y)=0`` исключаются), затем среднее;
- прогноз перед расчётом: пропуски → 0, отрицательные значения → 0.

Числа совпадают с циклом из ``score.py`` до последнего бита (проверено юнит-тестами),
но реализация векторизована: серии кодируются целыми числами (hash) и агрегируются
через ``groupby`` — цикл «по каждой серии маска по всем строкам» на 5 млрд сравнений
строк не используется.

Веса ТЗ: SMAPE 50%, WAPE 20%, RMSE 15%, |Bias| 10%, RMSLE 5% (см.
``METRIC_WEIGHTS``; нормализация и итоговый score — на стороне организаторов).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METRIC_WEIGHTS: Dict[str, float] = {
    "smape": 0.50,
    "wape": 0.20,
    "rmse": 0.15,
    "bias": 0.10,
    "rmsle": 0.05,
}

KEY_COLUMNS = ["date", "warehouse_name", "article"]
METRIC_NAMES = ("smape", "wape", "rmse", "bias", "rmsle")


def smape_series(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE одной серии, в процентах (0..200)."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 0
    if not mask.any():
        return 0.0
    return float(np.mean(numerator[mask] / denominator[mask]) * 100.0)


def _series_codes(series_ids: np.ndarray) -> np.ndarray:
    """Целочисленные коды серий (hash-таблица) — для векторного groupby."""
    codes, _ = pd.factorize(series_ids, sort=False)
    return codes


def smape(y_true: np.ndarray, y_pred: np.ndarray, series_ids: np.ndarray) -> float:
    """SMAPE по сериям (склад × товар), усреднённый.

    Векторный эквивалент цикла из ``score.py``: серия без дней с ненулевым
    знаменателем даёт 0 (среднее по NaN → fillna(0)).
    """
    if len(y_true) == 0:
        return 0.0
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    positive = denominator > 0
    ratio = np.full(len(y_true), np.nan, dtype="float64")
    np.divide(numerator, denominator, out=ratio, where=positive)
    means = pd.Series(ratio).groupby(_series_codes(series_ids)).mean().fillna(0.0)
    return float(means.mean() * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE — глобально, в процентах."""
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100.0)


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
    """Все 5 метрик для пары (факт, прогноз).

    Оба фрейма — с колонками ``date, warehouse_name, article, quantity``.
    Как в ``score.py``: merge — left от факта, пропуски прогноза → 0,
    отрицательные прогнозы клиппятся в 0. Дубликатов ключа быть не должно.
    """
    df = df_true.merge(
        df_pred,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_true", "_pred"),
    )
    nan_count = int(df["quantity_pred"].isna().sum())
    if nan_count > 0:
        logger.warning("Пропуски в прогнозе: %d строк — заполнены нулями", nan_count)
    df["quantity_pred"] = df["quantity_pred"].fillna(0.0).clip(lower=0)

    y_true = df["quantity_true"].to_numpy(dtype="float64")
    y_pred = df["quantity_pred"].to_numpy(dtype="float64")
    series_ids = (df["warehouse_name"].astype(str) + "|" + df["article"].astype(str)).to_numpy()

    return {
        "smape": smape(y_true, y_pred, series_ids),
        "wape": wape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias_abs(y_true, y_pred, series_ids),
        "rmsle": rmsle(y_true, y_pred),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Строка для консоли: ``SMAPE=…% WAPE=…% RMSE=… |Bias|=…% RMSLE=…``."""
    return (
        f"SMAPE={metrics['smape']:.2f}%  WAPE={metrics['wape']:.2f}%  RMSE={metrics['rmse']:.4f}  "
        f"|Bias|={metrics['bias']:.2f}%  RMSLE={metrics['rmsle']:.4f}"
    )
