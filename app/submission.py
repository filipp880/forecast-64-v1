"""Strict weekly forecast contract shared by the API, generator and scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["week", "warehouse_name", "article", "quantity"]
KEY = ["warehouse_name", "article"]
FORECAST_WEEKS = 9


def _dates(values: pd.Series, name: str = "week") -> pd.Series:
    """Accept exact ISO dates or timezone-naive midnight datetime values."""
    if pd.api.types.is_datetime64_any_dtype(values):
        if getattr(values.dt, "tz", None) is not None:
            raise ValueError(f"{name}: timezone is not allowed")
        dates = pd.to_datetime(values, errors="raise")
        if dates.isna().any() or (dates != dates.dt.normalize()).any():
            raise ValueError(f"{name}: dates must be non-null calendar dates")
        return dates
    text = values.astype("string")
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).all():
        raise ValueError(f"{name}: expected YYYY-MM-DD")
    return pd.to_datetime(text, format="%Y-%m-%d", errors="raise")


def normalized_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract distinct, valid pairs without inventing a Cartesian product."""
    if not set(KEY).issubset(frame.columns):
        raise ValueError(f"Expected pair columns: {KEY}")
    pairs = frame[KEY].copy()
    for column in KEY:
        values = pairs[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"Empty pair identifier: {column}")
        pairs[column] = values
    return pairs.drop_duplicates().sort_values(KEY).reset_index(drop=True)


def validate_forecast(
    forecast: pd.DataFrame,
    expected_pairs: pd.DataFrame,
    first_week,
    weeks: int = FORECAST_WEEKS,
) -> dict:
    """Reject malformed output; never fill, clip or silently drop forecasts.

    ``expected_pairs`` is an explicit catalogue, possibly with extra columns.
    Forecast rows may be in any order; every pair/week must occur exactly once.
    """
    if isinstance(weeks, bool) or weeks != FORECAST_WEEKS:
        raise ValueError(f"Expected exactly {FORECAST_WEEKS} forecast weeks")
    if list(forecast.columns) != COLUMNS:
        raise ValueError(f"Expected columns in order: {COLUMNS}")
    if forecast.empty or forecast.isna().any().any():
        raise ValueError("Empty forecast or null values")
    start = _dates(pd.Series([first_week]), "first_week").iloc[0]
    if start.weekday() != 0:
        raise ValueError("first_week must be a Monday")
    dates = _dates(forecast["week"])
    if not dates.dt.weekday.eq(0).all():
        raise ValueError("Every week must start on Monday")
    expected_dates = pd.date_range(start, periods=weeks, freq="7D")
    if set(dates) != set(expected_dates):
        raise ValueError("Forecast must contain the exact nine consecutive weeks")
    expected = normalized_pairs(expected_pairs)
    actual = normalized_pairs(forecast)
    if expected.empty:
        raise ValueError("The expected pair catalogue is empty")
    expected_index = pd.MultiIndex.from_frame(expected)
    actual_index = pd.MultiIndex.from_frame(actual)
    missing = expected_index.difference(actual_index)
    extra = actual_index.difference(expected_index)
    if len(missing) or len(extra):
        raise ValueError(f"Pair catalogue mismatch: missing={len(missing)}, extra={len(extra)}")
    keys = forecast[KEY].astype("string").copy()
    keys["week"] = dates.to_numpy()
    if keys.duplicated(["week", *KEY]).any():
        raise ValueError("Duplicate week + warehouse_name + article keys")
    if len(forecast) != len(expected) * weeks or not keys.groupby(KEY).size().eq(weeks).all():
        raise ValueError("Incomplete pair/week coverage")
    quantity = pd.to_numeric(forecast["quantity"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(quantity).all() or (quantity < 0).any():
        raise ValueError("quantity must contain finite, nonnegative numbers")
    return {
        "rows": len(forecast),
        "pairs": len(expected),
        "warehouses": int(expected.warehouse_name.nunique()),
        "articles": int(expected.article.nunique()),
        "weeks": weeks,
        "start": start.strftime("%Y-%m-%d"),
        "end": expected_dates[-1].strftime("%Y-%m-%d"),
        "last_day": (expected_dates[-1] + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
        "duplicates": 0,
        "nan": 0,
        "nonfinite": 0,
        "negative": 0,
        "zero_share": float((quantity == 0).mean()),
        "total": float(quantity.sum()),
        "coverage_policy": "explicit_expected_pairs",
    }
