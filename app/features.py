"""Shared as-of daily panel and complete weekly targets for training/serving."""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ["warehouse_name", "article"]
CAT_FEATURES = KEY + ["category_lvl_1", "category_lvl_2", "category_lvl_3", "characteristic"]
HISTORY_WINDOWS = (7, 14, 28, 56, 90)
HISTORY_FEATURES = [
    *[f"mean_{w}" for w in HISTORY_WINDOWS], "zero_share_28", "zero_share_90",
    "last_value", "days_since_last_sale", "trend_28_90", "n_sale_days_90",
    "stock_last", "stock_mean_28", "stock_zero_share_90", "stock_negative_share_90",
    "stock_reserved", "days_of_cover", "last_week", "mean_week_4", "mean_week_8",
    "mean_week_13", "zero_week_share_13", "mean_all", "history_days", "no_history",
]
CALENDAR_FEATURES = ["h", "h_week", "month", "week_of_year", "year", "start_gap_days"]
FEATURE_COLUMNS = CAT_FEATURES + HISTORY_FEATURES + CALENDAR_FEATURES
FORECAST_WEEKS = HORIZON = 9
PREPROCESSING_VERSION = 3
LOOKBACK_DAYS = 180
NEVER_SOLD = 999.0
COMPETITION_CUTOFF = pd.Timestamp("2026-07-26")
COMPETITION_FIRST_WEEK = pd.Timestamp("2026-07-27")
_SERIES_ATTRS = CAT_FEATURES[2:]


def next_week_start(last_date):
    last = pd.Timestamp(last_date).normalize()
    return last - pd.Timedelta(days=last.weekday()) + pd.Timedelta(days=7)


def forecast_end(cutoff, weeks=FORECAST_WEEKS, first_week=None):
    return pd.Timestamp(first_week if first_week is not None else next_week_start(cutoff)) + pd.Timedelta(days=7 * weeks - 1)


def truncate_history(sales, stock, cutoff=COMPETITION_CUTOFF):
    """Trim BOTH sources before constructing categories, features or training."""
    cut = pd.Timestamp(cutoff).normalize()
    return tuple(f.loc[pd.to_datetime(f["date"], errors="raise") <= cut].copy() for f in (sales, stock))


def build_panel(df_sales, df_stock, history_end=None):
    """Daily sales flow and signed opening stock; only within-day stock duplicates sum."""
    from .validators import validate_values
    errors = validate_values(df_sales, "sales") + validate_values(df_stock, "stock")
    if errors:
        raise ValueError("; ".join(errors))
    if df_sales.empty:
        raise ValueError("Нет истории продаж до отсечки")
    sales = df_sales[["date", *KEY, "quantity", *_SERIES_ATTRS]].copy()
    sales["date"] = pd.to_datetime(sales.date)
    sales["quantity"] = pd.to_numeric(sales.quantity, errors="raise")
    sales = sales.sort_values("date", kind="stable")
    stock = df_stock.rename(columns={"char_article": "article"}) if "article" not in df_stock else df_stock
    stock = stock[["date", *KEY, "stock"]].copy()
    stock["date"] = pd.to_datetime(stock.date)
    stock["stock"] = pd.to_numeric(stock.stock, errors="raise")
    for c in KEY:
        dtype = pd.CategoricalDtype(sorted(sales[c].astype(str).unique()))
        sales[c] = sales[c].astype(str).astype(dtype)
        stock[c] = stock[c].astype(str).astype(dtype)
    stock = stock.dropna(subset=KEY)
    daily = sales.groupby(["date", *KEY], observed=True, as_index=False).agg(
        quantity=("quantity", "sum"), **{c: (c, "first") for c in _SERIES_ATTRS})
    inventory = stock.groupby(["date", *KEY], observed=True, as_index=False)["stock"].sum()
    inventory["_stock_observed_date"] = inventory.date
    combos = daily[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
    end = pd.Timestamp(history_end) if history_end is not None else daily.date.max()
    if end < daily.date.max():
        raise ValueError("Обрежьте историю перед build_panel")
    dates = pd.date_range(daily.date.min(), end, freq="D")
    grid = combos.iloc[np.repeat(np.arange(len(combos)), len(dates))].reset_index(drop=True)
    grid["date"] = np.tile(dates.to_numpy(), len(combos))
    panel = grid.merge(daily, on=["date", *KEY], how="left", validate="one_to_one")
    panel = panel.merge(inventory, on=["date", *KEY], how="left", validate="one_to_one")
    first_sale = daily.groupby(KEY, observed=True).date.min().rename("_first_sale_date")
    panel = panel.join(first_sale, on=KEY)
    panel["quantity"] = panel.quantity.fillna(0).astype("float32")
    panel["stock"] = panel.stock.astype("float32")
    for c in _SERIES_ATTRS:
        panel[c] = panel[c].astype("category")
    carry = ["stock", "_stock_observed_date", *_SERIES_ATTRS]
    panel[carry] = panel.groupby(KEY, observed=True, sort=False)[carry].ffill()
    return panel


def preprocessing_spec(stock_policy="window", lookback_days=LOOKBACK_DAYS, horizon=HORIZON):
    if stock_policy != "window":
        raise ValueError("Weekly preprocessing supports stock_policy=window only")
    return dict(pipeline="app.features", schema_version=PREPROCESSING_VERSION,
                granularity="week", forecast_weeks=FORECAST_WEEKS,
                stock_aggregation="sum", negative_stock_policy="preserve",
                forecast_alignment="next_calendar_monday", stock_policy=stock_policy,
                lookback_days=lookback_days, horizon=horizon,
                history_windows=list(HISTORY_WINDOWS), never_sold=NEVER_SOLD,
                feature_columns=list(FEATURE_COLUMNS), cat_features=list(CAT_FEATURES),
                long_history_policy="all_available_mean")


def _history(panel, cutoff, lookback_days):
    return panel[(panel.date <= cutoff) & (panel.date > cutoff - pd.Timedelta(days=lookback_days))]


def origin_history(panel, cutoff, lookback_days, stock_policy="window"):
    if stock_policy != "window":
        raise ValueError("Legacy stock preprocessing is incompatible with weekly models")
    hist = _history(panel, cutoff, lookback_days).copy()
    hist["stock"] = hist.stock.where(hist._stock_observed_date > cutoff - pd.Timedelta(days=lookback_days))
    return hist


def series_features(panel, cutoff, lookback_days=LOOKBACK_DAYS, stock_policy="window"):
    cutoff = pd.Timestamp(cutoff)
    known = panel.loc[(panel._first_sale_date <= cutoff) & (panel.date <= cutoff)]
    hist = origin_history(known, cutoff, lookback_days, stock_policy)
    at = hist.loc[hist.date == cutoff].set_index(KEY)
    if at.empty:
        raise ValueError("Нет снимка истории на дату отсечки")
    feats = at[_SERIES_ATTRS].copy()
    index = feats.index
    group = lambda f: f.groupby(KEY, observed=True, sort=False)
    feats["last_value"], feats["stock_last"] = at.quantity, at.stock
    for w in HISTORY_WINDOWS:
        feats[f"mean_{w}"] = group(_history(hist, cutoff, w)).quantity.mean().reindex(index)
    for w in (28, 90):
        win = _history(hist, cutoff, w)
        feats[f"zero_share_{w}"] = group(win.assign(_zero=(win.quantity == 0).astype(float)))._zero.mean().reindex(index)
    win28, win90 = _history(hist, cutoff, 28), _history(hist, cutoff, 90)
    feats["n_sale_days_90"] = group(win90.assign(_positive=(win90.quantity > 0).astype(float)))._positive.sum().reindex(index)
    last_sale = group(known.loc[known.quantity > 0]).date.max().reindex(index)
    feats["days_since_last_sale"] = (cutoff - last_sale).dt.days.fillna(NEVER_SOLD)
    feats["trend_28_90"] = feats.mean_28 / np.maximum(feats.mean_90, .001)
    feats["stock_mean_28"] = group(win28).stock.mean().reindex(index)
    stocks = win90.dropna(subset=["stock"])
    for name, flag in (("stock_zero_share_90", stocks.stock == 0), ("stock_negative_share_90", stocks.stock < 0)):
        feats[name] = group(stocks.assign(_flag=flag.astype(float)))._flag.mean().reindex(index)
    feats["stock_reserved"] = -np.minimum(feats.stock_last, 0)
    feats["days_of_cover"] = np.maximum(feats.stock_last, 0) / np.maximum(feats.mean_28, .1)
    last_sunday = cutoff - pd.Timedelta(days=(cutoff.weekday() + 1) % 7)
    for n, name in ((1, "last_week"), (4, "mean_week_4"), (8, "mean_week_8"), (13, "mean_week_13")):
        feats[name] = group(_history(known, last_sunday, 7 * n)).quantity.sum().reindex(index).fillna(0) / n
    weekly = _history(known, last_sunday, 91)[["date", *KEY, "quantity"]].copy()
    weekly["week"] = weekly.date - pd.to_timedelta(weekly.date.dt.weekday, unit="D")
    sums = weekly.groupby([*KEY, "week"], observed=True).quantity.sum().reset_index()
    feats["zero_week_share_13"] = group(sums.assign(_zero=(sums.quantity == 0).astype(float)))._zero.mean().reindex(index)
    # Long context uses all provided history even when input exceeds 180 days.
    feats["mean_all"] = group(known).quantity.mean().reindex(index)
    feats["history_days"] = (cutoff - at._first_sale_date).dt.days + 1
    feats["no_history"] = 0.
    return feats


def _with_required_pairs(feats, required_pairs):
    if required_pairs is None:
        return feats
    from .submission import normalized_pairs
    pairs = normalized_pairs(required_pairs)
    source = feats.reset_index()
    target = pairs.merge(source, on=KEY, how="left", validate="one_to_one")
    missing = target.no_history.isna()
    if missing.any():
        item = source.groupby("article", observed=True).mean_28.mean()
        warehouse = source.groupby("warehouse_name", observed=True).mean_28.mean()
        daily = target.article.map(item).fillna(target.warehouse_name.map(warehouse)).fillna(float(source.mean_28.mean()))
        for c in HISTORY_FEATURES:
            target.loc[missing, c] = np.nan if c.startswith("stock_") or c == "days_of_cover" else 0.
        for w in HISTORY_WINDOWS:
            target.loc[missing, f"mean_{w}"] = daily[missing]
        target.loc[missing, "mean_all"] = daily[missing]
        for c in ("last_week", "mean_week_4", "mean_week_8", "mean_week_13"):
            target.loc[missing, c] = 7 * daily[missing]
        for c in ("zero_share_28", "zero_share_90", "zero_week_share_13"):
            target.loc[missing, c] = 1.
        target.loc[missing, "days_since_last_sale"] = NEVER_SOLD
        target.loc[missing, "no_history"] = 1.
        for c in _SERIES_ATTRS:
            attrs = source.dropna(subset=[c]).drop_duplicates("article").set_index("article")[c]
            target[c] = target[c].astype(object)
            target.loc[missing, c] = target.loc[missing, "article"].map(attrs)
    return target.set_index(KEY)


def make_features(panel, cutoff, horizons=range(1, HORIZON + 1), lookback_days=LOOKBACK_DAYS,
                  stock_policy="window", first_week=None, required_pairs=None):
    cutoff = pd.Timestamp(cutoff).normalize()
    horizons = list(horizons)
    if not horizons or any(int(h) != h or h < 1 or h > FORECAST_WEEKS for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("Горизонты должны быть уникальными номерами недель 1..9")
    first = pd.Timestamp(first_week if first_week is not None else next_week_start(cutoff))
    if first != first.normalize() or first.weekday() != 0 or first <= cutoff:
        raise ValueError("Первая неделя должна начинаться в понедельник после отсечки")
    feats = _with_required_pairs(series_features(panel, cutoff, lookback_days, stock_policy), required_pairs).reset_index()
    weeks = pd.DataFrame({"h": horizons, "h_week": horizons})
    weeks["week"] = first + pd.to_timedelta((weeks.h - 1) * 7, unit="D")
    weeks["month"] = weeks.week.dt.month
    weeks["week_of_year"] = weeks.week.dt.isocalendar().week.astype(int)
    weeks["year"] = weeks.week.dt.year
    weeks["start_gap_days"] = (first - cutoff).days
    result = feats.merge(weeks, how="cross")
    for c in CAT_FEATURES:
        result[c] = result[c].astype("string").fillna("na").astype(str)
    result[HISTORY_FEATURES] = result[HISTORY_FEATURES].astype("float32")
    return result[list(dict.fromkeys(["week", *FEATURE_COLUMNS]))].sort_values([*KEY, "h"]).reset_index(drop=True)


def make_training_frame(panel, cutoff, horizons=range(1, HORIZON + 1), lookback_days=LOOKBACK_DAYS,
                        stock_policy="window", first_week=None, required_pairs=None):
    horizons = list(horizons)
    first = pd.Timestamp(first_week if first_week is not None else next_week_start(cutoff))
    if not horizons or first + pd.Timedelta(days=7 * max(horizons) - 1) > panel.date.max():
        raise ValueError("Обучающие недели выходят за границу полных доступных фактов")
    frame = make_features(panel, cutoff, horizons, lookback_days, stock_policy, first, required_pairs)
    target = panel.loc[(panel.date >= first) & (panel.date <= first + pd.Timedelta(days=7 * max(horizons) - 1)), ["date", *KEY, "quantity"]].copy()
    target["week"] = target.date - pd.to_timedelta(target.date.dt.weekday, unit="D")
    target = target.groupby(["week", *KEY], observed=True, as_index=False).quantity.sum().rename(columns={"quantity": "target"})
    frame = frame.merge(target, on=["week", *KEY], how="left", validate="one_to_one")
    frame["target"] = frame.target.fillna(0)
    return frame


def monetary_weekly_features(sales):
    """Optional exp-before-sum finances, with null discounts contributing zero."""
    frame = sales[["date", *KEY, "sales_sum", "discount_sum"]].copy()
    frame["week"] = pd.to_datetime(frame.date) - pd.to_timedelta(pd.to_datetime(frame.date).dt.weekday, unit="D")
    for c in ("sales_sum", "discount_sum"):
        values = pd.to_numeric(frame[c], errors="raise")
        frame[c] = np.exp(values).where(values.notna(), 0.)
        if not np.isfinite(frame[c]).all():
            raise ValueError(f"Неконечные значения после exp: {c}")
    return frame.groupby(["week", *KEY], as_index=False)[["sales_sum", "discount_sum"]].sum()
