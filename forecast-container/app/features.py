"""Панель данных и признаки — общие для обучения (train.py) и инференса (app/predictor.py).

Ключевая договорённость: все признаки считаются по истории не глубже
``lookback_days`` (по умолчанию 180 дней) до даты отсечки ``cutoff``.
Это требование сценария с контейнером: через API приходят только последние
~180 дней, поэтому модель обязана работать на тех же признаках без доступа
к полной обучающей выборке.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

KEY = ["warehouse_name", "article"]

CAT_FEATURES = [
    "warehouse_name",
    "article",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "characteristic",
]

HISTORY_FEATURES = [
    "mean_7",
    "mean_14",
    "mean_28",
    "mean_56",
    "mean_90",
    "zero_share_28",
    "zero_share_90",
    "last_value",
    "days_since_last_sale",
    "trend_28_90",
    "n_sale_days_90",
    "dow_mean",
    "stock_last",
    "stock_mean_28",
    "stock_zero_share_90",
    "days_of_cover",
]

CALENDAR_FEATURES = ["h", "dow", "month", "day", "week_of_year", "is_weekend"]

FEATURE_COLUMNS = CAT_FEATURES + HISTORY_FEATURES + CALENDAR_FEATURES

HORIZON = 64
PREPROCESSING_VERSION = 2
LOOKBACK_DAYS = 180
NEVER_SOLD = 999.0
HISTORY_WINDOWS = (7, 14, 28, 56, 90)

_SERIES_ATTRS = ["category_lvl_1", "category_lvl_2", "category_lvl_3", "characteristic"]


def build_panel(df_sales: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
    """Собирает панель ``date × warehouse_name × article``.

    - продажи агрегируются по департаментам/городам (ТЗ: суммировать в паре «склад × товар»);
    - сетка дат делается полной, продажи без записей заполняются нулём;
    - остаток переносится вперёд (ffill) внутри серии;
    - отрицательные остатки (≈57 строк в обучающих данных) клиппятся в 0:
      по решению судей валидация их не отклоняет, так что модель учится ровно
      в том же пространстве, в котором работает на инференсе.
    """
    from .validators import validate_values
    errors = validate_values(df_sales, "sales") + validate_values(df_stock, "stock")
    if errors:
        raise ValueError("; ".join(errors))
    sales = df_sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0.0)

    stock = df_stock.copy()
    if "char_article" in stock.columns and "article" not in stock.columns:
        stock = stock.rename(columns={"char_article": "article"})
    stock["date"] = pd.to_datetime(stock["date"])
    stock["stock"] = pd.to_numeric(stock["stock"], errors="coerce").clip(lower=0.0)

    qty = sales.groupby(["date", *KEY], as_index=False)["quantity"].sum()
    attrs = sales.groupby(KEY, as_index=False)[_SERIES_ATTRS].first()
    first_sale = qty.groupby(KEY, as_index=False)["date"].min().rename(columns={"date": "_first_sale_date"})
    attrs = attrs.merge(first_sale, on=KEY, validate="one_to_one")
    for col in _SERIES_ATTRS:
        available = sales.loc[sales[col].notna()].groupby(KEY, as_index=False)["date"].min().rename(columns={"date": f"_available_{col}"})
        attrs = attrs.merge(available, on=KEY, how="left", validate="one_to_one")
    stock_daily = stock.groupby(["date", *KEY], as_index=False)["stock"].max()
    stock_daily["_stock_observed_date"] = stock_daily["date"]

    combos = qty[KEY].drop_duplicates()
    dates = pd.date_range(qty["date"].min(), qty["date"].max(), freq="D")
    grid = combos.assign(_k=1).merge(
        pd.DataFrame({"date": dates, "_k": 1}), on="_k"
    ).drop(columns="_k")

    panel = grid.merge(qty, on=["date", *KEY], how="left")
    panel = panel.merge(stock_daily, on=["date", *KEY], how="left")
    panel = panel.merge(attrs, on=KEY, how="left")
    panel["quantity"] = panel["quantity"].fillna(0.0)
    panel = panel.sort_values([*KEY, "date"]).reset_index(drop=True)
    panel[["stock", "_stock_observed_date"]] = panel.groupby(KEY, sort=False)[["stock", "_stock_observed_date"]].ffill()
    return panel


def preprocessing_spec(stock_policy="window", lookback_days=LOOKBACK_DAYS, horizon=HORIZON) -> dict:
    """Спецификация пайплайна предобработки для ``model/preprocessor.pkl``.

    Фиксирует параметры построения признаков на момент обучения; ``predictor``
    сверяет их с кодом и ``config.json`` — защита от расхождения
    «модель обучена на одних признаках, а сервис считает другие».
    """
    return {
        "pipeline": "app.features",
        "schema_version": PREPROCESSING_VERSION,
        "stock_policy": stock_policy,
        "lookback_days": lookback_days,
        "horizon": horizon,
        "history_windows": list(HISTORY_WINDOWS),
        "never_sold": NEVER_SOLD,
        "feature_columns": list(FEATURE_COLUMNS),
        "cat_features": list(CAT_FEATURES),
    }


def _history(panel: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    return panel[
        (panel["date"] <= cutoff)
        & (panel["date"] > cutoff - pd.Timedelta(days=lookback_days))
    ]


def origin_history(panel, cutoff, lookback_days, stock_policy="window"):
    if stock_policy not in ("legacy", "window"):
        raise ValueError(f"Неизвестная политика stock: {stock_policy}")
    hist = _history(panel, cutoff, lookback_days)
    if stock_policy == "window":
        hist = hist.copy()
        # На входе API нет наблюдений до левой границы окна.
        valid = hist["_stock_observed_date"] > cutoff - pd.Timedelta(days=lookback_days)
        hist["stock"] = hist["stock"].where(valid)
    return hist


def _share_zero(frame: pd.DataFrame, column: str, series_index: pd.MultiIndex) -> pd.Series:
    flag = frame[[*KEY, column]].assign(
        _flag=(frame[column] == 0).astype("float32")
    )
    return flag.groupby(KEY, sort=False)["_flag"].mean().reindex(series_index)


def series_features(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    lookback_days: int = LOOKBACK_DAYS,
    stock_policy: str = "window",
) -> pd.DataFrame:
    """Признаки уровня серии, посчитанные строго по данным до ``cutoff`` включительно."""
    hist = origin_history(panel, cutoff, lookback_days, stock_policy)
    if hist.empty:
        raise ValueError(
            f"Нет истории до {pd.Timestamp(cutoff).date()} — признаки не посчитать"
        )

    at_cutoff = hist[hist["date"] == cutoff].set_index(KEY)
    if at_cutoff.empty:
        raise ValueError("Нет снимка на дату отсечки")
    series_index = at_cutoff.index
    feats = pd.DataFrame(index=series_index)
    feats["last_value"] = at_cutoff["quantity"].reindex(series_index)
    feats["stock_last"] = at_cutoff["stock"].reindex(series_index)

    for window in HISTORY_WINDOWS:
        win = _history(hist, cutoff, window)
        feats[f"mean_{window}"] = (
            win.groupby(KEY, sort=False)["quantity"].mean().reindex(series_index)
        )

    win28 = _history(hist, cutoff, 28)
    win90 = _history(hist, cutoff, 90)

    feats["zero_share_28"] = _share_zero(win28, "quantity", series_index)
    feats["zero_share_90"] = _share_zero(win90, "quantity", series_index)

    win90_pos = win90[[*KEY, "quantity"]].assign(
        _flag=(win90["quantity"] > 0).astype("float32")
    )
    feats["n_sale_days_90"] = (
        win90_pos.groupby(KEY, sort=False)["_flag"].sum().reindex(series_index)
    )

    last_sale = (
        hist.loc[hist["quantity"] > 0]
        .groupby(KEY, sort=False)["date"]
        .max()
        .reindex(series_index)
    )
    feats["days_since_last_sale"] = (cutoff - last_sale).dt.days.fillna(NEVER_SOLD)

    feats["trend_28_90"] = feats["mean_28"] / (feats["mean_90"] + 1e-6)

    feats["stock_mean_28"] = (
        win28.groupby(KEY, sort=False)["stock"].mean().reindex(series_index)
    )
    stock90 = win90.dropna(subset=["stock"])
    feats["stock_zero_share_90"] = _share_zero(stock90, "stock", series_index)
    feats["days_of_cover"] = feats["stock_last"] / np.maximum(feats["mean_28"], 0.1)

    fill_zero = [
        col
        for col in HISTORY_FEATURES
        if not col.startswith("stock_") and col not in ("days_of_cover", "dow_mean")
    ]
    feats[fill_zero] = feats[fill_zero].fillna(0.0)

    attrs = at_cutoff[_SERIES_ATTRS].reindex(series_index).copy()
    if stock_policy == "window":
        for col in _SERIES_ATTRS:
            marker = f"_available_{col}"
            if marker in panel:
                available = at_cutoff[marker].reindex(series_index)
                attrs[col] = attrs[col].where(available <= cutoff)
    return feats.join(attrs)


def make_features(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizons: Sequence[int],
    lookback_days: int = LOOKBACK_DAYS,
    stock_policy: str = "window",
) -> pd.DataFrame:
    """Длинная рамка признаков: одна строка = серия × горизонт ``h``.

    Календарные признаки берутся с целевого дня (``cutoff + h``), все
    остальные — строго «по состоянию на ``cutoff``».
    """
    cutoff = pd.Timestamp(cutoff)
    if stock_policy == "window" and "_first_sale_date" in panel:
        panel = panel.loc[panel["_first_sale_date"] <= cutoff]
    horizons = list(horizons)
    hist = origin_history(panel, cutoff, lookback_days, stock_policy)

    dow_stats = (
        hist.assign(dow=hist["date"].dt.dayofweek)
        .groupby(KEY + ["dow"], sort=False)["quantity"]
        .mean()
        .rename("dow_mean")
        .reset_index()
    )

    target_days = pd.DataFrame({"h": horizons})
    target_days["date"] = cutoff + pd.to_timedelta(target_days["h"], unit="D")
    target_days["dow"] = target_days["date"].dt.dayofweek
    target_days["month"] = target_days["date"].dt.month
    target_days["day"] = target_days["date"].dt.day
    target_days["week_of_year"] = (
        target_days["date"].dt.isocalendar().week.astype(int)
    )
    target_days["is_weekend"] = (target_days["dow"] >= 5).astype(int)

    feats = series_features(panel, cutoff, lookback_days, stock_policy).reset_index()
    feats = feats.assign(_k=1).merge(target_days.assign(_k=1), on="_k").drop(columns="_k")
    feats = feats.merge(dow_stats, on=KEY + ["dow"], how="left")
    feats["dow_mean"] = feats["dow_mean"].fillna(feats["mean_28"])

    for col in CAT_FEATURES:
        feats[col] = feats[col].fillna("na").astype(str)

    feats[HISTORY_FEATURES] = feats[HISTORY_FEATURES].astype("float32")
    feats["h"] = feats["h"].astype(int)

    columns = list(dict.fromkeys([*KEY, "date", *FEATURE_COLUMNS]))
    return feats[columns].sort_values([*KEY, "h"]).reset_index(drop=True)


def make_training_frame(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizons: Sequence[int],
    lookback_days: int = LOOKBACK_DAYS,
    stock_policy: str = "window",
) -> pd.DataFrame:
    """Признаки + таргет ``target`` (факт на ``cutoff + h``; нет записи → 0).

    Горизонты, выходящие за пределы данных, передавать нельзя: они получат
    ложный таргет 0. Отбор валидных горизонтов — на стороне вызывающего кода.
    """
    horizons = list(horizons)
    if not horizons or min(horizons) < 1 or pd.Timestamp(cutoff) + pd.Timedelta(days=max(horizons)) > panel["date"].max():
        raise ValueError("Обучающие горизонты выходят за границу доступных фактов")
    frame = make_features(panel, cutoff, horizons, lookback_days, stock_policy)
    target = panel[["date", *KEY, "quantity"]]
    frame = frame.merge(target, on=["date", *KEY], how="left")
    frame["target"] = frame["quantity"].fillna(0.0)
    return frame.drop(columns=["quantity"])
