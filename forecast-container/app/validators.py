"""Валидация входных CSV для POST /predict (п. 10.4 ТЗ)."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

# Поля cost_price / volume_weight отсутствуют в текущей выгрузке организаторов
# (см. data/нюансы.txt), поэтому они объявлены необязательными.
SALES_REQUIRED_COLS = [
    "date",
    "composite_division",
    "cities_name",
    "warehouse_name",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "article",
    "characteristic",
    "quantity",
    "sales_sum",
    "discount_sum",
]
SALES_OPTIONAL_COLS = ["cost_price", "volume_weight"]

STOCK_REQUIRED_COLS = ["date", "warehouse_name", "stock"]
STOCK_ARTICLE_ALIASES = ("article", "char_article")

MIN_HISTORY_DAYS = 180


def _parse_dates(df: pd.DataFrame, source: str, errors: List[str]) -> Optional[pd.Series]:
    try:
        return pd.to_datetime(df["date"], format="%Y-%m-%d")
    except Exception:
        errors.append(f"Неверный формат даты в {source} (ожидается YYYY-MM-DD)")
        return None


def _check_history(dates: Optional[pd.Series], source: str, errors: List[str]) -> None:
    if dates is None or dates.empty:
        return
    span_days = (dates.max() - dates.min()).days
    unique_days = int(dates.nunique())
    if span_days >= MIN_HISTORY_DAYS or unique_days >= MIN_HISTORY_DAYS:
        return
    errors.append(
        f"Недостаточно истории в {source}: {unique_days} дн. "
        f"(интервал {span_days + 1} дн.), минимум {MIN_HISTORY_DAYS}"
    )


def validate_sales_csv(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    missing = [col for col in SALES_REQUIRED_COLS if col not in df.columns]
    if missing:
        errors.append(f"Отсутствуют колонки в sales: {missing}")
        return errors

    dates = _parse_dates(df, "sales", errors)
    if dates is None:
        return errors
    _check_history(dates, "sales", errors)

    quantity = pd.to_numeric(df["quantity"], errors="coerce")
    if (quantity < 0).any():
        errors.append("Найдены отрицательные значения в quantity")
    return errors


def validate_stock_csv(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    missing = [col for col in STOCK_REQUIRED_COLS if col not in df.columns]
    if not any(alias in df.columns for alias in STOCK_ARTICLE_ALIASES):
        missing.append("article | char_article")
    if missing:
        errors.append(f"Отсутствуют колонки в stock: {missing}")
        return errors

    dates = _parse_dates(df, "stock", errors)
    if dates is None:
        return errors
    _check_history(dates, "stock", errors)

    stock = pd.to_numeric(df["stock"], errors="coerce")
    if (stock < 0).any():
        errors.append("Найдены отрицательные значения в stock")
    return errors
