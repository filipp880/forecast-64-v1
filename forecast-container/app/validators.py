"""Валидация входных CSV для POST /predict (п. 10.4 ТЗ).

Отрицательный ``stock`` допустим и сохраняется: это зарезервированный товар.
Повторные stock-ключи суммируются в общей предобработке, не удаляются.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
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
    raw = df["date"]
    if pd.api.types.is_datetime64_any_dtype(raw):
        dates = pd.to_datetime(raw, errors="coerce")
        invalid = dates.isna() | (dates != dates.dt.normalize())
    else:
        text = raw.astype("string")
        valid_format = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
        dates = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        invalid = ~valid_format | dates.isna()
    if invalid.any():
        errors.append(f"Неверные или пустые даты в {source}: {int(invalid.sum())} строк; ожидается YYYY-MM-DD")
        return None
    return dates


def _check_history(dates: Optional[pd.Series], source: str, errors: List[str]) -> None:
    if dates is None or dates.empty:
        return
    # Глубина календарного интервала включительно; ежедневные транзакции не обязательны.
    days = (dates.max() - dates.min()).days + 1
    if days < MIN_HISTORY_DAYS:
        errors.append(f"Недостаточно истории в {source}: {days} дней, минимум {MIN_HISTORY_DAYS}")


def validate_values(df: pd.DataFrame, source: str) -> List[str]:
    """Общие строгие значения, без требования 180 дней (для train и panel)."""
    errors: List[str] = []
    for col in ["date", "warehouse_name", "article" if "article" in df else "char_article"]:
        if col not in df:
            errors.append(f"Отсутствует {col} в {source}")
            continue
        if col != "date" and (df[col].isna() | df[col].astype("string").str.strip().eq("").fillna(True)).any():
            errors.append(f"Пустые идентификаторы {col} в {source}")
    if "date" in df:
        _parse_dates(df, source, errors)
    col = "quantity" if source == "sales" else "stock"
    if col not in df:
        errors.append(f"Отсутствует {col} в {source}")
    else:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            errors.append(f"Нечисловые, NaN или бесконечные значения {col} в {source}")
        if col == "quantity" and (numeric < 0).any():
            errors.append("Найдены отрицательные значения в quantity")
    if "article" in df and "char_article" in df:
        if not df["article"].astype("string").equals(df["char_article"].astype("string")):
            errors.append(f"Конфликт article и char_article в {source}")
    return errors


def _validate(df: pd.DataFrame, source: str, required: List[str]) -> List[str]:
    if df.empty:
        return [f"Пустой файл: {source}"]
    if df.columns.duplicated().any():
        return [f"Повторяющиеся колонки в {source}"]
    missing = [col for col in required if col not in df]
    if source == "stock" and not any(c in df for c in STOCK_ARTICLE_ALIASES):
        missing.append("article | char_article")
    if missing:
        return [f"Отсутствуют колонки в {source}: {missing}"]
    errors = validate_values(df, source)
    if not errors:
        dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
        _check_history(dates, source, errors)
    return errors


def validate_sales_csv(df: pd.DataFrame) -> List[str]:
    errors = _validate(df, "sales", SALES_REQUIRED_COLS)
    source_key = ["date", "composite_division", "cities_name", "warehouse_name", "article"]
    if not errors and df.duplicated(source_key).any():
        errors.append("Повторные ключи sales date + composite_division + cities_name + warehouse_name + article")
    return errors


def validate_stock_csv(df: pd.DataFrame) -> List[str]:
    return _validate(df, "stock", STOCK_REQUIRED_COLS)
