"""Контракт submission: все наблюдаемые sales-пары, без вымышленных комбинаций."""
from __future__ import annotations
import numpy as np
import pandas as pd
COLUMNS = ['date', 'warehouse_name', 'article', 'quantity']
KEY = ['warehouse_name', 'article']

def validate_forecast(forecast: pd.DataFrame, sales: pd.DataFrame, horizon=64) -> dict:
    if list(forecast.columns) != COLUMNS:
        raise ValueError(f'Ожидаются колонки {COLUMNS}')
    if forecast.empty or forecast.isna().any().any():
        raise ValueError('Пустой прогноз или NaN')
    if not forecast.date.astype(str).str.fullmatch('\\d{4}-\\d{2}-\\d{2}').all():
        raise ValueError('Неверный формат даты прогноза')
    dates = pd.to_datetime(forecast.date, format='%Y-%m-%d', errors='raise')
    expected_dates = pd.date_range(pd.to_datetime(sales.date).max() + pd.Timedelta(days=1), periods=horizon)
    if set(dates) != set(expected_dates):
        raise ValueError('Неверные границы/неполный календарь прогноза')
    if forecast.duplicated(['date', *KEY]).any():
        raise ValueError('Повторяющиеся ключи прогноза')
    qty = pd.to_numeric(forecast.quantity, errors='coerce').to_numpy(float)
    if not np.isfinite(qty).all() or (qty < 0).any():
        raise ValueError('Некорректные quantity прогноза')
    expected = pd.MultiIndex.from_frame(sales[KEY].astype(str).drop_duplicates())
    actual = pd.MultiIndex.from_frame(forecast[KEY].astype(str).drop_duplicates())
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Не совпадает набор наблюдаемых sales-пар')
    if len(forecast) != len(expected) * horizon or not forecast.groupby(KEY).size().eq(horizon).all():
        raise ValueError('Неполное покрытие пар на горизонте')
    return dict(rows=len(forecast), pairs=len(expected), warehouses=int(forecast.warehouse_name.nunique()), articles=int(forecast.article.nunique()), days=horizon, start=expected_dates[0].strftime('%Y-%m-%d'), end=expected_dates[-1].strftime('%Y-%m-%d'), duplicates=0, nan=0, nonfinite=0, negative=0, zero_share=float((qty == 0).mean()), total=float(qty.sum()), coverage_policy='observed_sales_pairs', cartesian_rows=int(sales.warehouse_name.nunique() * sales.article.nunique() * horizon))
