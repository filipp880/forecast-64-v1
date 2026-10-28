"""Explicit full-product submission coverage and cold-pair fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .submission import KEY, normalized_pairs, validate_forecast


def cartesian_pairs(sales):
    observed=normalized_pairs(sales)
    return pd.MultiIndex.from_product(
        [sorted(observed.warehouse_name.unique()),sorted(observed.article.unique())],names=KEY).to_frame(index=False)


def complete_cartesian_forecast(forecast,sales,cutoff,first_week,round_int=True):
    """Preserve every observed-pair prediction and add only never-observed pairs.

    The existing no-history fallback is mean_week_4 of the article across its
    observed warehouses: 7 * mean(mean_28). Missing warehouse histories count
    only when that warehouse/article pair has previously been observed.
    """
    cutoff=pd.Timestamp(cutoff)
    sales=sales.loc[pd.to_datetime(sales.date)<=cutoff].copy()
    sales['date']=pd.to_datetime(sales.date)
    sales['quantity']=pd.to_numeric(sales.quantity,errors='raise').astype(float)
    if not np.isfinite(sales.quantity).all() or sales.quantity.lt(0).any():raise ValueError('Invalid history quantity')
    observed=normalized_pairs(sales)
    validate_forecast(forecast,observed,first_week)
    pairs=cartesian_pairs(sales)
    observed_index=pd.MultiIndex.from_frame(observed)
    missing=pd.MultiIndex.from_frame(pairs).difference(observed_index).to_frame(index=False)
    if missing.empty:return forecast.copy(),pairs,missing
    if sales.date.min()>cutoff-pd.Timedelta(days=27):raise ValueError('Need at least 28 calendar days for fallback')
    recent=sales.loc[sales.date>cutoff-pd.Timedelta(days=28)]
    totals=recent.groupby('article',observed=True).quantity.sum()
    warehouse_count=observed.groupby('article',observed=True).size()
    weekly=totals.reindex(warehouse_count.index,fill_value=0)/(4*warehouse_count)
    values=missing.article.map(weekly).to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid cold-pair fallback')
    if round_int:values=np.rint(values)
    missing['fallback_quantity']=values
    weeks=pd.DataFrame({'week':pd.date_range(first_week,periods=9,freq='7D').strftime('%Y-%m-%d')})
    extra=missing.rename(columns={'fallback_quantity':'quantity'}).merge(weeks,how='cross')
    extra=extra[['week',*KEY,'quantity']]
    result=pd.concat([forecast,extra],ignore_index=True).sort_values(['week',*KEY]).reset_index(drop=True)
    validate_forecast(result,pairs,first_week)
    return result,pairs,missing
