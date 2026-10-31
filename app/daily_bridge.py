"""Past-only daily targets with strict conversion to complete calendar weeks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import CAT_FEATURES,KEY,next_week_start

DAILY_HISTORY=['mean_7','mean_14','mean_28','mean_56','mean_90','zero_share_28','zero_share_90',
    'last_value','days_since_last_sale','trend_28_90','n_sale_days_90','dow_mean',
    'stock_last','stock_mean_28','stock_zero_share_90','days_of_cover']
DAILY_COLUMNS=CAT_FEATURES+DAILY_HISTORY+['h','dow','month','day','week_of_year','is_weekend','start_gap_days']
WEEK_KEY=['week',*KEY]


class DailyHistory:
    def __init__(self,sales,history_end):
        s=sales.loc[pd.to_datetime(sales.date)<=pd.Timestamp(history_end),['date',*KEY,'quantity']].copy()
        s['date']=pd.to_datetime(s.date)
        for k in KEY:s[k]=s[k].astype(str).astype('category')
        daily=s.groupby([*KEY,'date'],observed=True).quantity.sum().astype('float32')
        self.daily=daily.unstack('date')
        self.daily.index=self.daily.index.set_levels([x.astype(str) for x in self.daily.index.levels])
        self.first=s.groupby(KEY,observed=True).date.min()
        self.first.index=self.first.index.set_levels([x.astype(str) for x in self.first.index.levels])
        self.start=s.date.min();self.end=pd.Timestamp(history_end)

    def day_of_week_means(self,pairs,cutoff):
        cutoff=pd.Timestamp(cutoff)
        dates=pd.date_range(cutoff-pd.Timedelta(days=179),cutoff)
        first=self.first.reindex(pairs).to_numpy(dtype='datetime64[ns]')
        y=self.daily.reindex(index=pairs,columns=dates).fillna(0).to_numpy(float)
        known=(dates.to_numpy()[None,:]>=first[:,None])&(dates.to_numpy()[None,:]>=self.start.to_datetime64())
        means=np.full((len(pairs),7),np.nan)
        for dow in range(7):
            m=dates.weekday==dow;count=known[:,m].sum(axis=1)
            means[:,dow]=np.divide((y[:,m]*known[:,m]).sum(axis=1),count,
                out=np.full(len(pairs),np.nan),where=count>0)
        return means

    def expand(self,weekly,cutoff,training=False,label_end=None):
        cutoff=pd.Timestamp(cutoff);w=weekly.copy()
        if cutoff>self.end:raise ValueError('Feature cutoff exceeds available history')
        w['week']=pd.to_datetime(w.week)
        for k in KEY:w[k]=w[k].astype(str)
        if w.duplicated(WEEK_KEY).any():raise ValueError('Duplicate weekly source keys')
        if (w.week.dt.weekday!=0).any() or (w.week<=cutoff).any():raise ValueError('Invalid weekly calendar alignment')
        pairs=pd.MultiIndex.from_frame(w[KEY].drop_duplicates())
        means=self.day_of_week_means(pairs,cutoff)
        pair_lookup=pd.Series(np.arange(len(pairs)),index=pairs)
        d=w.iloc[np.repeat(np.arange(len(w)),7)].reset_index(drop=True)
        d['date']=d.week+pd.to_timedelta(np.tile(np.arange(7),len(w)),unit='D')
        d['h']=(d.date-cutoff).dt.days;d['dow']=d.date.dt.weekday
        d['month']=d.date.dt.month;d['day']=d.date.dt.day
        d['week_of_year']=d.date.dt.isocalendar().week.astype(int);d['is_weekend']=(d.dow>=5).astype(int)
        d['start_gap_days']=(next_week_start(cutoff)-cutoff).days
        index=pd.MultiIndex.from_frame(d[KEY]);codes=index.map(pair_lookup).to_numpy(int)
        d['dow_mean']=means[codes,d.dow.to_numpy(int)]
        d['dow_mean']=d.dow_mean.fillna(d.mean_28)
        d['feature_origin']=cutoff
        if training:
            boundary=pd.Timestamp(label_end if label_end is not None else self.end)
            if d.date.max()>boundary or boundary>self.end:raise ValueError('Daily labels exceed permitted facts')
            dates=pd.DatetimeIndex(sorted(d.date.unique()))
            values=self.daily.reindex(index=pairs,columns=dates).fillna(0).to_numpy(dtype='float32')
            col=dates.get_indexer(pd.DatetimeIndex(d.date))
            d['target']=values[codes,col]
        keep=list(dict.fromkeys(['date','week','feature_origin',*DAILY_COLUMNS]+(['target'] if training else [])))
        result=d[keep].copy()
        for c in CAT_FEATURES:result[c]=result[c].astype('category')
        for c in result.select_dtypes(include=[np.number]):result[c]=result[c].astype('float32')
        return result


def aggregate_daily(frame,prediction,expected):
    """Never invent missing daily predictions or include partial calendar weeks."""
    p=np.asarray(prediction,float)
    if p.shape!=(len(frame),) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid daily forecast')
    d=frame[['date',*KEY]].copy();d['date']=pd.to_datetime(d.date)
    for k in KEY:d[k]=d[k].astype(str)
    if d.duplicated(['date',*KEY]).any():raise ValueError('Duplicate daily prediction keys')
    d['week']=d.date-pd.to_timedelta(d.date.dt.weekday,unit='D');d['quantity']=p
    grouped=d.groupby(WEEK_KEY,observed=True).agg(quantity=('quantity','sum'),days=('date','nunique'))
    if (grouped.days!=7).any():raise ValueError('A predicted calendar week has fewer than seven days')
    reference=expected[WEEK_KEY].copy();reference['week']=pd.to_datetime(reference.week)
    for k in KEY:reference[k]=reference[k].astype(str)
    if reference.duplicated(WEEK_KEY).any():raise ValueError('Duplicate expected weekly keys')
    index=pd.MultiIndex.from_frame(reference)
    if len(index)!=len(grouped) or len(index.difference(grouped.index)) or len(grouped.index.difference(index)):
        raise ValueError('Daily and weekly forecast catalogues or periods differ')
    result=grouped.quantity.reindex(index).to_numpy(float)
    if not np.isclose(result.sum(),p.sum(),rtol=1e-12,atol=1e-8):raise ValueError('Aggregation did not preserve total quantity')
    return result
