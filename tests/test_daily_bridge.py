"""Calendar-week aggregation, truth boundaries and past-only daily features; no fitting."""
import numpy as np
import pandas as pd
import pytest
from app.daily_bridge import DailyHistory,aggregate_daily,DAILY_HISTORY

def inputs(cutoff='2026-01-18'):
    cutoff=pd.Timestamp(cutoff)
    s=pd.DataFrame({'date':pd.date_range(cutoff-pd.Timedelta(days=200),cutoff+pd.Timedelta(days=63)),
        'warehouse_name':'W','article':'A','quantity':1.})
    w=pd.DataFrame({'week':pd.date_range(cutoff-pd.Timedelta(days=cutoff.weekday())+pd.Timedelta(days=7),periods=9,freq='7D'),
        'warehouse_name':'W','article':'A'})
    for c in ['category_lvl_1','category_lvl_2','category_lvl_3','characteristic']:w[c]='C'
    for c in DAILY_HISTORY:
        if c!='dow_mean':w[c]=1.
    return s,w

def test_sixty_three_days_sum_to_nine_calendar_weeks_in_reference_order():
    s,w=inputs();h=DailyHistory(s,'2026-03-22');d=h.expand(w,'2026-01-18',training=True)
    assert len(d)==63 and d.h.min()==1 and d.h.max()==63
    p=aggregate_daily(d,d.target,w.iloc[::-1])
    np.testing.assert_array_equal(p,np.full(9,7.))

@pytest.mark.parametrize('error',['missing','duplicate','nan','negative','wrong_catalogue'])
def test_bad_daily_predictions_cannot_be_silently_fixed(error):
    s,w=inputs();d=DailyHistory(s,'2026-03-22').expand(w,'2026-01-18');p=np.ones(len(d))
    if error=='missing':d=d.iloc[:-1];p=p[:-1]
    if error=='duplicate':d=pd.concat([d,d.iloc[:1]],ignore_index=True);p=np.ones(len(d))
    if error=='nan':p[0]=np.nan
    if error=='negative':p[0]=-1
    if error=='wrong_catalogue':w=w.assign(article='B')
    with pytest.raises(ValueError):aggregate_daily(d,p,w)

def test_future_sales_do_not_change_day_of_week_features():
    s,w=inputs();changed=s.copy();changed.loc[changed.date>'2026-01-18','quantity']=999999.
    a=DailyHistory(s,'2026-03-22').expand(w,'2026-01-18')
    b=DailyHistory(changed,'2026-03-22').expand(w,'2026-01-18')
    pd.testing.assert_frame_equal(a,b)

def test_unavailable_labels_raise_instead_of_becoming_zeros():
    s,w=inputs();h=DailyHistory(s,'2026-02-01')
    with pytest.raises(ValueError,match='labels exceed'):h.expand(w,'2026-01-18',training=True)
    valid=w.loc[w.week+pd.Timedelta(days=6)<='2026-02-01']
    d=h.expand(valid,'2026-01-18',training=True)
    assert len(d)==14 and d.target.sum()==14

def test_non_sunday_api_cutoff_still_predicts_nine_complete_calendar_weeks():
    s,w=inputs('2026-01-20');d=DailyHistory(s,'2026-01-20').expand(w,'2026-01-20')
    assert d.date.min()==pd.Timestamp('2026-01-26') and d.date.max()==pd.Timestamp('2026-03-29')
    assert len(d)==63 and d.h.min()==6 and d.h.max()==68
    assert len(aggregate_daily(d,np.ones(63),w))==9
