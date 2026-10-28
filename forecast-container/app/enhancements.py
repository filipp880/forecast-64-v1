"""As-of feature ablations shared by research training and serving."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import KEY, FEATURE_COLUMNS, make_features
from .ru_calendar import CALENDAR_VERSION, calendar_days, weekly_calendar

CALENDAR_COLUMNS = ["ru_workdays","ru_holidays","ru_working_saturdays","ru_transferred","ru_transfer_known",
                    "ru_long_break","ru_days_after_holiday","ru_days_before_holiday","year_sin","year_cos","month_cross",
                    "sales_per_workday_90","sales_per_offday_90"]
DYNAMIC_COLUMNS = ["ewma_week_4","ewma_week_13","std_week_13","positive_cv2_13","adi_13","week_trend_4_13",
                   "tsb_week","sba_week","article_week_mean","warehouse_week_mean","category_week_mean","category_share"]
MONEY_COLUMNS = ["log_revenue_28","log_discount_28","log_unit_revenue_28","discount_fraction_28","revenue_trend","money_known"]
PROFILES = {"base":[],"calendar":CALENDAR_COLUMNS,"dynamics":DYNAMIC_COLUMNS,
            "monetary":MONEY_COLUMNS,"all":CALENDAR_COLUMNS+DYNAMIC_COLUMNS+MONEY_COLUMNS}


def feature_columns(profile="all"):
    return list(FEATURE_COLUMNS)+list(PROFILES[profile])


def add_money(panel, sales):
    """Invert provided np.log BEFORE aggregation; null discount contributes 0."""
    if not {"sales_sum","discount_sum"} <= set(sales):
        return panel
    frame=sales[["date",*KEY,"sales_sum","discount_sum"]].copy()
    frame["date"]=pd.to_datetime(frame.date)
    for c in ("sales_sum","discount_sum"):
        v=pd.to_numeric(frame[c],errors="raise")
        frame[c]=np.exp(v).where(v.notna(),0.)
        if not np.isfinite(frame[c]).all():raise ValueError("Nonfinite monetary exp")
    for c in KEY:frame[c]=frame[c].astype(panel[c].dtype)
    money=frame.groupby(["date",*KEY],observed=True,as_index=False)[["sales_sum","discount_sum"]].sum()
    return panel.merge(money,on=["date",*KEY],how="left",validate="one_to_one")


def intermittent(weekly, alpha=.1):
    """Vectorized TSB and Croston-SBA, in units per WEEK (no arbitrary /2)."""
    y=np.asarray(weekly,float)
    p=np.zeros(len(y));z=np.zeros(len(y));q=np.ones(len(y));age=np.ones(len(y));seen=np.zeros(len(y),bool)
    for col in y.T:
        positive=col>0
        first=positive & ~seen
        p=np.where(first,1.,(1-alpha)*p+alpha*positive)
        z=np.where(first,col,np.where(positive,(1-alpha)*z+alpha*col,z))
        q=np.where(first,age,np.where(positive,(1-alpha)*q+alpha*age,q))
        seen|=positive
        age=np.where(positive,1.,age+1)
    return p*z,(1-alpha/2)*z/np.maximum(q,1.)


def augment_features(panel, frame, cutoff):
    cutoff=pd.Timestamp(cutoff)
    known=panel.loc[(panel.date<=cutoff)&(panel._first_sale_date<=cutoff)]
    pairs=frame[KEY].drop_duplicates().set_index(KEY).index
    end=cutoff-pd.Timedelta(days=(cutoff.weekday()+1)%7)
    hist=known.loc[(known.date>end-pd.Timedelta(days=91))&(known.date<=end),["date",*KEY,"quantity"]].copy()
    hist["week"]=hist.date-pd.to_timedelta(hist.date.dt.weekday,unit="D")
    sums=hist.groupby([*KEY,"week"],observed=True).quantity.sum().unstack("week")
    weeks=pd.date_range(end-pd.Timedelta(days=90),end-pd.Timedelta(days=6),freq="7D")
    y=sums.reindex(index=pairs,columns=weeks).fillna(0).to_numpy(float)
    extra=pd.DataFrame(index=pairs)
    positive=(y>0); n=positive.sum(axis=1)
    avg=y.sum(axis=1)/np.maximum(n,1)
    var=((y-avg[:,None])**2*positive).sum(axis=1)/np.maximum(n,1)
    for span in (4,13):
        level=y[:,0].copy();a=2/(span+1)
        for value in y[:,1:].T:level=a*value+(1-a)*level
        extra[f"ewma_week_{span}"]=level
    extra["std_week_13"]=y.std(axis=1)
    extra["positive_cv2_13"]=var/np.maximum(avg**2,.001)
    extra["adi_13"]=13/np.maximum(n,1)
    extra["week_trend_4_13"]=y[:,-4:].mean(axis=1)/np.maximum(y.mean(axis=1),.01)
    extra["tsb_week"],extra["sba_week"]=intermittent(y)
    past=known.loc[known.date>cutoff-pd.Timedelta(days=90)]
    cal=calendar_days(pd.DatetimeIndex(past.date.unique()),cutoff)
    work=past.date.map(cal.working).to_numpy()
    for name,mask in (("sales_per_workday_90",work>0),("sales_per_offday_90",work==0)):
        extra[name]=past.loc[mask].groupby(KEY,observed=True).quantity.mean().reindex(pairs)
    if "sales_sum" in known:
        for days in (28,90):
            recent=known.loc[known.date>cutoff-pd.Timedelta(days=days)]
            total=recent.groupby(KEY,observed=True)[["quantity","sales_sum","discount_sum"]].sum(min_count=1).reindex(pairs)
            if days==28:
                extra["log_revenue_28"]=np.log1p(total.sales_sum)
                extra["log_discount_28"]=np.log1p(total.discount_sum)
                extra["log_unit_revenue_28"]=np.log1p(total.sales_sum/np.maximum(total.quantity,1))
                extra["discount_fraction_28"]=total.discount_sum/np.maximum(total.sales_sum+total.discount_sum,1e-9)
                extra["money_known"]=total.sales_sum.notna().astype(float)
                rev=total.sales_sum/28
            else:extra["revenue_trend"]=rev/np.maximum(total.sales_sum/90,1e-9)
    else:
        for c in MONEY_COLUMNS:extra[c]=0. if c=="money_known" else np.nan
    base=frame.drop_duplicates(KEY).set_index(KEY)
    volume=base.mean_28*7
    extra["article_week_mean"]=volume.groupby(level="article").transform("mean")
    extra["warehouse_week_mean"]=volume.groupby(level="warehouse_name").transform("mean")
    extra["category_week_mean"]=volume.groupby(base.category_lvl_1).transform("mean")
    denom=volume.groupby(base.category_lvl_1).transform("sum")
    extra["category_share"]=volume/np.maximum(denom,.01)
    result=frame.merge(extra.reset_index(),on=KEY,how="left",validate="many_to_one")
    future=weekly_calendar(result.week.unique(),cutoff)
    result=result.merge(future.reset_index(),on="week",how="left",validate="many_to_one")
    for c in CALENDAR_COLUMNS+DYNAMIC_COLUMNS+MONEY_COLUMNS:
        result[c]=result[c].replace([np.inf,-np.inf],np.nan).astype("float32")
    return result


def enhanced_spec():
    return dict(version=1, feature_columns=feature_columns(),calendar_version=CALENDAR_VERSION,
                calendar_as_of="feature_origin",monetary="exp_before_sum_null_discount_zero",
                regional_calendar=False,unknown_transfers="statutory_only_with_known_flag")
