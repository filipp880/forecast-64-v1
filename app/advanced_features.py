"""Past-only serving transforms, identical to the verified all40 experiment.

History comes exclusively from the current request; no local data files or
experiment modules are used during inference. Missing history stays unknown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import KEY

ADVANCED_COLUMNS = ['warehouse_name', 'article', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3', 'characteristic', 'mean_7', 'mean_14', 'mean_28', 'mean_56', 'mean_90', 'zero_share_28', 'zero_share_90', 'last_value', 'days_since_last_sale', 'trend_28_90', 'n_sale_days_90', 'stock_last', 'stock_mean_28', 'stock_zero_share_90', 'stock_negative_share_90', 'stock_reserved', 'days_of_cover', 'last_week', 'mean_week_4', 'mean_week_8', 'mean_week_13', 'zero_week_share_13', 'mean_all', 'history_days', 'no_history', 'h', 'h_week', 'month', 'week_of_year', 'year', 'start_gap_days', 'ewma_week_4', 'ewma_week_13', 'std_week_13', 'positive_cv2_13', 'adi_13', 'week_trend_4_13', 'tsb_week', 'sba_week', 'article_week_mean', 'warehouse_week_mean', 'category_week_mean', 'category_share', 'weekly_lag_1', 'weekly_lag_2', 'weekly_lag_3', 'weekly_lag_4', 'weekly_lag_5', 'weekly_lag_6', 'weekly_lag_7', 'weekly_lag_8', 'weekly_lag_9', 'weekly_lag_10', 'weekly_lag_11', 'weekly_lag_12', 'weekly_lag_13', 'weekly_support_13', 'event_lag_1', 'event_lag_2', 'event_lag_3', 'event_lag_4', 'event_lag_5', 'event_lag_6', 'event_lag_7', 'event_lag_8', 'event_lag_9', 'event_lag_10', 'event_lag_11', 'event_lag_12', 'event_lag_13', 'event_known_1', 'event_known_2', 'event_known_3', 'event_known_4', 'event_known_5', 'event_known_6', 'event_known_7', 'event_known_8', 'event_known_9', 'event_known_10', 'event_known_11', 'event_known_12', 'event_known_13', 'event_rate_13', 'event_last_positive_age', 'event_age_at_horizon', 'event_recent_zero_streak', 'event_recent_positive_streak', 'event_interval_mean', 'event_interval_std', 'event_support_fraction', 'event_rate_x_h', 'event_lag_1_x_h', 'event_lag_2_x_h', 'event_lag_3_x_h', 'event_lag_4_x_h', 'ru_workdays', 'ru_holidays', 'ru_working_saturdays', 'ru_transferred', 'ru_transfer_known', 'ru_long_break', 'ru_days_after_holiday', 'ru_days_before_holiday', 'year_sin', 'year_cos', 'month_cross', 'sales_per_workday_90', 'sales_per_offday_90', 'log_revenue_28', 'log_discount_28', 'log_unit_revenue_28', 'discount_fraction_28', 'revenue_trend', 'money_known', 'avail_observed_days_28', 'avail_positive_days_28', 'avail_fraction_28', 'avail_sales_per_positive_day_28', 'avail_observed_days_90', 'avail_positive_days_90', 'avail_fraction_90', 'avail_sales_per_positive_day_90', 'availability_positive_streak', 'availability_nonpositive_streak', 'availability_restock_transitions_28', 'availability_known_last_day', 'long_lag_14', 'long_lag_15', 'long_lag_16', 'long_lag_17', 'long_lag_18', 'long_lag_19', 'long_lag_20', 'long_lag_21', 'long_lag_22', 'long_lag_23', 'long_lag_24', 'long_lag_25', 'long_lag_26', 'long_support_26', 'long_mean_26', 'long_lag_27', 'long_lag_28', 'long_lag_29', 'long_lag_30', 'long_lag_31', 'long_lag_32', 'long_lag_33', 'long_lag_34', 'long_lag_35', 'long_lag_36', 'long_lag_37', 'long_lag_38', 'long_lag_39', 'long_lag_40', 'long_lag_41', 'long_lag_42', 'long_lag_43', 'long_lag_44', 'long_lag_45', 'long_lag_46', 'long_lag_47', 'long_lag_48', 'long_lag_49', 'long_lag_50', 'long_lag_51', 'long_lag_52', 'long_support_52', 'long_mean_52', 'change_ratio_2_8', 'change_absolute_2_8', 'regime_change_ratio', 'regime_absolute_change', 'regime_recent_support', 'regime_prior_support', 'regime_id', 'life_age_days', 'life_positive_weeks_13', 'life_positive_weeks_52', 'life_recent_share', 'year_analog_26', 'year_analog_26_known', 'year_analog_52', 'year_analog_52_known', 'season_month_sin', 'season_month_cos', 'season_category_month_mean', 'season_category_month_support', 'neighbors_article_mean4', 'neighbors_other_warehouses_mean4', 'neighbors_other_warehouses_support', 'neighbors_similar_warehouse_mean4', 'neighbors_similar_warehouse_support', 'neighbors_relative_demand', 'stock_event_has_stock_x_age', 'stock_event_no_stock_x_streak', 'stock_event_zero_fraction_x_rate', 'stock_event_zero_fraction_x_h', 'stock_event_last_to_mean', 'stock_event_cover_x_rate', 'stock_event_negative_x_gap', 'stock_event_known']


class AdvancedHistory:
    def __init__(self,sales,stock,cutoff):
        cutoff=pd.Timestamp(cutoff)
        s=sales.loc[pd.to_datetime(sales.date)<=cutoff,['date',*KEY,'quantity']].copy()
        s['date']=pd.to_datetime(s.date);s['quantity']=s.quantity.astype('float32')
        for k in KEY:s[k]=s[k].astype(str).astype('category')
        self.first=s.groupby(KEY,observed=True).date.min()
        self.first.index=self.first.index.set_levels([self.first.index.levels[i].astype(str) for i in range(2)])
        self.start=s.date.min();s['week']=s.date-pd.to_timedelta(s.date.dt.weekday,unit='D')
        self.weekly=s.groupby([*KEY,'week'],observed=True).quantity.sum().unstack('week')
        self.daily=s.groupby([*KEY,'date'],observed=True).quantity.sum().unstack('date')
        article='char_article' if 'char_article' in stock.columns else 'article'
        st=stock.loc[pd.to_datetime(stock.date)<=cutoff,['date','warehouse_name',article,'stock']].copy()
        st=st.rename(columns={article:'article'});st['date']=pd.to_datetime(st.date);st['stock']=st.stock.astype('float32')
        for k in KEY:st[k]=st[k].astype(str).astype('category')
        self.stock=st.groupby([*KEY,'date'],observed=True).stock.sum().unstack('date')
        for table in [self.weekly,self.daily,self.stock]:
            table.index=table.index.set_levels([table.index.levels[i].astype(str) for i in range(2)])

    def add(self,frame,cutoff):
        cutoff=pd.Timestamp(cutoff);base=frame.drop_duplicates(KEY).copy()
        for k in KEY:base[k]=base[k].astype(str)
        pairs=pd.MultiIndex.from_frame(base[KEY]);first=self.first.reindex(pairs).to_numpy(dtype='datetime64[ns]')
        end=cutoff-pd.Timedelta(days=(cutoff.weekday()+1)%7)
        weeks=pd.date_range(end-pd.Timedelta(weeks=52)+pd.Timedelta(days=1),end-pd.Timedelta(days=6),freq='7D')
        y=self.weekly.reindex(index=pairs,columns=weeks).fillna(0).to_numpy(dtype='float32')
        known=(weeks.to_numpy()[None,:]>=first[:,None])&(weeks.to_numpy()[None,:]>=self.start.to_datetime64())
        y[~known]=np.nan
        extra=pd.DataFrame(index=pairs)
        for lag in range(14,53):extra[f'long_lag_{lag}']=y[:,-lag]
        for n in [26,52]:
            part=y[:,-n:];support=np.isfinite(part).sum(axis=1)
            extra[f'long_support_{n}']=support
            extra[f'long_mean_{n}']=np.divide(np.nansum(part,axis=1),support,out=np.full(len(y),np.nan),where=support>0)
        recent=np.nan_to_num(y[:,-2:],nan=0).sum(axis=1)/np.maximum(np.isfinite(y[:,-2:]).sum(axis=1),1)
        previous=np.nan_to_num(y[:,-10:-2],nan=0).sum(axis=1)/np.maximum(np.isfinite(y[:,-10:-2]).sum(axis=1),1)
        extra['change_ratio_2_8']=(recent+.1)/(previous+.1)
        extra['change_absolute_2_8']=recent-previous
        extra['life_age_days']=(cutoff.to_datetime64()-first)/np.timedelta64(1,'D')
        extra['life_positive_weeks_13']=(y[:,-13:]>0).sum(axis=1)
        extra['life_positive_weeks_52']=(y>0).sum(axis=1)
        extra['life_recent_share']=(np.nansum(y[:,-4:],axis=1)+.1)/(np.nansum(y,axis=1)+.1)
        dates=pd.date_range(cutoff-pd.Timedelta(days=89),cutoff)
        stock=self.stock.reindex(index=pairs,columns=dates).to_numpy(dtype='float32')
        quantity=self.daily.reindex(index=pairs,columns=dates).fillna(0).to_numpy(dtype='float32')
        for n in [28,90]:
            st=stock[:,-n:];q=quantity[:,-n:];obs=np.isfinite(st);available=obs&(st>0)
            extra[f'avail_observed_days_{n}']=obs.sum(axis=1)
            extra[f'avail_positive_days_{n}']=available.sum(axis=1)
            extra[f'avail_fraction_{n}']=np.divide(available.sum(axis=1),obs.sum(axis=1),out=np.full(len(y),np.nan),where=obs.sum(axis=1)>0)
            extra[f'avail_sales_per_positive_day_{n}']=np.divide((q*available).sum(axis=1),available.sum(axis=1),
                out=np.full(len(y),np.nan),where=available.sum(axis=1)>0)
        positive=np.isfinite(stock)&(stock>0);zero=np.isfinite(stock)&(stock<=0)
        extra['availability_positive_streak']=np.cumprod(positive[:,::-1],axis=1).sum(axis=1)
        extra['availability_nonpositive_streak']=np.cumprod(zero[:,::-1],axis=1).sum(axis=1)
        transitions=positive[:,1:]&zero[:,:-1]
        extra['availability_restock_transitions_28']=transitions[:,-28:].sum(axis=1)
        extra['availability_known_last_day']=np.isfinite(stock[:,-1]).astype(float)
        # Deterministic, label-free neighbor profiles; no KMeans fit is hidden here.
        profile=base[KEY+['category_lvl_2','mean_week_4','mean_week_13','zero_week_share_13']].copy()
        article=profile.groupby('article',observed=True).agg(mean4=('mean_week_4','mean'),mean13=('mean_week_13','mean'),
            zeros=('zero_week_share_13','mean'),category=('category_lvl_2','first'))
        a=article[['mean4','mean13','zeros']].to_numpy(float);a[:,:2]=np.log1p(np.maximum(a[:,:2],0))
        a=np.nan_to_num(a);a=(a-np.median(a,axis=0))/np.maximum(np.std(a,axis=0),.1)
        distance=((a[:,None,:]-a[None,:,:])**2).sum(axis=2);np.fill_diagonal(distance,np.inf)
        categories=article.category.astype(str).to_numpy();neighbor=[]
        for i in range(len(article)):
            eligible=np.flatnonzero((categories==categories[i])&(np.arange(len(article))!=i))
            if len(eligible)<3:eligible=np.flatnonzero(np.arange(len(article))!=i)
            chosen=eligible[np.argsort(distance[i,eligible],kind='stable')[:5]]
            neighbor.append(float(article.mean4.iloc[chosen].mean()) if len(chosen) else np.nan)
        lookup=pd.Series(neighbor,index=article.index)
        extra['neighbors_article_mean4']=pairs.get_level_values('article').map(lookup).to_numpy(float)
        own=base.set_index(KEY).mean_week_4.reindex(pairs)
        total=profile.groupby('article',observed=True).mean_week_4.sum();count=profile.groupby('article',observed=True).size()
        extra['neighbors_other_warehouses_mean4']=(pairs.get_level_values('article').map(total).to_numpy(float)-own.to_numpy(float))/np.maximum(pairs.get_level_values('article').map(count).to_numpy(float)-1,1)
        extra['neighbors_other_warehouses_support']=pairs.get_level_values('article').map(count).to_numpy(float)-1
        wh_profile=profile.pivot_table(index='warehouse_name',columns='category_lvl_2',values='mean_week_4',
            aggfunc='sum',observed=True).fillna(0)
        wh_profile=wh_profile.div(wh_profile.sum(axis=1).replace(0,1),axis=0)
        distance_wh=((wh_profile.to_numpy()[:,None,:]-wh_profile.to_numpy()[None,:,:])**2).sum(axis=2)
        np.fill_diagonal(distance_wh,np.inf)
        demand=profile.pivot_table(index='article',columns='warehouse_name',values='mean_week_4',aggfunc='mean',observed=True)
        estimates={};supports={}
        for i,warehouse in enumerate(wh_profile.index):
            nearest=np.argsort(distance_wh[i],kind='stable')[:min(3,max(len(wh_profile)-1,0))]
            columns=wh_profile.index[nearest];a=demand.reindex(columns=columns).to_numpy(float)
            weights=1/(distance_wh[i,nearest]+.1);valid=np.isfinite(a)
            numerator=(np.nan_to_num(a)*weights).sum(axis=1);denominator=(valid*weights).sum(axis=1)
            estimates[str(warehouse)]=pd.Series(np.divide(numerator,denominator,out=np.full(len(a),np.nan),where=denominator>0),index=demand.index)
            supports[str(warehouse)]=pd.Series(valid.sum(axis=1),index=demand.index)
        extra['neighbors_similar_warehouse_mean4']=[estimates.get(str(wh),pd.Series(dtype=float)).get(str(article),np.nan) for wh,article in pairs]
        extra['neighbors_similar_warehouse_support']=[supports.get(str(wh),pd.Series(dtype=float)).get(str(article),0) for wh,article in pairs]
        extra['neighbors_relative_demand']=(own.to_numpy(float)+.1)/(extra.neighbors_similar_warehouse_mean4.to_numpy(float)+.1)
        augmented=frame.copy()
        for k in KEY:augmented[k]=augmented[k].astype(str)
        augmented=augmented.merge(extra.reset_index(),on=KEY,how='left',validate='many_to_one',sort=False)
        for lag in [26,52]:
            values=np.full(len(augmented),np.nan)
            for week in augmented.week.unique():
                old=pd.Timestamp(week)-pd.Timedelta(weeks=lag)
                if old+pd.Timedelta(days=6)>cutoff:continue
                selected=augmented.week.eq(week);index=pd.MultiIndex.from_frame(augmented.loc[selected,KEY])
                raw=self.weekly.reindex(index=index,columns=[old]).fillna(0).iloc[:,0].to_numpy(float)
                f=self.first.reindex(index).to_numpy(dtype='datetime64[ns]')
                raw[(old.to_datetime64()<f)|(old<self.start)]=np.nan;values[selected]=raw
            augmented[f'year_analog_{lag}']=values
            augmented[f'year_analog_{lag}_known']=np.isfinite(values).astype(float)
        # Category/calendar interactions use historical observations only.
        category_code,category_labels=pd.factorize(base.category_lvl_2.astype(str),sort=True)
        category_by_pair=pd.Series(category_code,index=pairs)
        row_categories=pd.MultiIndex.from_frame(augmented[KEY]).map(category_by_pair).to_numpy(int)
        month=pd.to_datetime(augmented.week).dt.month.to_numpy(int)
        augmented['season_month_sin']=np.sin(2*np.pi*month/12)
        augmented['season_month_cos']=np.cos(2*np.pi*month/12)
        seasonal_mean=np.full(len(augmented),np.nan);seasonal_support=np.zeros(len(augmented))
        for m in np.unique(month):
            historical=y[:,weeks.month==m]
            sums=np.bincount(category_code,weights=np.nansum(historical,axis=1),minlength=len(category_labels))
            support=np.bincount(category_code,weights=np.isfinite(historical).sum(axis=1),minlength=len(category_labels))
            means=np.divide(sums,support,out=np.full(len(sums),np.nan),where=support>0)
            rows=month==m;seasonal_mean[rows]=means[row_categories[rows]];seasonal_support[rows]=support[row_categories[rows]]
        augmented['season_category_month_mean']=seasonal_mean
        augmented['season_category_month_support']=seasonal_support
        for c in list(extra.columns)+[f'year_analog_{n}{s}' for n in [26,52] for s in ['', '_known']]:
            augmented[c]=augmented[c].replace([np.inf,-np.inf],np.nan).astype('float32')
        return augmented


def extra_features(frame):
    """Uses only named past inputs, never target, actual or future observations."""
    lag=frame[[f'weekly_lag_{i}' for i in range(1,11)]].to_numpy(float)
    known=np.isfinite(lag)
    recent=np.divide(np.nansum(lag[:,:2],axis=1),known[:,:2].sum(axis=1),
        out=np.full(len(frame),np.nan),where=known[:,:2].sum(axis=1)>0)
    past=np.divide(np.nansum(lag[:,2:],axis=1),known[:,2:].sum(axis=1),
        out=np.full(len(frame),np.nan),where=known[:,2:].sum(axis=1)>=4)
    change=(recent+.1)/(past+.1)
    result=pd.DataFrame(index=frame.index)
    result['regime_change_ratio']=change
    result['regime_absolute_change']=recent-past
    result['regime_recent_support']=known[:,:2].sum(axis=1)
    result['regime_prior_support']=known[:,2:].sum(axis=1)
    result['regime_id']=np.where(~np.isfinite(change),-1,np.where(change<.7,0,np.where(change>1.3,2,1)))
    stock=frame.stock_last.to_numpy(float);missing=~np.isfinite(stock)
    positive=np.where(missing,np.nan,(stock>0).astype(float))
    zero=frame.stock_zero_share_90.to_numpy(float)
    age=frame.event_last_positive_age.to_numpy(float)
    streak=frame.event_recent_zero_streak.to_numpy(float)
    rate=frame.event_rate_13.to_numpy(float)
    result['stock_event_has_stock_x_age']=positive*age
    result['stock_event_no_stock_x_streak']=(1-positive)*streak
    result['stock_event_zero_fraction_x_rate']=zero*rate
    result['stock_event_zero_fraction_x_h']=zero*frame.h.to_numpy(float)
    result['stock_event_last_to_mean']=stock/np.maximum(np.abs(frame.stock_mean_28.to_numpy(float)),1.)
    result['stock_event_cover_x_rate']=frame.days_of_cover.to_numpy(float)*rate
    result['stock_event_negative_x_gap']=np.where(missing,np.nan,(stock<0).astype(float))*age
    result['stock_event_known']=np.isfinite(stock).astype(float)
    return result.replace([np.inf,-np.inf],np.nan).astype('float32')

def advanced_spec():
    return dict(version=1,history_complete_weeks=52,unknown_history='nan_with_support',
                source='request_only_as_of_cutoff',feature_columns=ADVANCED_COLUMNS)


def augment_advanced_features(sales,stock,frame,cutoff):
    result=AdvancedHistory(sales,stock,cutoff).add(frame,cutoff)
    extra=extra_features(result)
    for column in extra:
        if column not in result:result[column]=extra[column].to_numpy()
    for column in result.select_dtypes(include=[np.number]):result[column]=result[column].astype('float32')
    missing=set(ADVANCED_COLUMNS)-set(result)
    if missing:raise ValueError('Missing advanced feature columns: '+repr(sorted(missing)))
    return result
