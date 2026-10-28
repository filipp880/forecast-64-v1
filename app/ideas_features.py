"""Features from ideas.txt. Every statistic is rebuilt from origin-past only.

Missing history is NaN with support/availability, never an invented zero sale.
This module is separate from preprocessing v4 so existing artifacts stay loadable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import KEY

EVENTS = {'new_year': (1, 1), 'feb23': (2, 23), 'mar8': (3, 8),
          'may1': (5, 1), 'may9': (5, 9), 'jun12': (6, 12), 'nov4': (11, 4)}
GROUPS = {
    'month': ['month_has_15', 'month_has_end', 'month_begin_days', 'month_end_days',
              'days_to_15', 'days_to_month_end'],
    'events': [f'{e}_{s}' for e in EVENTS for s in
               ('days_before', 'days_after', 'during', 'pre_week', 'post_week', 'category_response')],
    'warehouse': [f'warehouse_{s}' for s in ('volume_1', 'volume_4', 'volume_13', 'trend_4_13',
                   'active_1', 'active_4', 'active_13', 'active_trend')],
    'season': [f'season_{lag}_{s}' for lag in (26, 52) for s in ('quantity', 'known', 'article', 'warehouse')],
    'similarity': ['warehouse_cluster', *[f'warehouse_cluster_distance_{i}' for i in range(3)]],
    'shape': [*[f'weekly_lag_{i}' for i in range(1, 14)], 'weekly_peak_4', 'weekly_peak_13',
              'weekly_peak_age', 'weekly_support_13'],
    'log_history': [f'mean_log_week_{n}' for n in (4, 8, 13)],
    'group_zeros': [f'{group}_{s}_{n}' for n in (4, 8, 13) for group in ('article', 'warehouse')
                    for s in ('zero_share', 'zero_difference', 'support')],
    'analog': ['target_lag_13', 'target_lag_13_known'],
    'shares': [f'pair_{measure}_share_{group}' for measure in ('volume', 'events')
               for group in ('article', 'warehouse')],
    'coverage': ['article_active_warehouses_4', 'article_observed_warehouses',
                 'article_coverage_4', 'article_coverage_change',
                 *[f'{g}_{s}' for g in ('article', 'category') for s in ('positive_rate', 'support')]],
    'cnn_context': [f'article_weekly_lag_{i}' for i in range(1, 14)],
}


def ideas_columns(groups):
    return [column for group in groups for column in GROUPS[group]]


def event_calendar(weeks):
    """Known fixed federal event dates; no assumed uplift or regional mapping."""
    result = pd.DataFrame(index=pd.DatetimeIndex(weeks).unique().sort_values())
    for week in result.index:
        days = pd.date_range(week, periods=7)
        result.loc[week, 'month_has_15'] = int((days.day == 15).any())
        result.loc[week, 'month_has_end'] = int(days.is_month_end.any())
        result.loc[week, 'month_begin_days'] = int((days.day <= 5).sum())
        result.loc[week, 'month_end_days'] = int(((days.days_in_month-days.day) < 5).sum())
        result.loc[week, 'days_to_15'] = min(abs((d-pd.Timestamp(d.year, d.month, 15)).days) for d in days)
        result.loc[week, 'days_to_month_end'] = int((days.days_in_month-days.day).min())
        for name, (month, day) in EVENTS.items():
            dates = [pd.Timestamp(year, month, day) for year in range(week.year-1, week.year+3)]
            before = min((e-days[-1]).days for e in dates if e >= days[-1])
            after = min((days[0]-e).days for e in dates if e <= days[0])
            during = any(days[0] <= e <= days[-1] for e in dates)
            if during: before, after = 0, 0
            result.loc[week, f'{name}_days_before'] = min(before, 366)
            result.loc[week, f'{name}_days_after'] = min(after, 366)
            result.loc[week, f'{name}_during'] = int(during)
            result.loc[week, f'{name}_pre_week'] = int(any(week+pd.Timedelta(days=7) <= e <= week+pd.Timedelta(days=13) for e in dates))
            result.loc[week, f'{name}_post_week'] = int(any(week-pd.Timedelta(days=7) <= e < week for e in dates))
    return result.astype('float32')


def weekly_history(panel, pairs, cutoff, periods=61):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff-pd.Timedelta(days=(cutoff.weekday()+1)%7)
    weeks = pd.date_range(end-pd.Timedelta(weeks=periods)+pd.Timedelta(days=1), periods=periods, freq='7D')
    known = panel.loc[(panel.date<=end)&(panel.date>=weeks[0])&(panel._first_sale_date<=cutoff)]
    hist = known[['date', *KEY, 'quantity']].copy()
    hist['week'] = hist.date-pd.to_timedelta(hist.date.dt.weekday, unit='D')
    sums = hist.groupby([*KEY, 'week'], observed=True).quantity.sum().unstack('week')
    y = sums.reindex(index=pairs, columns=weeks).to_numpy(dtype='float32')
    first = known.groupby(KEY, observed=True)._first_sale_date.first().reindex(pairs)
    # Full calendar weeks only. First recorded date is an availability boundary,
    # not evidence of the true commercial launch of an article.
    input_start = pd.Timestamp(panel.date.min())
    starts = first.fillna(cutoff+pd.Timedelta(days=1)).to_numpy(dtype='datetime64[ns]')
    valid = (weeks.to_numpy()[None, :] >= starts[:, None]) & (weeks.to_numpy()[None, :] >= input_start.to_datetime64())
    raw = y.copy()
    raw_valid = ((weeks+pd.Timedelta(days=6)).to_numpy()[None, :] >= starts[:, None]) & (weeks.to_numpy()[None, :] >= input_start.to_datetime64())
    raw[~raw_valid] = np.nan
    y[~valid] = np.nan
    return weeks, y, valid, raw


def _mean(a, axis=1):
    n = np.isfinite(a).sum(axis=axis)
    return np.divide(np.nansum(a, axis=axis), n, out=np.full(n.shape, np.nan), where=n>0)


def augment_ideas(panel, frame, cutoff):
    """Accept an already v4-augmented frame; target/actual columns are ignored."""
    from sklearn.cluster import KMeans
    base = frame.drop_duplicates(KEY).set_index(KEY)
    pairs = base.index
    weeks, y, valid, raw = weekly_history(panel, pairs, cutoff)
    extra = pd.DataFrame(index=pairs)
    last = y[:, -13:]
    for i in range(1, 14): extra[f'weekly_lag_{i}'] = last[:, -i]
    for n in (4, 13):
        a = last[:, -n:]
        extra[f'weekly_peak_{n}'] = np.where(np.isfinite(a).any(axis=1), np.nan_to_num(a, nan=-1).max(axis=1), np.nan)
    extra['weekly_peak_age'] = np.nan_to_num(last[:, ::-1], nan=-1).argmax(axis=1)
    extra.loc[~np.isfinite(last).any(axis=1), 'weekly_peak_age'] = np.nan
    extra['weekly_support_13'] = np.isfinite(last).sum(axis=1)
    for n in (4, 8, 13): extra[f'mean_log_week_{n}'] = _mean(np.log1p(last[:, -n:]))
    quantities = pd.DataFrame(y, index=pairs, columns=weeks)
    raw_quantities = pd.DataFrame(raw, index=pairs, columns=weeks)
    article_history = raw_quantities.groupby(level='article').mean()
    for i in range(1, 14):
        extra[f'article_weekly_lag_{i}'] = pairs.get_level_values('article').map(article_history.iloc[:, -i]).to_numpy(float)
    volume = pd.Series(np.nansum(raw[:, -4:], axis=1), index=pairs)
    events = pd.Series((raw[:, -4:]>0).sum(axis=1), index=pairs)
    for group in KEY:
        short = 'warehouse' if group=='warehouse_name' else 'article'
        for label, measure in (('volume', volume), ('events', events)):
            total = measure.groupby(level=group).transform('sum')
            extra[f'pair_{label}_share_{short}'] = measure/total.replace(0, np.nan)
        totals = raw_quantities.groupby(level=group).sum(min_count=1)
        if short == 'warehouse':
            active = (raw_quantities>0).groupby(level=group).sum()
            for n in (1, 4, 13):
                extra[f'warehouse_volume_{n}'] = pairs.get_level_values(group).map(totals.iloc[:, -n:].mean(axis=1)).to_numpy(float)
                extra[f'warehouse_active_{n}'] = pairs.get_level_values(group).map(active.iloc[:, -n:].mean(axis=1)).to_numpy(float)
            extra['warehouse_trend_4_13'] = extra.warehouse_volume_4/extra.warehouse_volume_13.replace(0, np.nan)
            extra['warehouse_active_trend'] = extra.warehouse_active_4/extra.warehouse_active_13.replace(0, np.nan)
        for n in (4, 8, 13):
            a = last[:, -n:]; support = pd.Series(np.isfinite(a).sum(axis=1), index=pairs)
            zeros = pd.Series(((a==0)&np.isfinite(a)).sum(axis=1), index=pairs)
            denom = support.groupby(level=group).transform('sum')
            rate = zeros.groupby(level=group).transform('sum')/denom.replace(0, np.nan)
            extra[f'{short}_zero_share_{n}'] = rate
            extra[f'{short}_zero_difference_{n}'] = zeros/support.replace(0, np.nan)-rate
            extra[f'{short}_support_{n}'] = denom
    active4 = pd.Series(np.any(raw[:, -4:]>0, axis=1), index=pairs)
    previous4 = pd.Series(np.any(raw[:, -8:-4]>0, axis=1), index=pairs)
    observed = pd.Series(np.isfinite(raw).any(axis=1), index=pairs)
    extra['article_active_warehouses_4'] = active4.groupby(level='article').transform('sum')
    extra['article_observed_warehouses'] = observed.groupby(level='article').transform('sum')
    extra['article_coverage_4'] = extra.article_active_warehouses_4/extra.article_observed_warehouses.replace(0, np.nan)
    extra['article_coverage_change'] = (active4.astype(int)-previous4.astype(int)).groupby(level='article').transform('sum')/extra.article_observed_warehouses.replace(0, np.nan)
    for group in ('article', 'category'):
        labels = pairs.get_level_values('article') if group=='article' else base.category_lvl_1.astype(str)
        supports = pd.Series(np.isfinite(last).sum(axis=1), index=pairs).groupby(labels, observed=True).transform('sum')
        positives = pd.Series((last>0).sum(axis=1), index=pairs).groupby(labels, observed=True).transform('sum')
        extra[f'{group}_positive_rate'] = positives/supports.replace(0, np.nan)
        extra[f'{group}_support'] = supports
    # Warehouse similarity: past category shares only; no labels, no saved future fit.
    profile = pd.DataFrame({'warehouse': pairs.get_level_values('warehouse_name'),
                            'category': base.category_lvl_1.astype(str).to_numpy(), 'volume': volume.to_numpy()})
    profile = profile.pivot_table(index='warehouse', columns='category', values='volume', aggfunc='sum', observed=True).fillna(0)
    profile = profile.div(profile.sum(axis=1).replace(0, 1), axis=0)
    warehouse_stats = extra.reset_index().drop_duplicates('warehouse_name').set_index('warehouse_name')
    # Similarity includes category structure, volume dynamics AND sparsity.
    profile.columns = ['category_share_'+str(c) for c in profile.columns]
    profile['demand_trend'] = warehouse_stats.warehouse_trend_4_13.reindex(profile.index).fillna(1).clip(0,5)/5
    profile['active_trend'] = warehouse_stats.warehouse_active_trend.reindex(profile.index).fillna(1).clip(0,5)/5
    profile['zero_share'] = warehouse_stats.warehouse_zero_share_13.reindex(profile.index).fillna(1)
    km = KMeans(n_clusters=min(3, len(profile)), n_init=10, random_state=42).fit(profile)
    # Canonical order by dominant category and complete centroid coordinates.
    order = sorted(range(len(km.cluster_centers_)), key=lambda k:(int(km.cluster_centers_[k].argmax()), *km.cluster_centers_[k]))
    mapping = {k:i for i,k in enumerate(order)}
    labels = pd.Series([mapping[k] for k in km.labels_], index=profile.index)
    extra['warehouse_cluster'] = pairs.get_level_values('warehouse_name').map(labels).to_numpy(float)
    distances = km.transform(profile)[:, order]
    for i in range(3):
        extra[f'warehouse_cluster_distance_{i}'] = pairs.get_level_values('warehouse_name').map(pd.Series(distances[:, min(i, len(order)-1)], index=profile.index)).to_numpy(float)
    # Event response is a past category diagnostic, not a causal uplift estimate.
    past_cal = event_calendar(weeks)
    categories = base.category_lvl_1.astype(str)
    for name in EVENTS:
        flag = (past_cal[f'{name}_during']+past_cal[f'{name}_pre_week']+past_cal[f'{name}_post_week']).to_numpy()>0
        event_mean = pd.Series(_mean(y[:, flag]), index=pairs).groupby(categories, observed=True).transform('mean')
        other_mean = pd.Series(_mean(y[:, ~flag]), index=pairs).groupby(categories, observed=True).transform('mean')
        extra[f'{name}_category_response'] = event_mean/other_mean.replace(0, np.nan)
    result = frame.merge(extra.reset_index(), on=KEY, how='left', validate='many_to_one')
    cal = event_calendar(result.week)
    result = result.merge(cal.rename_axis('week').reset_index(), on='week', how='left', validate='many_to_one')
    for name in EVENTS:
        result[f'{name}_category_response'] *= (result[f'{name}_during']+result[f'{name}_pre_week']+result[f'{name}_post_week']).clip(upper=1)
    # Label-aligned seasonal analogs vary with target week, without using targets.
    for lag in (13, 26, 52):
        values = np.full(len(result), np.nan)
        group_values = {g: np.full(len(result), np.nan) for g in ('article', 'warehouse')}
        for week in result.week.unique():
            old = pd.Timestamp(week)-pd.Timedelta(weeks=lag)
            if old not in quantities or old+pd.Timedelta(days=6)>pd.Timestamp(cutoff): continue
            mask = result.week.eq(week)
            lookup = quantities[old]
            values[mask] = pd.MultiIndex.from_frame(result.loc[mask, KEY]).map(lookup).to_numpy(float)
            for label, key in (('article','article'), ('warehouse','warehouse_name')):
                mean = lookup.groupby(level=key).mean()
                group_values[label][mask] = result.loc[mask,key].map(mean).to_numpy(float)
        if lag==13:
            result['target_lag_13'] = values
            result['target_lag_13_known'] = np.isfinite(values).astype(float)
        else:
            result[f'season_{lag}_quantity'] = values
            result[f'season_{lag}_known'] = np.isfinite(values).astype(float)
            for g in group_values: result[f'season_{lag}_{g}'] = group_values[g]
    for col in ideas_columns(GROUPS): result[col] = result[col].replace([np.inf,-np.inf], np.nan).astype('float32')
    return result


def ideas_spec():
    return dict(version=1, groups=GROUPS, history='past_only_complete_weeks_with_support',
                seasonal_missing='NaN_and_known_flag', similarity='origin_past_category_share_dynamics_sparsity_kmeans3',
                calendar='fixed_federal_events_no_assumed_uplift', target_alignment='target_week_minus_lag_weeks')
