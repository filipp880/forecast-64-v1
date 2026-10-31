"""Request-only features for the frozen daily expert, summed into calendar weeks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .daily_bridge import DailyHistory, DAILY_COLUMNS
from .features import CAT_FEATURES, KEY

EXTRA = ['stock_negative_share_90', 'stock_reserved', 'last_week', 'mean_week_4', 'mean_week_8',
         'mean_week_13', 'zero_week_share_13', 'mean_all', 'history_days', 'ewma_week_4',
         'ewma_week_13', 'std_week_13', 'positive_cv2_13', 'adi_13', 'week_trend_4_13',
         'tsb_week', 'sba_week', 'article_week_mean', 'warehouse_week_mean', 'category_week_mean',
         'category_share']
JOINT_DAILY_COLUMNS = list(dict.fromkeys(DAILY_COLUMNS + EXTRA +
    [f'bridge_lag_{i}' for i in range(1, 14)] +
    ['bridge_support_13', 'bridge_event_rate_13', 'bridge_week_horizon']))


def joint_spec():
    return dict(version=1, source='request_only_as_of_cutoff', daily_feature_columns=JOINT_DAILY_COLUMNS,
                daily_offset='log(max(0.5 * mean_28 + 0.5 * dow_mean, 0.02))',
                daily_output='exp(clip(raw_score + offset, -15, 15))',
                aggregation='sum_seven_complete_calendar_days_then_weekly_event_gate_then_rint',
                unknown_history='nan_with_support')


def make_joint_daily(sales, weekly, cutoff):
    """Reproduce round2 enrichment without importing a training script or cache."""
    history = DailyHistory(sales, cutoff)
    daily = history.expand(weekly, cutoff)
    w = weekly.copy()
    for frame in (daily, w):
        frame['week'] = pd.to_datetime(frame.week)
        for key in KEY:
            frame[key] = frame[key].astype(str)
    wk = ['week', *KEY]
    index = pd.MultiIndex.from_frame(w[wk])
    codes = index.get_indexer(pd.MultiIndex.from_frame(daily[wk]))
    if not index.is_unique or (codes < 0).any():
        raise ValueError('Daily/weekly feature alignment differs')
    for column in EXTRA:
        daily[column] = w[column].to_numpy(dtype='float32')[codes]
    pairs = pd.MultiIndex.from_frame(w[KEY].drop_duplicates())
    pair_codes = pairs.get_indexer(pd.MultiIndex.from_frame(daily[KEY]))
    cut = pd.Timestamp(cutoff)
    end = cut - pd.Timedelta(days=(cut.weekday() + 1) % 7)
    dates = pd.date_range(end - pd.Timedelta(days=90), end)
    values = history.daily.reindex(index=pairs, columns=dates).fillna(0).to_numpy(float)
    values = values.reshape(len(pairs), 13, 7).sum(axis=2)
    starts = pd.date_range(dates[0], periods=13, freq='7D')
    first = history.first.reindex(pairs).to_numpy(dtype='datetime64[ns]')
    known = ((starts.to_numpy()[None, :] >= first[:, None]) &
             (starts.to_numpy()[None, :] >= history.start.to_datetime64()))
    values[~known] = np.nan
    for i in range(1, 14):
        daily[f'bridge_lag_{i}'] = values[pair_codes, -i].astype('float32')
    support = known.sum(axis=1)
    rates = np.divide((values > 0).sum(axis=1), support,
                      out=np.full(len(pairs), np.nan), where=support > 0)
    daily['bridge_support_13'] = support[pair_codes].astype('float32')
    daily['bridge_event_rate_13'] = rates[pair_codes].astype('float32')
    next_monday = cut - pd.Timedelta(days=cut.weekday()) + pd.Timedelta(days=7)
    daily['bridge_week_horizon'] = ((daily.week - next_monday).dt.days / 7 + 1).astype('float32')
    daily = daily[['date', 'week', 'feature_origin', *JOINT_DAILY_COLUMNS]].copy()
    for column in CAT_FEATURES:
        daily[column] = daily[column].astype('category')
    return daily
