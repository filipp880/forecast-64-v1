"""Past-only lag features for a pooled weekly event classifier.

Unknown history remains NaN; a known zero has event=0 and known=1.
No labels or future quantities are read by this transformation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .enhancements import feature_columns

LAG_COLUMNS = [f'weekly_lag_{i}' for i in range(1, 14)]
INPUT_COLUMNS = [*feature_columns('dynamics'), *LAG_COLUMNS, 'weekly_support_13']
EVENT_COLUMNS = [*[f'event_lag_{i}' for i in range(1, 14)],
                 *[f'event_known_{i}' for i in range(1, 14)],
                 'event_rate_13', 'event_last_positive_age', 'event_age_at_horizon',
                 'event_recent_zero_streak', 'event_recent_positive_streak',
                 'event_interval_mean', 'event_interval_std', 'event_support_fraction',
                 'event_rate_x_h', *[f'event_lag_{i}_x_h' for i in range(1, 5)]]
FEATURE_COLUMNS = [*INPUT_COLUMNS, *EVENT_COLUMNS]


def make_event_features(frame):
    """Produce only allowed numeric/categorical inputs, ignoring target columns."""
    x = frame[INPUT_COLUMNS].copy()
    lags = frame[LAG_COLUMNS].to_numpy(dtype='float32')
    known = np.isfinite(lags)
    positive = known & (lags > 0)
    support = known.sum(axis=1)
    h = frame.h.to_numpy(dtype='float32')
    for i in range(13):
        x[f'event_lag_{i+1}'] = np.where(known[:, i], positive[:, i], np.nan).astype('float32')
        x[f'event_known_{i+1}'] = known[:, i].astype('float32')
    rate = np.divide(positive.sum(axis=1), support, out=np.full(len(x), np.nan), where=support > 0)
    ages = np.where(positive, np.arange(1, 14)[None, :], np.inf).min(axis=1)
    ages[~np.isfinite(ages)] = np.nan
    x['event_rate_13'] = rate
    x['event_last_positive_age'] = ages
    x['event_age_at_horizon'] = ages + h - 1
    for label, mask in [('zero', known & ~positive), ('positive', positive)]:
        x[f'event_recent_{label}_streak'] = np.cumprod(mask, axis=1).sum(axis=1)
    # Adjacent past events determine intervals; no right-censoring is invented.
    positions = np.where(positive, np.arange(1, 14)[None, :], np.nan)
    count = positive.sum(axis=1)
    first = np.where(positive, np.arange(1, 14)[None, :], np.inf).min(axis=1)
    last = np.where(positive, np.arange(1, 14)[None, :], -np.inf).max(axis=1)
    mean = np.divide(last - first, count - 1, out=np.full(len(x), np.nan), where=count > 1)
    previous = np.full(len(x), np.nan)
    squared_sum = np.zeros(len(x))
    for column in positions.T:
        gap = column - previous
        squared_sum += np.where(np.isfinite(gap), gap ** 2, 0)
        previous = np.where(np.isfinite(column), column, previous)
    second = np.divide(squared_sum, count - 1, out=np.full(len(x), np.nan), where=count > 1)
    x['event_interval_mean'] = mean
    x['event_interval_std'] = np.sqrt(np.maximum(second - mean ** 2, 0))
    x['event_support_fraction'] = support / 13
    x['event_rate_x_h'] = rate * h
    for i in range(1, 5):
        x[f'event_lag_{i}_x_h'] = x[f'event_lag_{i}'].to_numpy(float) * h
    numeric = x.select_dtypes(include=[np.number]).columns
    x[numeric] = x[numeric].astype('float32')
    return x[FEATURE_COLUMNS]
