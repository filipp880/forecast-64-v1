"""Chronological boundaries are defined by complete target-week end dates."""
from __future__ import annotations

import pandas as pd


def last_complete_sunday(value):
    value = pd.Timestamp(value).normalize()
    return value - pd.Timedelta(days=(value.weekday() + 1) % 7)


def research_split(test_cutoffs, horizon=9, step=28, n_train=12):
    if not test_cutoffs or min(horizon, step, n_train) <= 0 or step % 7:
        raise ValueError("Positive weekly horizons and a step divisible by seven are required")
    first_test = min(pd.Timestamp(c) for c in test_cutoffs)
    valid = last_complete_sunday(first_test) - pd.Timedelta(weeks=horizon)
    latest_train = valid - pd.Timedelta(weeks=horizon)
    train = [latest_train - pd.Timedelta(days=step * k) for k in range(n_train)]
    assert max(train) + pd.Timedelta(weeks=horizon) <= valid
    assert valid + pd.Timedelta(weeks=horizon) <= first_test
    return train, valid
