"""Границы временных экспериментов определяются датами таргетов."""
from __future__ import annotations
import pandas as pd

def research_split(test_cutoffs, horizon: int, step: int, n_train: int):
    if not test_cutoffs or min(horizon, step, n_train) <= 0:
        raise ValueError('Некорректные параметры временного разделения')
    first_test = min((pd.Timestamp(c) for c in test_cutoffs))
    valid = first_test - pd.Timedelta(days=horizon)
    latest_train = valid - pd.Timedelta(days=horizon)
    train = [latest_train - pd.Timedelta(days=step * k) for k in range(n_train)]
    assert max(train) + pd.Timedelta(days=horizon) <= valid
    assert valid + pd.Timedelta(days=horizon) <= first_test
    return (train, valid)
