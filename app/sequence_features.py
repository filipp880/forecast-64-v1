"""The same 13 complete, availability-masked weekly lags used by OOF caches."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import KEY
from .event_features import FEATURE_COLUMNS, make_event_features


def sequence_spec():
    return {'version': 1, 'lookback_complete_weeks': 13,
            'unknown_history': 'nan_with_support', 'feature_columns': FEATURE_COLUMNS,
            'count_decoder': 'expected_smape_absolute_error_v1'}


def add_sequence_features(panel, frame, cutoff):
    from .ideas_features import weekly_history
    pairs = pd.MultiIndex.from_frame(frame[KEY].drop_duplicates())
    _, values, _, _ = weekly_history(panel, pairs, cutoff, periods=13)
    extra = pd.DataFrame(index=pairs)
    for lag in range(1, 14):
        extra[f'weekly_lag_{lag}'] = values[:, -lag]
    extra['weekly_support_13'] = np.isfinite(values).sum(axis=1)
    result = frame.merge(extra.reset_index(), on=KEY, how='left', validate='many_to_one')
    derived = make_event_features(result)
    additional = [c for c in derived if c not in result]
    return pd.concat([result, derived[additional]], axis=1)


def combined_feature_columns():
    from .enhancements import feature_columns
    return list(dict.fromkeys([*feature_columns(), *FEATURE_COLUMNS]))
