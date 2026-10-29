"""Tests for temporal event features and the cached purged experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_events as events
from app.event_features import INPUT_COLUMNS, LAG_COLUMNS, make_event_features
from app.features import CAT_FEATURES


def feature_frame(rows=3):
    f = pd.DataFrame({c: np.zeros(rows, dtype='float32') for c in INPUT_COLUMNS if c not in CAT_FEATURES})
    for c in CAT_FEATURES:
        f[c] = 'a'
    f['h'] = [1, 2, 3][:rows]
    f[LAG_COLUMNS] = np.nan
    f.loc[0, ['weekly_lag_1', 'weekly_lag_2', 'weekly_lag_4']] = [0., 3., 2.]
    f.loc[1, ['weekly_lag_1', 'weekly_lag_2', 'weekly_lag_3']] = [2., 1., 0.]
    return f


def test_unknown_history_is_distinct_from_observed_zero_and_ignores_labels():
    frame = feature_frame()
    x = make_event_features(frame.assign(actual=[99, 99, 99], target=[0, 0, 0]))
    assert x.loc[0, 'event_lag_1'] == 0
    assert x.loc[0, 'event_known_1'] == 1
    assert np.isnan(x.loc[0, 'event_lag_3'])
    assert x.loc[0, 'event_known_3'] == 0
    assert x.loc[0, 'event_recent_zero_streak'] == 1
    assert x.loc[1, 'event_recent_positive_streak'] == 2
    assert np.isnan(x.loc[2, 'event_rate_13'])
    assert x.loc[2, 'event_support_fraction'] == 0
    pd.testing.assert_frame_equal(x, make_event_features(frame))
    assert not {'target', 'actual', 'feature_origin'} & set(x.columns)


def test_past_intervals_and_horizon_age():
    x = make_event_features(feature_frame())
    assert x.loc[0, 'event_last_positive_age'] == 2
    assert x.loc[0, 'event_interval_mean'] == 2
    assert x.loc[0, 'event_interval_std'] == 0
    assert x.loc[1, 'event_age_at_horizon'] == 2


def test_purged_split_does_not_allow_late_target_labels():
    origin = '2026-03-22'
    manifest = {f'{date}-True.parquet': {} for date in
                ['2025-05-04', '2025-06-01', '2025-06-29', '2025-07-27',
                 '2025-08-24', '2025-09-21', '2025-10-19', '2025-11-16',
                 '2025-12-14', '2026-01-18', '2026-02-08']}
    schedule = events.split_names(origin, manifest)
    assert schedule['valid'] == '2026-01-18-True.parquet'
    assert all(events.forecast_end(name[:10], 9) <= pd.Timestamp(schedule['inner_origin'])
               for name in schedule['inner'])
    assert schedule['valid'] not in schedule['refit']
    assert '2026-02-08-True.parquet' in schedule['refit']


def test_load_frames_rejects_labels_past_outer_origin(monkeypatch):
    frame = feature_frame(1).assign(week=pd.Timestamp('2026-03-23'),
                                    feature_origin=pd.Timestamp('2026-03-22'), target=1.)
    monkeypatch.setattr(events, 'cached', lambda *args, **kwargs: frame)
    with pytest.raises(ValueError, match='boundary'):
        events.load_frames('2026-03-22', {}, ['2026-03-22-True.parquet'], '2026-03-22')


def test_encoding_unknown_category_does_not_fit_on_validation():
    train = feature_frame(1)
    _, categories = events.encode(train)
    valid = train.copy()
    valid['article'] = 'unseen'
    x, categories_after = events.encode(valid, categories)
    assert pd.isna(x.article.iloc[0])
    assert categories == categories_after


def test_evaluate_replaces_only_probability(monkeypatch, tmp_path):
    import pickle
    origin = '2026-03-22'
    old = pd.DataFrame({'week': pd.to_datetime(['2026-03-23', '2026-03-23']),
                        'warehouse_name': ['w', 'w'], 'article': ['a', 'b'],
                        'actual': [2., 4.], 'h': [1, 1], 'mean_28': [1/7, 2/7],
                        'n_sale_days_90': [10., 10.], 'zero_week_share_13': [.2, .2]})
    folder = tmp_path / origin
    folder.mkdir()
    old.to_parquet(folder / 'evaluation.parquet', index=False)
    values = {'probability': np.ones(2), 'magnitude': np.array([2., 4.]), 'lgbm_raw': np.array([2., 4.])}
    (folder / 'dynamics-values.pkl').write_bytes(pickle.dumps(values))
    monkeypatch.setattr(events, 'CHAMPION_ROOT', tmp_path)
    recipe = {'source': 'dynamics', 'post': {'round_int': True}}
    unchanged = events.evaluate(origin, old, np.ones(2), recipe)
    assert unchanged['candidate'] == unchanged['champion']
    changed = events.evaluate(origin, old, np.zeros(2), recipe)
    assert changed['candidate']['smape'] > changed['champion']['smape']


def test_development_origins_exclude_confirmation_and_joint_selection_is_common():
    assert '2026-05-24' not in events.DEVELOPMENT_ORIGINS
    assert events.JOINT_SELECTION_ORIGINS == ['2026-01-18', '2026-03-22']
    assert len(events.CANDIDATES) == 2
    assert events.COMMON_PARAMS['n_jobs'] <= 6
