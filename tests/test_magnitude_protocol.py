"""Chronological and fixed-mask regressions for regular magnitude calibration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_magnitude as magnitude


def frame():
    return pd.DataFrame({'actual': [2., 4., 8., 0.], 'poisson_magnitude': [1., 2., 4., 3.],
                         'champion_prediction': [1., 0., 3., 2.], 'mean_28': [.1, 1., 2., .5],
                         'n_sale_days_90': [45., 50., 20., 60.],
                         'label_end': pd.Timestamp('2026-01-01')})


def test_calibration_rejects_unfinished_entire_oof_horizon():
    f = frame()
    with pytest.raises(ValueError, match='not ended'):
        magnitude.calibration_rows([f], '2025-12-31')
    rows = magnitude.calibration_rows([f], '2026-01-01')
    assert len(rows) == 1
    assert rows.actual.iloc[0] == 2


def test_magnitude_calibration_preserves_mask_and_nonregular_values():
    f = frame()
    calibration = magnitude.fit_calibration(magnitude.calibration_rows([f], '2026-01-01'))
    p = magnitude.apply_calibration(f, calibration, 'median_ratio')
    np.testing.assert_array_equal(p > 0, f.champion_prediction.to_numpy() > 0)
    assert p[1] == 0
    assert p[2] == f.champion_prediction.iloc[2]
    assert p[0] == 2
    # A future actual=0 is not consulted by apply; the same positive policy applies.
    assert p[3] == 6
    shuffled_truth = f.copy()
    shuffled_truth['actual'] = [999., 99., 0., 1.]
    np.testing.assert_array_equal(p, magnitude.apply_calibration(shuffled_truth, calibration, 'median_ratio'))


def test_fixed_grid_and_small_bin_fallback():
    f = frame().iloc[[0]].copy()
    f['actual'] = 1.2
    calibration = magnitude.fit_calibration(f)
    for record in calibration['bins'].values():
        assert record['weight'] == 0
        assert record['smape_ratio'] == 1.2
        assert record['median_ratio'] == 1.2
    assert magnitude.choose_risk_ratio(np.ones(5)) == 1.


def test_strict_demand_bin_boundaries():
    f = pd.DataFrame({'mean_28': [0., 2/7, 10/7]})
    np.testing.assert_array_equal(magnitude.demand_bin(f), [0, 1, 2])


def test_declared_origins_never_include_may_confirmation():
    assert magnitude.EVALUATION == ['2026-01-18', '2026-03-22']
    assert magnitude.WARMUP == '2025-10-19'
    assert len(magnitude.CANDIDATES) == 2
