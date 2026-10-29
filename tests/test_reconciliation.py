import numpy as np

from scripts.research_reconciliation import soft_reconcile


def test_soft_reconciliation_uses_only_history_and_preserves_zero_support():
    base = np.array([8., 0., 2., 3.])
    reconciled, report = soft_reconcile(base, [2., 4., 1., 1.], ['a','a','b','b'], ['w','x','w','x'], .2)
    assert np.isfinite(reconciled).all()
    assert (reconciled >= 0).all()
    assert report['objective_after'] < report['objective_before']
    np.testing.assert_array_equal(reconciled>0, base>0)
    assert reconciled.sum() < base.sum()


def test_matching_history_margins_leave_bottom_forecast_unchanged():
    base = np.array([1., 0., 2., 3.])
    result, report = soft_reconcile(base, base, ['a','a','b','b'], ['w','x','w','x'], .2)
    np.testing.assert_allclose(result, base)
    assert report['objective_before'] == report['objective_after'] == 0
