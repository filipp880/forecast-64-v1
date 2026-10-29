import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import research_count_distribution as count


def test_quantity_bins_preserve_fractional_values_and_exact_zero():
    y=[0,.25,1,1.2,2,2.8,3,4,4.2,7,8,15,15.1]
    np.testing.assert_array_equal(count.quantity_class(y),[0,1,1,2,2,3,3,4,5,5,6,6,7])
    with pytest.raises(ValueError):count.quantity_class([-1])


def test_representatives_only_use_supplied_fit_train_and_missing_fallback():
    reps=count.representatives([0,.25,.75,1.2,1.8,40.,60.])
    assert reps[1]==.5 and reps[2]==1.5 and reps[7]==50
    assert reps[5]==6


def test_bayes_action_zero_positive_and_fractional_representative():
    reps=np.array([0,.6,1.6,2.7,3.5,5.5,10.,20.])
    q=count.decode(np.eye(8)[[0,1,2]],reps,2.,0)
    np.testing.assert_allclose(q,[0,.6,1.6])
    risks=count.risk_table(reps,[0,1],2.,0)
    assert risks[0,0]==0 and risks[0,1]==pytest.approx(.7/60*200)


def test_score_matches_metric_definition():
    assert count.scalar_score([0,1],[0,1])==0
    assert count.scalar_score([0,1],[1,0])==pytest.approx(.7*200/60+.3*200/50)


def test_aggressive_gate_does_not_hide_auxiliary_loss():
    baseline={o:{n:1. for n in ['smape','wape','rmse','bias','rmsle']} for o in count.ORIGINS}
    candidate={o:{'smape':.99,'wape':.98,'rmse':1.5,'bias':1.5,'rmsle':1.5} for o in count.ORIGINS}
    report=count.assess(candidate,baseline)
    assert report['aggressive_eligible'] and not report['balanced_eligible']
    assert report['mean_ratios']['rmse']==1.5


def test_no_may_origin_or_automatic_promotion():
    assert count.ORIGINS==['2026-01-18','2026-03-22']
    assert len(count.SCOPES)*len(count.MULTIPLIERS)==6
