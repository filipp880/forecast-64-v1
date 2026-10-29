import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import research_count_distribution as count
import research_count_decoder_followup as followup


def test_followup_grid_is_separate_and_never_extends_into_may():
    assert followup.MULTIPLIERS==[3.,4.,8.,16.]
    assert count.MULTIPLIERS==[0.,1.,2.]
    assert followup.ORIGINS==['2026-01-18','2026-03-22']


def test_larger_mae_penalty_can_change_zero_decision_without_new_model():
    probabilities=np.array([[.48,.3,.22,0,0,0,0,0]])
    reps=np.array([0.,1.,2.,3.,4.,6.,12.,24.])
    assert count.decode(probabilities,reps,2.,0)[0]==0
    assert count.decode(probabilities,reps,2.,16)[0]==1
