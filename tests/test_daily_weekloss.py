"""Finite-difference and calendar-block checks for the weekly-aware daily objective. No fitting."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('daily_weekloss_test',ROOT/'train_daily_to_weekly_round2.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.initialize()

@pytest.mark.parametrize('variant',m.VARIANTS,ids=[x['id'] for x in m.VARIANTS])
def test_gradient_matches_declared_weekly_plus_daily_loss(variant):
    z=np.linspace(-1.,1.,14);y=np.array([0,0,1,0,2,0,0,3,0,1,0,0,0,0.],float)
    gradient,hessian=m.gradients(z,y,variant);epsilon=1e-5;numeric=[]
    for i in range(len(z)):
        plus=z.copy();minus=z.copy();plus[i]+=epsilon;minus[i]-=epsilon
        numeric.append((m.loss_value(plus,y,variant)-m.loss_value(minus,y,variant))/(2*epsilon))
    np.testing.assert_allclose(gradient,numeric,rtol=1e-6,atol=1e-7)
    assert (hessian>0).all() and np.isfinite(hessian).all()

@pytest.mark.parametrize('variant',m.VARIANTS,ids=[x['id'] for x in m.VARIANTS])
def test_small_gradient_step_reduces_training_loss(variant):
    z=np.zeros(14);y=np.array([0,0,1,0,2,0,0,3,0,1,0,0,0,0.])
    gradient,hessian=m.gradients(z,y,variant)
    assert m.loss_value(z-.001*gradient/hessian,y,variant)<m.loss_value(z,y,variant)

def blocks():
    f=pd.DataFrame({'date':pd.date_range('2026-01-19',periods=14),'week':np.repeat(pd.to_datetime(['2026-01-19','2026-01-26']),7),
        'feature_origin':pd.Timestamp('2026-01-18'),'warehouse_name':'W','article':'A','target':1.})
    return f

def test_weekly_loss_groups_preserve_calendar_targets():
    np.testing.assert_array_equal(m.group_targets(blocks()),[7.,7.])

@pytest.mark.parametrize('column,value',[('article','B'),('feature_origin',pd.Timestamp('2026-01-11')),
    ('date',pd.Timestamp('2026-02-01'))])
def test_weekly_loss_rejects_mixed_or_incomplete_groups(column,value):
    f=blocks();f.loc[2,column]=value
    with pytest.raises(ValueError):m.group_targets(f)

def test_zero_target_week_has_positive_gradient_toward_lower_demand():
    g,h=m.gradients(np.zeros(7),np.zeros(7),m.VARIANTS[1])
    assert (g>0).all() and (h>0).all()

def test_environment_relaunches_round2_instead_of_completed_round1(monkeypatch):
    import importlib.metadata
    import os
    import subprocess
    existing=m.ROOT/'.venv-all40'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    original=m.Path.exists
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'0.0.0')
    monkeypatch.setattr(m.Path,'exists',lambda path:True if path==existing else original(path))
    monkeypatch.setattr(sys,'executable',str(m.ROOT/'different-interpreter.exe'))
    commands=[];monkeypatch.setattr(subprocess,'check_call',lambda command:commands.append(command))
    assert m.environment(check=True)
    assert len(commands)==1 and m.Path(commands[0][1]).name=='train_daily_to_weekly_round2.py'
