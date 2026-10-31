"""No-fit checks for policy cells, count risk, balanced sampling and residual bounds."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import pytest
from scripts import weekly_refinement_common as c

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('weekly_refinement_test',ROOT/'train_weekly_refinement.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.initialize()

def test_demand_boundaries_and_horizon_groups_match_production():
    f=pd.DataFrame({'mean_28':np.array([0,.5,2,10,50])/7,'h':[1,3,4,6,9]})
    r,h=c.cells(f);np.testing.assert_array_equal(r,[0,1,2,3,4]);np.testing.assert_array_equal(h,[0,0,1,1,2])

def test_decoder_matches_published_fixed_lambda_function():
    from app.models.all40_model import decode_count
    schema=dict(representatives=[0.,.5,1.5,5.],training_mean=2.)
    p=np.array([[.8,.1,.05,.05],[0.,.2,.3,.5]])
    np.testing.assert_array_equal(c.decode(p,schema,3.),decode_count(p,schema))

@pytest.mark.parametrize('variant',m.VARIANTS,ids=[x['id'] for x in m.VARIANTS])
def test_count_bins_reserve_zero_and_cover_last_class(variant):
    y=np.array([0.,.01,.5,1.,2.,1000.]);lab=m.labels(y,variant['edges'])
    assert lab[0]==0 and (lab[1:]>0).all() and lab[-1]==len(variant['edges'])+1

def test_equal_series_weights_give_equal_total_pair_mass():
    f=pd.DataFrame({'warehouse_name':['W']*4,'article':['A','A','A','B'],'mean_28':[1.]*4})
    w=m.sample_weights(f,'equal_series');assert np.isclose(w[:3].sum(),w[3]) and np.isclose(w.mean(),1.)

def test_rare_weights_use_past_demand_not_target():
    f=pd.DataFrame({'warehouse_name':['W']*2,'article':['A','B'],'mean_28':[.1,2.],'target':[999,0]})
    a=m.sample_weights(f,'rare');f['target']=[0,999];b=m.sample_weights(f,'rare')
    np.testing.assert_array_equal(a,b);assert np.isclose(a[0]/a[1],1.5)

def test_residual_correction_is_bounded_and_non_negative():
    prefix=np.array([0.,1.,20.]);delta=np.array([-99.,99.,99.])
    np.testing.assert_array_equal(m.residual_apply(prefix,delta),[0.,2.,25.])

def test_policy_reproduces_mixing_and_final_rounding():
    f=pd.DataFrame({'mean_28':[0.,1/7,5/7,100/7],'h':[1,4,7,9]})
    p=np.array([[1,0],[0,1],[0,1],[0,1.]])
    context=dict(legacy=np.array([2.,9.,9.,2.]),refreshed=np.array([4.,4.,4.,4.]),
        probability=p,schema=dict(representatives=[0.,2.],training_mean=1.),cells=c.cells(f),decoded={})
    policy=c.baseline_policy();result=c.apply_policy(context,policy)
    np.testing.assert_array_equal(result,np.rint(.75*np.array([2.,2.,2.,2.])+.25*4.))
