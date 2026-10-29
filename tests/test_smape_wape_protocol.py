import copy

import numpy as np
import pandas as pd

from app.metrics import compute_all_metrics
from scripts.research_smape_wape import (ORIGINS, assessment, build_recipe,
    metric_denominators, metrics_from_stats, policy_grid, policy_prediction, segment_mask, sufficient_stats)


def test_sufficient_statistics_add_exactly_across_series_segments():
    y=np.array([0.,1.,2.,3.,0.,5.])
    p=np.array([0.,2.,1.,2.,1.,4.])
    codes=np.array([0,0,0,1,1,1])
    a=codes==0
    total=sufficient_stats(y[a],p[a],codes[a])+sufficient_stats(y[~a],p[~a],codes[~a])
    measured=metrics_from_stats(total,metric_denominators(y,codes))
    frame=pd.DataFrame({'week':list(pd.date_range('2026-01-05',periods=3,freq='7D'))*2,
                        'warehouse_name':'w','article':['a']*3+['b']*3,'quantity':y})
    predicted=frame.assign(quantity=p)
    direct=compute_all_metrics(frame,predicted)
    for key in direct:
        np.testing.assert_allclose(measured[key],direct[key],rtol=1e-12)


def test_segments_use_only_past_means_at_declared_boundaries():
    f=pd.DataFrame({'mean_28':np.array([0.,.5,2.,10.,50.])/7,'actual':[999.]*5})
    np.testing.assert_array_equal(segment_mask(f,.5,2.),[False,True,False,False,False])
    np.testing.assert_array_equal(segment_mask(f,50.,None),[False,False,False,False,True])


def test_two_primary_metrics_both_must_improve():
    champion={o:dict(smape=60.,wape=50.,rmse=3.,bias=70.,rmsle=.5) for o in ORIGINS}
    candidate=copy.deepcopy(champion)
    for values in candidate.values():
        values.update(smape=59.,wape=49.)
    assert assessment(candidate,champion)['eligible']
    candidate[ORIGINS[0]]['wape']=52.
    assert not assessment(candidate,champion)['eligible']


def test_grid_is_finite_and_routing_recipe_keeps_champion_for_unselected_segments():
    champion={'source':'dynamics','post':{'medium_gate':.5,'regular_scale':1.1,'round_int':True,'round_threshold':.5}}
    grid=policy_grid(champion)
    assert all(len(v['policies'])<=100 for v in grid.values())
    selection={name:'champion' for name in grid}
    selection['rare']='zero'
    recipe=build_recipe(champion,selection,grid)
    assert recipe['segments']==[{'min_weekly_mean':0.,'max_weekly_mean':.5,'action':'zero'}]
    assert champion.get('segments') is None


def test_search_evaluator_matches_deployable_runtime_router():
    from app.models.research_model import recipe_prediction
    champion={'source':'dynamics','post':{'medium_gate':.5,'regular_scale':1.1,'round_int':True,'round_threshold':.5}}
    grid=policy_grid(champion)
    frame=pd.DataFrame({'warehouse_name':['w']*45,'article':np.repeat(['a','b','c','d','e'],9),
                         'mean_28':np.repeat(np.array([.2,1.,3.,15.,60.])/7,9),
                         'n_sale_days_90':np.repeat([3.,15.,30.,60.,80.],9)})
    values={'dynamics':{'probability':np.linspace(.2,.9,45),'magnitude':np.linspace(.5,20,45),
                         'lgbm_raw':np.linspace(.1,15,45)},'mae':{'raw':np.linspace(0,19,45)}}
    selection={n:'champion' for n in grid}
    selection['rare']='zero'
    chosen=next(n for n,p in grid['regular']['policies'].items()
                if p.get('recipe',{}).get('source')=='mae' and p.get('blend_weight')==.5
                and p.get('keep_base_mask') and p['recipe']['post']['round_int']==False)
    selection['regular']=chosen
    recipe=build_recipe(champion,selection,grid)
    base=recipe_prediction(values,frame,champion)
    manual=base.copy()
    for name,definition in grid.items():
        if selection[name]=='champion':
            continue
        mask=segment_mask(frame,definition['lower'],definition['upper'])
        q=policy_prediction(values,frame,base,definition['policies'][selection[name]])
        manual[mask]=q[mask]
    np.testing.assert_array_equal(manual,recipe_prediction(values,frame,recipe))
