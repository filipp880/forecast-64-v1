import copy

import numpy as np

from scripts.select_smape_wape_joint import (combine_predictions,compile_recipe,count_options,
    regular_options,metric_assessment,choose_record,ORIGINS)


def test_declared_finite_budget_and_regular_upper_boundary():
    assert len(count_options())==15
    assert len(regular_options())==13
    assert 6*len(count_options())*len(regular_options())==1170
    weekly=np.array([.2,1.,3.,15.,60.])
    q=combine_predictions(np.ones(5),np.full(5,2.),np.full(5,3.),weekly,
        {'scope':'low_medium','lambda_value':8.},
        {'scope':'regular','variant':'recent_mae','blend':1.})
    np.testing.assert_array_equal(q,[1.,2.,2.,3.,1.])


def test_compilation_uses_original_base_and_regular_overwrites_count():
    champion={'source':'dynamics','post':{'round_int':True}}
    router={**champion,'segments':[{'min_weekly_mean':0.,'max_weekly_mean':.5,'action':'zero'}]}
    recipe=compile_recipe(champion,router,{'scope':'low_medium','lambda_value':8.},
                           {'scope':'all_regular','variant':'recent_mae','blend':.5})
    assert recipe['source']=='dynamics'
    assert recipe['segments'][0]['action']=='zero'
    count=recipe['segments'][1]
    assert (count['min_weekly_mean'],count['max_weekly_mean'])==(.5,2.)
    assert count['recipe']['source']=='metric_count_lambda8'
    regular=recipe['segments'][2]
    assert (regular['min_weekly_mean'],regular['max_weekly_mean'])==(2.,None)
    assert regular['keep_base_zeros'] is True
    assert 'keep_base_mask' not in regular
    assert champion=={'source':'dynamics','post':{'round_int':True}}


def test_joint_primary_and_auxiliary_tags_are_separate():
    champions={o:dict(smape=60.,wape=50.,rmse=3.,bias=70.,rmsle=.5) for o in ORIGINS}
    candidate=copy.deepcopy(champions)
    for value in candidate.values():value.update(smape=58.,wape=49.,rmse=4.)
    result=metric_assessment(candidate,champions)
    assert result['primary_eligible'] and not result['balanced']
    candidate[ORIGINS[0]]['wape']=55.
    assert not metric_assessment(candidate,champions)['primary_eligible']


def test_tie_prefers_balanced_before_complexity():
    def record(balanced,source):
        return dict(router_id='r',count_id='none',regular_id='none',recipe={'source':source},
            assessment={'primary_eligible':True,'balanced':balanced,'primary_score':.99},
            october_guard={'passed':True})
    balanced=record(True,'a');other=record(False,'b')
    assert choose_record([other,balanced]) is balanced
