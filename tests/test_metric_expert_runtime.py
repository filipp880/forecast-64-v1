import json

import numpy as np
import pandas as pd
import pytest

from app.models.metric_experts import bayes_quantities, class_representatives, quantity_classes
from app.models.research_model import WeeklyResearchModel, recipe_prediction
from app.features import CAT_FEATURES, KEY
from app.pipeline import prepare_panel, feature_frame
from app.predictor import Predictor, ModelNotReadyError
from train import save_artifact


def inputs():
    dates = pd.date_range('2025-01-01', '2025-12-31')
    sales = pd.DataFrame({'date': dates, 'quantity': np.arange(len(dates)) % 3,
                          'sales_sum': 1., 'discount_sum': np.nan})
    for c in CAT_FEATURES:
        sales[c] = 'x'
    newer = sales.loc[sales.date >= '2025-10-01'].copy()
    newer['article'] = 'new'
    newer['quantity'] = 3.
    sales = pd.concat([sales, newer], ignore_index=True)
    return sales, sales[['date', *KEY]].assign(stock=10.)


def test_sequence_builder_matches_original_cached_feature_algorithm():
    from app.enhancements import augment_features
    from app.ideas_features import augment_ideas
    from app.event_features import make_event_features
    from app.sequence_features import add_sequence_features
    sales, stock = inputs()
    panel = prepare_panel(sales, stock, 'research', {'recipe': {'source':'dynamics'}})
    base = feature_frame(panel, '2025-10-26', 'research')
    original = make_event_features(augment_ideas(panel, base, '2025-10-26'))
    efficient = make_event_features(add_sequence_features(panel, base, '2025-10-26'))
    pd.testing.assert_frame_equal(efficient, original, check_dtype=False)
    recent = efficient[efficient.article == 'new']
    assert recent.event_support_fraction.max() < 1.
    assert recent.weekly_lag_13.isna().all()


def test_count_decoder_matches_frozen_experimental_decoder():
    from scripts.research_count_distribution import decode
    y = np.array([0., .5, 1., 2., 4., 6., 10., 30.])
    representatives = class_representatives(y)
    probabilities = np.random.default_rng(42).dirichlet(np.ones(8), size=50)
    for multiplier in (0, 1, 2, 3, 4, 8, 16):
        np.testing.assert_array_equal(bayes_quantities(probabilities, representatives, y.mean(), multiplier),
                                      decode(probabilities, representatives, y.mean(), multiplier))
    np.testing.assert_array_equal(quantity_classes(y), [0, 1, 1, 2, 4, 5, 6, 7])


def test_one_sided_zero_mask_allows_removing_false_positive():
    f = pd.DataFrame({'mean_28':[1.], 'warehouse_name':['w'], 'article':['a']})
    values = {'mae': {'raw':np.array([1.])}, 'quantile': {'raw':np.array([0.])}}
    recipe = {'source':'mae', 'post':{'round_int':False}, 'segments':[
        {'min_weekly_mean':0., 'max_weekly_mean':None,
         'recipe':{'source':'quantile','post':{'round_int':False}}, 'keep_base_zeros':True}]}
    assert recipe_prediction(values,f,recipe)[0] == 0.


def test_expert_artifact_roundtrip_and_preprocessing_guard(tmp_path):
    sales, stock = inputs()
    name = 'metric_count_lambda8'
    recipe = {'source':'dynamics', 'post':{'round_int':True}, 'segments':[
        {'min_weekly_mean':0., 'max_weekly_mean':None,
         'recipe':{'source':name,'post':{'round_int':False}}, 'blend_weight':.5}]}
    params = {'recipe':recipe, 'threads':1, 'estimators':2,
              'expert_specs':{name:{'kind':'count','multiplier':8.}}}
    panel = prepare_panel(sales,stock,'research',params)
    train = feature_frame(panel,'2025-10-26','research',training=True,model_params=params)
    model = WeeklyResearchModel(params)
    model.fit(train,train.target.to_numpy(float),n_iterations={
        'dynamics':{'hurdle':{'classifier':2,'regressor':2},'lgbm':2}, name:2})
    config = save_artifact(model,tmp_path,'2025-12-31','2025-12-31',len(train),1)
    assert config['preprocessing_version'] == 7
    loaded = Predictor(tmp_path).load()
    end = feature_frame(panel,'2025-12-31','research',model_params=params)
    np.testing.assert_allclose(loaded.predict(sales,stock).quantity,model.predict(end))
    config['metric_experts']['version'] = 0
    (tmp_path/'config.json').write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(ModelNotReadyError,match='metric_experts'):
        Predictor(tmp_path).load()
