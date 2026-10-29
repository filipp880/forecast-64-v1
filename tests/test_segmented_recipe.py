import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.research_model import (recipe_prediction, recipe_member_names,
                                      validate_segments, WeeklyResearchModel)
from app.pipeline import needs_money
from app.features import CAT_FEATURES, KEY
from app.predictor import Predictor, ModelNotReadyError
from train import save_artifact


def test_segment_half_open_ranges_blends_and_mask_are_past_only():
    frame = pd.DataFrame({'mean_28': np.array([0., .5, 1., 2., 20.])/7,
                          'warehouse_name': 'w', 'article': list('abcde')})
    values = {'mae': {'raw': np.array([0., 1., 2., 3., 4.])},
              'poisson': {'raw': np.array([5., 0., 1., 4., 8.])}}
    recipe = {'source': 'mae', 'post': {'round_int': False}, 'segments': [
        {'min_weekly_mean': 0., 'max_weekly_mean': .5, 'action': 'zero'},
        {'min_weekly_mean': .5, 'max_weekly_mean': 2., 'blend_weight': .5,
         'keep_base_mask': True, 'recipe': {'source': 'poisson', 'post': {'round_int': False}}}]}
    np.testing.assert_allclose(recipe_prediction(values, frame, recipe), [0., .5, 1.5, 3., 4.])
    assert recipe_member_names(recipe) == {'mae', 'poisson'}
    assert needs_money('research', {'recipe': recipe})
    changed = frame.assign(actual=[999., 0., 0., 999., 10.])
    np.testing.assert_array_equal(recipe_prediction(values, changed, recipe), recipe_prediction(values, frame, recipe))


@pytest.mark.parametrize('segments', [
    [{'min_weekly_mean': 0., 'max_weekly_mean': 2., 'action': 'zero'},
     {'min_weekly_mean': 1., 'max_weekly_mean': 3., 'action': 'zero'}],
    [{'min_weekly_mean': -1., 'max_weekly_mean': 2., 'action': 'zero'}],
    [{'min_weekly_mean': 1., 'max_weekly_mean': 2., 'action': 'zero', 'blend_weight': 2.}],
    [{'min_weekly_mean': 1., 'max_weekly_mean': 2., 'action': 'unknown'}],
])
def test_invalid_segment_specs_rejected(segments):
    with pytest.raises(ValueError):
        validate_segments({'source': 'dynamics', 'segments': segments})


def test_segmented_artifact_roundtrip_and_schema_guard(tmp_path):
    dates = pd.date_range('2025-01-01', periods=365)
    sales = pd.DataFrame({'date': dates, 'quantity': np.arange(len(dates)) % 3,
                          'sales_sum': 1., 'discount_sum': np.nan})
    for c in CAT_FEATURES:
        sales[c] = 'x'
    stock = sales[['date', *KEY]].assign(stock=10.)
    recipe = {'source': 'dynamics', 'post': {'round_int': True}, 'segments': [
        {'min_weekly_mean': 0., 'max_weekly_mean': None,
         'recipe': {'source': 'mae', 'post': {'round_int': False}}, 'blend_weight': .5}]}
    from app.pipeline import prepare_panel, feature_frame
    panel = prepare_panel(sales, stock, 'research', {'recipe': recipe})
    x = feature_frame(panel, dates.max(), 'research')
    model = WeeklyResearchModel({'recipe': recipe, 'threads': 1, 'estimators': 2})
    model.fit(x, np.arange(9, dtype=float))
    assert set(model.members) == {'dynamics', 'mae'}
    config = save_artifact(model, tmp_path, dates.max(), dates.max(), 9, 1)
    assert config['preprocessing_version'] == 6
    loaded = Predictor(tmp_path).load()
    np.testing.assert_allclose(loaded.predict(sales, stock).quantity, model.predict(x))
    config['preprocessing_version'] = 4
    (tmp_path/'config.json').write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ModelNotReadyError, match='preprocessing_version'):
        Predictor(tmp_path).load()
