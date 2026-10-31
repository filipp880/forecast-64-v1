"""Request-only joint serving contracts; all model weights are already fitted."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.predictor import Predictor
from app.joint_features import make_joint_daily
from app.pipeline import prepare_panel, feature_frame
from app.advanced_features import augment_advanced_features
from test_all40_serving import inputs

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / 'artifacts/joint-serving-20261031/model'


@pytest.fixture(scope='module')
def predictor():
    # The committed root bundle works after release export, including in a clean checkout.
    directory = STAGED if (STAGED / 'model.pkl').exists() else ROOT / 'model'
    p = Predictor(directory).load()
    if p.model.params.get('serving_profile') != 'joint_v1':
        pytest.skip('joint release weights not available')
    return p


def test_joint_repeat_future_invariance_subset_and_cold_pairs(predictor, monkeypatch):
    import lightgbm as lgb
    monkeypatch.setattr(lgb, 'train', lambda *a, **k: pytest.fail('Serving attempted to train'))
    sales, stock = inputs()
    expected = predictor.predict(sales, stock, history_cutoff='2026-07-26')
    pd.testing.assert_frame_equal(expected, predictor.predict(sales, stock, history_cutoff='2026-07-26'))
    future = sales.iloc[:1].assign(date=pd.Timestamp('2026-08-01'), quantity=1e6, article='future_only')
    future_stock = stock.iloc[:1].assign(date=pd.Timestamp('2026-08-01'), stock=1e6)
    actual = predictor.predict(pd.concat([sales, future], ignore_index=True),
        pd.concat([stock, future_stock], ignore_index=True), history_cutoff='2026-07-26')
    pd.testing.assert_frame_equal(expected, actual)
    pairs = sales[['warehouse_name', 'article']].drop_duplicates().iloc[:1]
    subset = predictor.predict(sales, stock, history_cutoff='2026-07-26', required_pairs=pairs)
    pd.testing.assert_frame_equal(subset, expected.merge(pairs, on=['warehouse_name', 'article']).reset_index(drop=True))
    assert len(expected) == 36 and np.isfinite(expected.quantity).all()


@pytest.mark.parametrize('cutoff,first', [('2026-07-24', '2026-07-27'), ('2026-07-27', '2026-08-03')])
def test_daily_calendar_uses_complete_weeks_at_non_sunday_origins(predictor, cutoff, first):
    sales, stock = inputs()
    sales = sales.loc[sales.date <= cutoff]
    if cutoff == '2026-07-27':
        sales = pd.concat([sales, sales.iloc[:1].assign(date=pd.Timestamp(cutoff))], ignore_index=True)
    stock = stock.loc[stock.date <= cutoff]
    params = predictor.model.params
    panel = prepare_panel(sales, stock, 'research', params, cutoff)
    weekly = feature_frame(panel, cutoff, 'research', model_params=params)
    weekly = augment_advanced_features(sales, stock, weekly, cutoff)
    daily = make_joint_daily(sales, weekly, cutoff)
    assert str(daily.date.min().date()) == first
    assert daily.groupby(['week', 'warehouse_name', 'article'], observed=True).date.nunique().eq(7).all()
    assert daily.bridge_week_horizon.min() == 1 and daily.bridge_week_horizon.max() == 9
    assert daily.h.min() == (pd.Timestamp(first) - pd.Timestamp(cutoff)).days
    assert daily.bridge_lag_13.isna().any() or daily.bridge_support_13.min() > 0
    assert np.isfinite(predictor.model.predict_from_history(weekly, sales, cutoff)).all()


def test_joint_graph_and_daily_transform_are_validated(predictor):
    model = copy.copy(predictor.model)
    model.params = copy.deepcopy(model.params)
    model.params['joint_graph']['daily_weight'] = .25
    with pytest.raises(ValueError, match='graph'):
        model.check_schema()
    part = copy.copy(predictor.model.daily)
    part.schema = copy.deepcopy(part.schema)
    part.schema['prediction'] = 'raw'
    with pytest.raises(ValueError, match='transform'):
        part.check_schema()
    with pytest.raises(RuntimeError, match='never trains'):
        predictor.model.fit(None, None)
    with pytest.raises(ValueError, match='raw request history'):
        predictor.model.predict(pd.DataFrame())


def test_joint_http_request_and_cache_match(predictor, monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setattr(main, 'predictor', predictor)
    files = {name: (name + '.csv', (ROOT / 'data' / filename).read_bytes()) for name, filename in
             [('sales', 'sample_sales_6mo.csv'), ('stock', 'sample_stock_6mo.csv')]}
    with TestClient(main.app) as client:
        first = client.post('/predict', files=files)
        assert first.status_code == 200, first.text
        assert client.get('/predict/all').content == first.content
        repeat = client.post('/predict', files=files)
        assert repeat.status_code == 200 and repeat.content == first.content


@pytest.mark.parametrize('profile', ['all40_v1', 'joint_v1'])
def test_generic_trainer_rejects_exported_graph_before_reading_data(profile):
    import train
    with pytest.raises(SystemExit) as error:
        train.parse_args(['--model-type', 'research', '--model-params', json.dumps(dict(serving_profile=profile))])
    assert error.value.code == 2
