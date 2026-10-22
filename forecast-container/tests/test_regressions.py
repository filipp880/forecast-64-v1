import sys, os, json, pickle, shutil
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'forecast-container'))
os.environ['MODEL_DIR'] = str(ROOT / 'forecast-container/model')
from app.validators import validate_sales_csv, validate_stock_csv
from app.metrics import compute_all_metrics
from app.predictor import Predictor, ModelNotReadyError

@pytest.fixture
def inputs():
    n = 180
    dates = pd.date_range('2025-01-01', periods=n).strftime('%Y-%m-%d')
    s = pd.DataFrame({c: ['x'] * n for c in ['composite_division', 'cities_name', 'warehouse_name', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3', 'article', 'characteristic']})
    s['date'] = dates
    s['quantity'] = 1.0
    s['sales_sum'] = 1.0
    s['discount_sum'] = 0.0
    t = pd.DataFrame({'date': dates, 'warehouse_name': 'x', 'article': 'x', 'stock': 1.0})
    return (s, t)

@pytest.mark.parametrize('value', ['abc', np.nan, np.inf, -np.inf])
def test_bad_quantity_rejected(inputs, value):
    (s, t) = inputs
    s['quantity'] = s['quantity'].astype(object)
    s.loc[0, 'quantity'] = value
    assert validate_sales_csv(s)

@pytest.mark.parametrize('column', ['date', 'warehouse_name', 'article'])
def test_empty_key_rejected(inputs, column):
    (s, t) = inputs
    s.loc[0, column] = None
    assert validate_sales_csv(s)

def test_negative_stock_allowed(inputs):
    (s, t) = inputs
    t.loc[0, 'stock'] = -2
    assert validate_stock_csv(t) == []

def test_exact_180_and_179(inputs):
    (s, t) = inputs
    assert not validate_sales_csv(s)
    assert validate_sales_csv(s.iloc[1:])

def test_numeric_stock_rejected(inputs):
    (s, t) = inputs
    t['stock'] = t['stock'].astype(object)
    t.loc[0, 'stock'] = 'broken'
    assert validate_stock_csv(t)

def test_duplicate_metric_keys_rejected():
    t = pd.DataFrame({'date': ['2025-01-01'] * 2, 'warehouse_name': 'w', 'article': 'a', 'quantity': 1.0})
    with pytest.raises(ValueError):
        compute_all_metrics(t, t)

def test_origin_alignment():
    t = pd.DataFrame({'origin': ['2024-12-01', '2024-12-02', '2024-12-02'], 'date': ['2025-01-01', '2025-01-01', '2025-01-02'], 'warehouse_name': 'w', 'article': 'a', 'quantity': 1.0})
    p = t.copy()
    p.quantity = [1.0, 1.0, 0.0]
    assert compute_all_metrics(t, p)['wape'] == pytest.approx(100 / 3)

def test_atomic_load(tmp_path):
    src = ROOT / 'forecast-container/model'
    for f in src.iterdir():
        shutil.copyfile(f, tmp_path / f.name)
    path = tmp_path / 'preprocessor.pkl'
    spec = pickle.loads(path.read_bytes())
    spec['horizon'] = 65
    path.write_bytes(pickle.dumps(spec))
    p = Predictor(tmp_path)
    with pytest.raises(ModelNotReadyError):
        p.load()
    assert not p.is_loaded
    assert p.config == {}

def test_research_targets_disjoint():
    from app.temporal import research_split
    (train, valid) = research_split([pd.Timestamp('2025-10-18')], 64, 28, 7)
    assert max(train) + pd.Timedelta(days=64) <= valid
    assert valid + pd.Timedelta(days=64) <= pd.Timestamp('2025-10-18')

def test_zero_zero_stays_in_smape():
    from app.metrics import smape_series
    assert smape_series(np.array([0.0, 1.0]), np.array([0.0, 0.0])) == 100.0

def test_stock_window_has_no_stale_carry(inputs):
    from app.features import build_panel, make_features
    (s, t) = inputs
    s = pd.concat([s, s.iloc[:20]], ignore_index=True)
    s['date'] = pd.date_range('2025-01-01', periods=200)
    t = t.iloc[:1].copy()
    t['stock'] = 7.0
    cut = s.date.max()
    full = make_features(build_panel(s, t), cut, [1])
    recent = make_features(build_panel(s.iloc[-180:], t.iloc[0:0]), cut, [1])
    assert np.isnan(full.stock_last.iloc[0]) and np.isnan(recent.stock_last.iloc[0])
    legacy = make_features(build_panel(s, t), cut, [1], stock_policy='legacy')
    assert legacy.stock_last.iloc[0] == 7

def test_future_values_do_not_change_features(inputs):
    from app.features import build_panel, make_features
    (s, t) = inputs
    cut = pd.Timestamp(s.date.iloc[-5])
    before = make_features(build_panel(s, t), cut, [1, 2])
    s.loc[pd.to_datetime(s.date) > cut, 'quantity'] = 999
    t.loc[pd.to_datetime(t.date) > cut, 'stock'] = 999
    pd.testing.assert_frame_equal(before, make_features(build_panel(s, t), cut, [1, 2]))

def test_submission_wrong_start_and_duplicates(inputs):
    from app.submission import validate_forecast
    (s, t) = inputs
    dates = pd.date_range(pd.to_datetime(s.date).max() + pd.Timedelta(days=1), periods=64).strftime('%Y-%m-%d')
    f = pd.DataFrame({'date': dates, 'warehouse_name': 'x', 'article': 'x', 'quantity': 0.0})
    assert validate_forecast(f, s)['rows'] == 64
    with pytest.raises(ValueError):
        validate_forecast(pd.concat([f, f.iloc[:1]]), s)
    f.loc[0, 'date'] = '2000-01-01'
    with pytest.raises(ValueError):
        validate_forecast(f, s)

def test_api_errors_history_and_filters(inputs):
    import app.main as main
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    (s, t) = inputs

    def post(a, b):
        return c.post('/predict', files={'sales': ('s.csv', a.to_csv(index=False).encode()), 'stock': ('t.csv', b.to_csv(index=False).encode())})
    assert c.post('/predict').json()['code'] == 'VALIDATION_ERROR'
    s_bad = s.copy()
    s_bad['quantity'] = 'broken'
    assert post(s_bad, t).status_code == 422
    assert post(s.iloc[1:], t).status_code == 422
    t['stock'] = -2
    r = post(s, t)
    assert r.status_code == 200
    import io
    f = pd.read_csv(io.BytesIO(r.content))
    assert len(f) == 64
    for u in ['/predict?article=x', '/predict?warehouse=x', '/predict?article=x&warehouse=x', '/predict/all']:
        rr = c.get(u)
        assert rr.status_code == 200
        pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(rr.content)), f)
    assert len(c.get('/predict/all?format=json').json()) == 64
    assert c.get('/predict').status_code == 400
    assert c.get('/predict/all?format=invalid').status_code == 400

def test_metrics_string_and_datetime_keys():
    t = pd.DataFrame({'date': [pd.Timestamp('2025-01-01')], 'warehouse_name': 'w', 'article': 'a', 'quantity': 1.0})
    p = t.copy()
    p.date = ['2025-01-01']
    assert compute_all_metrics(t, p)['rmse'] == 0

def test_no_fake_training_targets(inputs):
    from app.features import build_panel, make_training_frame
    (s, t) = inputs
    panel = build_panel(s, t)
    with pytest.raises(ValueError):
        make_training_frame(panel, panel.date.max(), [1])

def test_baseline_origin_and_rounded(inputs):
    from app.features import build_panel
    from baseline import baseline_forecast, truth_for
    (s, t) = inputs
    panel = build_panel(s, t)
    cut = panel.date.max() - pd.Timedelta(days=64)
    truth = truth_for(panel, cut, 64)
    truth['origin'] = str(cut.date())
    for method in ['zero', 'mean28_round', 'mean90_round', 'dow_round']:
        pred = baseline_forecast(panel, cut, method, 64, s[['warehouse_name', 'article']].drop_duplicates())
        pred['origin'] = str(cut.date())
        assert np.isfinite(compute_all_metrics(truth, pred)['smape'])

def test_calibration_incumbent_parity():
    from app.calibration import combine
    p = np.array([0.2, 0.5, 0.8])
    m = np.array([2.0, 3.0, 4.0])
    lg = np.array([0.2, 0.8, 2.0])
    mean = np.ones(3)
    freq = np.array([2, 20, 60])
    np.testing.assert_array_equal(combine(p, m, lg, mean, freq), np.rint(0.3 * np.where(p >= 0.5, m, 0) + 0.7 * lg))

def test_future_catalog_not_in_strict_snapshot(inputs):
    from app.features import build_panel, make_features
    (s, t) = inputs
    s.loc[s.index[-1], 'article'] = 'future_product'
    cut = pd.to_datetime(s.date.iloc[-2])
    panel = build_panel(s, t)
    assert 'future_product' not in set(make_features(panel, cut, [1]).article)
    assert 'future_product' in set(make_features(panel, cut, [1], stock_policy='legacy').article)

def test_direct_prediction_without_python_network(inputs, monkeypatch):
    import socket

    def denied(*args, **kwargs):
        raise AssertionError('Сетевое обращение при инференсе')
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    p = Predictor(ROOT / 'forecast-container/model').load()
    (s, t) = inputs
    assert len(p.predict(s, t)) == 64

def test_failed_reload_clears_prior_cache(tmp_path, inputs):
    src = ROOT / 'forecast-container/model'
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, tmp_path / f.name)
    p = Predictor(tmp_path).load()
    (s, t) = inputs
    p.predict(s, t)
    (tmp_path / 'model.pkl').write_bytes(b'invalid')
    with pytest.raises(ModelNotReadyError):
        p.load()
    assert not p.is_loaded and p._cached_forecast is None

def test_registry_is_fixed_before_validation(tmp_path):
    sys.path.insert(0, str(ROOT / 'forecast-container/scripts'))
    from select_candidate import initialize_grid
    initialize_grid(tmp_path)
    saved = (tmp_path / 'candidate_grid.json').read_bytes()
    initialize_grid(tmp_path)
    assert (tmp_path / 'candidate_grid.json').read_bytes() == saved
