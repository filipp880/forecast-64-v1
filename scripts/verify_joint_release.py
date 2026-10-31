"""No-fit parity checks against frozen historical and final joint artifacts."""
from __future__ import annotations
import ast
import copy
import gc
import json
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_joint_release import OUT, DAILY, REFERENCE, SOURCE, EXPECTED_FORECAST, read, sha, write
from app.models.all40_model import All40WeeklyModel, NativePart
from app.models.joint_model import JointWeeklyDailyModel, DailyQuantityPart
from app.joint_features import JOINT_DAILY_COLUMNS, make_joint_daily
from app.csv_io import read_csv_checked
from app.metrics import compute_all_metrics
from app.predictor import Predictor

ALL40 = ROOT / 'artifacts/all40-0a385edab6b1'
ROUND2 = DAILY.parents[2]
ORIGINS = ['2026-01-18', '2026-03-22']
WK = ['week', 'warehouse_name', 'article']


def compare_daily(actual, reference):
    pd.testing.assert_frame_equal(actual[['date', *WK]].astype(str), reference[['date', *WK]].astype(str),
                                  check_dtype=False)
    maximum = 0.
    for c in JOINT_DAILY_COLUMNS:
        if c in ('warehouse_name', 'article', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3', 'characteristic'):
            np.testing.assert_array_equal(actual[c].astype(str), reference[c].astype(str))
        else:
            a, b = actual[c].to_numpy(float), reference[c].to_numpy(float)
            np.testing.assert_array_equal(a, b, err_msg=c)
            maximum = max(maximum, float(np.max(np.abs(np.nan_to_num(a - b)))))
    return dict(rows=len(actual), columns=len(JOINT_DAILY_COLUMNS), max_difference=maximum)


def reference_functions():
    requirements = ROOT / 'artifacts/repository-before-root-layout-20261028/docs/requirements'
    document = next(requirements.glob('*обновлен.md')).read_text(encoding='utf-8')
    names = {'smape_series', 'smape', 'wape', 'rmse', 'bias_abs_series', 'bias_abs', 'rmsle'}
    for block in document.split('```python')[1:]:
        tree = ast.parse(block.split('```')[0])
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes} == names:
            namespace = dict(np=np, Optional=Optional)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original_requirements_metric_functions>', 'exec'), namespace)
            return namespace
    raise ValueError('Original requirements metric functions not found')


def main():
    import lightgbm as lgb
    def forbidden(*args, **kwargs):
        raise RuntimeError('Training prohibited during verification')
    lgb.train = forbidden
    sales = read_csv_checked(ROOT / 'data/sales_dataset_hakaton_train.csv')
    reference_metrics = reference_functions()
    source_model = Predictor(SOURCE).load().model
    manifest = read(REFERENCE / 'joint_candidate_manifest.json')
    result = dict(verified=False, training_executed=False, historical={},
                  requirements_metric_reference='function definitions extracted from original updated requirements')
    for origin in ORIGINS:
        print('VERIFY historical', origin, flush=True)
        f = pd.read_parquet(ALL40 / 'features' / f'{origin}_{origin}-False.parquet.parquet')
        daily = make_joint_daily(sales, f, origin)
        saved = pd.read_parquet(ROUND2 / 'evaluation_data' / f'{origin}_daily.parquet')
        parity = compare_daily(daily, saved)
        folder = ROUND2 / 'models/weekly_poisson_aux' / origin
        daily_part = DailyQuantityPart((folder / 'refit_model.txt').read_text(encoding='utf-8'), read(folder / 'refit_schema.json'))
        daily_part.check_schema()
        prediction = daily_part.predict(daily)
        saved_prediction = np.load(ROUND2 / 'evaluation_data' / f'{origin}_weekly_poisson_aux_daily_prediction.npy')
        np.testing.assert_array_equal(prediction, saved_prediction)
        parts = {}
        for name in source_model.parts:
            folder = ALL40 / 'models' / name / origin
            parts[name] = NativePart((folder / 'refit_model.txt').read_text(encoding='utf-8'), read(folder / 'refit_schema.json'))
        bundle = ROOT / 'artifacts/colab-sequential-20261029-final/colab_round08/data' / f'{origin}.npz'
        with np.load(bundle, allow_pickle=False) as data:
            frozen = data['baseline'].copy()
        baseline = copy.copy(source_model.baseline)
        baseline.predict = lambda frame, p=frozen: p.copy()
        model = JointWeeklyDailyModel(All40WeeklyModel(baseline, parts), daily_part)
        model.check_schema()
        quantities = model.predict_from_history(f, sales, origin)
        truth = f[WK].assign(quantity=f.actual.to_numpy(float))
        forecast = f[WK].assign(quantity=quantities)
        measured = compute_all_metrics(truth, forecast)
        codes = pd.factorize(pd.MultiIndex.from_frame(f[['warehouse_name', 'article']]))[0]
        y = f.actual.to_numpy(float)
        original = dict(smape=reference_metrics['smape'](y, quantities, codes),
            wape=reference_metrics['wape'](y, quantities), rmse=reference_metrics['rmse'](y, quantities),
            bias=reference_metrics['bias_abs'](y, quantities, codes), rmsle=reference_metrics['rmsle'](y, quantities))
        for key in measured:
            np.testing.assert_allclose(measured[key], manifest['historical_metrics'][origin][key], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(measured[key], original[key], rtol=1e-12, atol=1e-12)
        result['historical'][origin] = dict(daily_features=parity, daily_native_prediction_max_difference=0.,
            graph_metrics_match_saved=True, original_tz_metrics_match=True, metrics=measured,
            baseline_source='frozen origin-specific predictions, never July-fitted weights')
        del daily, saved, prediction, model, f, parts
        gc.collect()
    print('VERIFY final daily feature parity', flush=True)
    f = pd.read_parquet(ROOT / 'artifacts/weekly-refinement-865f00c31568/final_features.parquet')
    final_daily = make_joint_daily(sales, f, '2026-07-26')
    reference = pd.read_parquet(ROUND2 / 'evaluation_data/final_daily.parquet')
    result['final_daily_features'] = compare_daily(final_daily, reference)
    # Generator runs the complete request-only path twice and records deterministic equality.
    generated = OUT / 'forecast.csv'
    result['forecast_sha256'] = sha(generated)
    assert sha(generated) == EXPECTED_FORECAST
    generated_manifest = read(OUT / 'forecast.manifest.json')
    result['generator'] = dict(exact_submitted_bytes=True, rows=generated_manifest['rows'],
        predict_seconds=generated_manifest['predict_seconds'],
        repeat_seconds=generated_manifest['deterministic_repeat_seconds'],
        excluded_future_rows=generated_manifest['excluded_future_rows'])
    result['verified'] = True
    write(OUT / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
