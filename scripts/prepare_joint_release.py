"""Export the already fitted, externally best joint graph; never fit a model."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.predictor import Predictor
from app.joint_features import joint_spec
from app.models.joint_model import JointWeeklyDailyModel, DailyQuantityPart, joint_graph

SOURCE = ROOT / 'artifacts/all40-serving-20261030/model'
DAILY = ROOT / 'artifacts/daily-weekloss-7c8fdc7c3173/models/weekly_poisson_aux/FINAL'
REFERENCE = ROOT / 'artifacts/daily-weekloss-review-20261030'
OUT = ROOT / 'artifacts/joint-serving-20261031'
EXPECTED_FORECAST = 'acbfbd096f73a0b1903aa73e1f35c9220a14b9539e596b24221d9abce603601f'
EXPECTED_WEEKLY = '201c6b747ee351feff7f2017c2c45820862ad53fa0ab9cd9332425a4945d59fd'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    import lightgbm as lgb
    def forbidden(*args, **kwargs):
        raise RuntimeError('Training prohibited during export')
    lgb.train = forbidden
    manifest = read(REFERENCE / 'joint_candidate_manifest.json')
    assert manifest['graph'] == joint_graph()
    assert sha(REFERENCE / 'joint_candidate_forecast.csv') == EXPECTED_FORECAST == manifest['forecast_sha256']
    assert sha(SOURCE / 'model.pkl') == EXPECTED_WEEKLY == manifest['reference_model_sha256']
    assert sha(DAILY / 'refit_model.txt') == manifest['daily_model_sha256']
    assert read(REFERENCE / 'verification.json')['verified']
    for filename, digest in read(DAILY / 'complete.json')['files'].items():
        assert sha(DAILY / filename) == digest, filename
    schema = read(DAILY / 'refit_schema.json')
    assert schema['target_end'] == manifest['label_end'] == '2026-07-26'
    baseline = Predictor(SOURCE).load().model
    model = JointWeeklyDailyModel(baseline, DailyQuantityPart(
        (DAILY / 'refit_model.txt').read_text(encoding='utf-8'), schema))
    model.check_schema()
    model_dir = OUT / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / 'model.pkl')
    config = copy.deepcopy(read(SOURCE / 'config.json'))
    config.update(preprocessing_version=9, joint_features=joint_spec(), model_params=model.params)
    config['packaged_at'] = datetime.now(timezone.utc).isoformat()
    config['validation'] = dict(scheme='reused_january_march_development_windows',
        independent_confirmation=False, origins=manifest['historical_metrics'],
        weighted_ratio=manifest['historical_weighted_ratio'],
        official_score_known=True,
        external_evidence=dict(source='user_leaderboard_screenshot_2026-10-31',
            submission='joint_candidate_forecast', score=0.370968,
            raw_metrics=dict(smape=61.556648, wape=49.142695, rmse=4.157557, bias=87.498300, rmsle=0.544662),
            scorer_file_hash_confirmed=False))
    config['provenance'] = dict(history_cutoff='2026-07-26', packaged_without_training=True,
        winner='joint_candidate_forecast', source_weekly_model_sha256=sha(SOURCE / 'model.pkl'),
        source_daily_model_sha256=sha(DAILY / 'refit_model.txt'),
        source_daily_schema_sha256=sha(DAILY / 'refit_schema.json'),
        source_candidate_manifest_sha256=sha(REFERENCE / 'joint_candidate_manifest.json'),
        expected_forecast_sha256=EXPECTED_FORECAST, component_iterations=model.selected_iterations)
    preprocessor = pickle.loads((SOURCE / 'preprocessor.pkl').read_bytes())
    preprocessor.update(schema_version=9, joint_features=joint_spec())
    (model_dir / 'preprocessor.pkl').write_bytes(pickle.dumps(preprocessor, protocol=4))
    config.update(model_artifact_sha256=sha(model_dir / 'model.pkl'),
                  preprocessor_sha256=sha(model_dir / 'preprocessor.pkl'))
    write(model_dir / 'config.json', config)
    Predictor(model_dir).load()
    write(OUT / 'preparation.json', dict(prepared=True, training_executed=False,
        expected_forecast_sha256=EXPECTED_FORECAST, graph=joint_graph(), model_sha256=sha(model_dir / 'model.pkl')))
    print(json.dumps(read(OUT / 'preparation.json'), indent=2))


if __name__ == '__main__':
    main()
