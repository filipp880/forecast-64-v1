"""Frozen two-candidate temporal event experiment, reusing verified past-only caches.

Development never reads May confirmation data. It replaces only the champion's
probability, preserving its positive magnitude, log-quantity branch and recipe.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score

from app.event_features import INPUT_COLUMNS, FEATURE_COLUMNS, make_event_features
from app.features import CAT_FEATURES, KEY, forecast_end, next_week_start
from app.models.research_model import recipe_prediction
from research_weekly import (GUARDRAILS, assess_candidate, fast_metrics, freeze_record,
                             frozen_record, read, sha, write)

CACHE_ROOT = ROOT / 'artifacts/ideas-20261028-r2'
CHAMPION_ROOT = ROOT / 'artifacts/improvements-20261028-r1'
DEVELOPMENT_ORIGINS = ['2025-10-19', '2026-01-18', '2026-03-22']
JOINT_SELECTION_ORIGINS = ['2026-01-18', '2026-03-22']
CANDIDATES = {
    'event31': {'num_leaves': 31, 'max_depth': 7, 'reg_alpha': 5., 'reg_lambda': 20.},
    'event63': {'num_leaves': 63, 'max_depth': 8, 'reg_alpha': 5., 'reg_lambda': 20.},
}
COMMON_PARAMS = dict(objective='binary', learning_rate=.05, n_estimators=400,
                     min_child_samples=100, subsample=.9, subsample_freq=1,
                     colsample_bytree=.8, random_state=42, n_jobs=6,
                     deterministic=True, force_col_wise=True, verbosity=-1)


def split_names(origin, manifest):
    outer = pd.Timestamp(origin)
    selection = outer - pd.Timedelta(weeks=9)
    names = sorted(n for n in manifest if n.endswith('-True.parquet'))
    inner = [n for n in names if forecast_end(n[:10], 9) <= selection][-16:]
    valid = f'{selection.date()}-True.parquet'
    if valid not in names or not inner:
        raise ValueError('Purged training/validation cache missing')
    # Recover the exact original monthly schedule, excluding the extra inner valid cache.
    refit = [n for n in names if n != valid][-16:]
    return dict(inner=inner, valid=valid, refit=refit,
                test=f'{origin}-False.parquet', inner_origin=str(selection.date()))


def inventory(origin):
    """Verify cache -> original numeric cache -> original complete-record chain."""
    cache_dir = CACHE_ROOT / origin
    old_dir = CHAMPION_ROOT / origin
    manifest = read(cache_dir / 'cache_manifest.json')
    complete = read(old_dir / 'complete.json')
    old_manifest = read(old_dir / 'cache/manifest.json')
    if sha(old_dir / 'cache/manifest.json') != complete['files']['cache/manifest.json']:
        raise ValueError('Original cache manifest changed')
    schedule = split_names(origin, manifest)
    paths = [cache_dir / 'cache_manifest.json', old_dir / 'complete.json',
             old_dir / 'cache/manifest.json']
    for name in sorted(set(schedule['inner'] + schedule['refit'] + [schedule['valid'], schedule['test']])):
        record = manifest[name]
        source = old_dir / 'cache' / name
        target = cache_dir / 'cache' / name
        if record['source_sha256'] != old_manifest[name] or sha(source) != old_manifest[name]:
            raise ValueError('Original cache source changed: ' + name)
        if sha(target) != record['sha256']:
            raise ValueError('Lag cache changed: ' + name)
        paths.extend([source, target])
    for name in ['evaluation.parquet', 'dynamics-values.pkl']:
        path = old_dir / name
        if sha(path) != complete['files'][name]:
            raise ValueError('Frozen champion evaluation changed: ' + name)
        paths.append(path)
    return schedule, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def code_paths():
    return [Path(__file__), ROOT / 'app/event_features.py', ROOT / 'app/features.py',
            ROOT / 'app/enhancements.py', ROOT / 'app/calibration.py',
            ROOT / 'app/models/research_model.py', ROOT / 'app/models/base.py',
            ROOT / 'scripts/research_weekly.py']


def initialize(out):
    if (out / 'protocol.json').exists():
        return verify(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    schedules = {}
    for origin in DEVELOPMENT_ORIGINS:
        schedules[origin], records = inventory(origin)
        hashes.update(records)
    champion_config = read(ROOT / 'model/config.json')
    recipe = champion_config['model_params']['recipe']
    if recipe['source'] != 'dynamics':
        raise ValueError('Expected current dynamics champion')
    for name in ['model.pkl', 'preprocessor.pkl', 'config.json']:
        path = ROOT / 'model' / name
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in code_paths():
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    raw_hashes = {f'data/{n}_dataset_hakaton_train.csv': sha(ROOT / f'data/{n}_dataset_hakaton_train.csv')
                  for n in ['sales', 'stock']}
    protocol = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    candidates=CANDIDATES, common_params=COMMON_PARAMS,
                    development_origins=DEVELOPMENT_ORIGINS,
                    selection_origins=JOINT_SELECTION_ORIGINS,
                    diagnostic_guard_origins=['2025-10-19'], confirmation_origin='2026-05-24',
                    confirmation_status='not_read_or_scored_during_development',
                    selection_rule='joint E1/E2 comparison on Jan/Mar only; October is additional guard',
                    guardrails=GUARDRAILS, champion_recipe=recipe,
                    replacement='classifier probability only; same frozen magnitude, log branch, recipe',
                    features=FEATURE_COLUMNS, schedules=schedules, files=hashes, raw_data_hashes=raw_hashes,
                    cached_history_previously_inspected=True, seed=42, threads=6,
                    memory_limit_gib=4., label='binary quantity > 0',
                    label_boundaries='inner full 9 weeks end <= inner selection; valid/refit labels end <= outer origin')
    freeze_record(out / 'protocol.json', protocol)
    print('EVENT PROTOCOL FROZEN', sha(out / 'protocol.json'), flush=True)
    return protocol


def verify(out):
    protocol = frozen_record(out / 'protocol.json')
    for name, digest in {**protocol['files'], **protocol['raw_data_hashes']}.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen event input/source changed: ' + name)
    if protocol['candidates'] != CANDIDATES or protocol['common_params'] != COMMON_PARAMS:
        raise ValueError('Frozen candidates changed')
    return protocol


def cached(origin, name, labels=True):
    columns = list(dict.fromkeys([*INPUT_COLUMNS, 'week', 'feature_origin', 'target' if labels else 'actual']))
    frame = pd.read_parquet(CACHE_ROOT / origin / 'cache' / name, columns=columns)
    cut = pd.Timestamp(name[:10])
    if not pd.to_datetime(frame.feature_origin).eq(cut).all():
        raise ValueError('Feature origin differs from cache name')
    expected = next_week_start(cut) + pd.to_timedelta((frame.h - 1) * 7, unit='D')
    if not pd.to_datetime(frame.week).eq(expected).all():
        raise ValueError('Target week differs from feature origin/horizon')
    if frame.duplicated(['week', *KEY]).any():
        raise ValueError('Duplicate cache keys')
    return frame


def load_frames(origin, schedule, names, boundary):
    frames = []
    for name in names:
        frame = cached(origin, name)
        if (pd.to_datetime(frame.week) + pd.Timedelta(days=6)).max() > pd.Timestamp(boundary):
            raise ValueError('Training labels exceed permitted boundary')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def encode(frame, categories=None):
    x = make_event_features(frame)
    if categories is None:
        categories = {c: sorted(x[c].astype(str).unique()) for c in CAT_FEATURES}
    for column in CAT_FEATURES:
        x[column] = pd.Categorical(x[column].astype(str), categories=categories[column])
    return x, categories


def fit_one(train, spec, valid=None, iterations=None):
    x, categories = encode(train)
    params = {**COMMON_PARAMS, **spec}
    if iterations is not None:
        params['n_estimators'] = int(iterations)
    model = lgb.LGBMClassifier(**params)
    def memory_budget_callback(environment):
        if psutil.Process().memory_info().rss > 4 * 2**30:
            raise MemoryError('Event fitting exceeded frozen 4 GiB process budget')

    kwargs = dict(callbacks=[memory_budget_callback])
    if valid is not None:
        xv, _ = encode(valid, categories)
        kwargs.update(eval_X=xv, eval_y=(valid.target.to_numpy(float) > 0).astype('int8'),
                      eval_metric='binary_logloss',
                      callbacks=[memory_budget_callback, lgb.early_stopping(40, verbose=False)])
    model.fit(x, (train.target.to_numpy(float) > 0).astype('int8'),
              categorical_feature=CAT_FEATURES, **kwargs)
    return model.booster_, categories


def event_diagnostics(frame, y, probability, prediction):
    actual = y > 0
    predicted = prediction > 0
    mismatch = actual != predicted
    weights = np.ones(len(y)) / len(y)  # every pair has exactly nine target weeks
    denom = np.maximum(y + prediction, 1e-20)
    loss = 200 * np.abs(y - prediction) / denom
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    return dict(brier=float(np.mean((probability - actual) ** 2)),
                logloss=float(-np.mean(actual*np.log(np.clip(probability, 1e-12, 1)) +
                                      (~actual)*np.log(np.clip(1-probability, 1e-12, 1)))),
                auc=float(roc_auc_score(actual, probability)) if actual.any() and not actual.all() else None,
                event_mismatch_rate=float(mismatch.mean()), precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                false_zero_pp=float((loss * weights * (actual & ~predicted)).sum()),
                false_positive_pp=float((loss * weights * (~actual & predicted)).sum()),
                positive_quantity_pp=float((loss * weights * (actual & predicted)).sum()))


def evaluate(origin, frame, probabilities, recipe):
    old = pd.read_parquet(CHAMPION_ROOT / origin / 'evaluation.parquet')
    key_columns = ['week', *KEY]
    if not frame[key_columns].reset_index(drop=True).equals(old[key_columns].reset_index(drop=True)):
        # Category-vs-string dtype differences do not alter key identity.
        left = frame[key_columns].astype(str).reset_index(drop=True)
        right = old[key_columns].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError('Event predictions do not align with frozen champion rows')
    y = old.actual.to_numpy(float)
    if not np.array_equal(y, frame.actual.to_numpy(float)):
        raise ValueError('Cached truth differs from champion truth')
    values = pickle.loads((CHAMPION_ROOT / origin / 'dynamics-values.pkl').read_bytes())
    champion = recipe_prediction({'dynamics': values}, old, recipe)
    candidate_values = {**values, 'probability': probabilities}
    prediction = recipe_prediction({'dynamics': candidate_values}, old, recipe)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(old[KEY]))
    report = dict(champion=fast_metrics(y, champion, codes), candidate=fast_metrics(y, prediction, codes),
                  champion_events=event_diagnostics(old, y, values['probability'], champion),
                  candidate_events=event_diagnostics(old, y, probabilities, prediction), horizons={}, segments={})
    masks = {f'h{h}': old.h.eq(h) for h in range(1, 10)}
    masks.update({'past_0.5_to_2': old.mean_28.mul(7).between(.5, 2, inclusive='left'),
                  'rare': old.zero_week_share_13.ge(.6), 'regular': old.zero_week_share_13.lt(.3)})
    for name, mask in masks.items():
        if not mask.any():
            continue
        subcodes, _ = pd.factorize(pd.MultiIndex.from_frame(old.loc[mask, KEY]))
        record = dict(rows=int(mask.sum()), champion=fast_metrics(y[mask], champion[mask], subcodes),
                      candidate=fast_metrics(y[mask], prediction[mask], subcodes),
                      events=event_diagnostics(old.loc[mask], y[mask], probabilities[mask], prediction[mask]))
        report['horizons' if name.startswith('h') else 'segments'][name] = record
    return report


class MemoryMonitor:
    def __init__(self, limit_gib=4.):
        self.limit = limit_gib * 2**30
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
            self.stop.wait(.2)

    def check(self):
        if psutil.Process().memory_info().rss > self.limit:
            raise MemoryError('Event experiment memory budget exceeded')


def development(out, protocol):
    if (out / 'development.json').exists():
        return frozen_record(out / 'development.json')
    started = time.perf_counter()
    monitor = MemoryMonitor(protocol['memory_limit_gib'])
    monitor.thread.start()
    results = {name: {} for name in protocol['candidates']}
    try:
        for origin in protocol['development_origins']:
            schedule = protocol['schedules'][origin]
            directory = out / origin
            directory.mkdir(exist_ok=True)
            train = load_frames(origin, schedule, schedule['inner'], schedule['inner_origin'])
            valid = load_frames(origin, schedule, [schedule['valid']], origin)
            monitor.check()
            iterations = {}
            for name, spec in protocol['candidates'].items():
                print('EVENT INNER', origin, name, len(train), flush=True)
                booster, _ = fit_one(train, spec, valid)
                iterations[name] = booster.current_iteration()
                del booster
                gc.collect()
                monitor.check()
            del train, valid
            gc.collect()
            refit = load_frames(origin, schedule, schedule['refit'], origin)
            test = cached(origin, schedule['test'], labels=False)
            monitor.check()
            for name, spec in protocol['candidates'].items():
                print('EVENT REFIT', origin, name, len(refit), iterations[name], flush=True)
                booster, categories = fit_one(refit, spec, iterations=iterations[name])
                xtest, _ = encode(test, categories)
                probability = booster.predict(xtest, num_threads=6)
                booster.save_model(str(directory / f'{name}.txt'))
                write(directory / f'{name}.categories.json', categories)
                np.save(directory / f'{name}.probability.npy', probability)
                report = evaluate(origin, test, probability, protocol['champion_recipe'])
                report.update(iterations=iterations[name], inner_origin=schedule['inner_origin'],
                              train_rows=len(refit), feature_count=len(FEATURE_COLUMNS))
                freeze_record(directory / f'{name}.json', report)
                results[name][origin] = report
                print('EVENT METRICS', origin, name, json.dumps(report['candidate']), flush=True)
                del booster, xtest, probability
                gc.collect()
                monitor.check()
            del refit, test
            gc.collect()
        assessments = {}
        for name in results:
            assessment = assess_candidate(
                {o: results[name][o]['candidate'] for o in protocol['selection_origins']},
                {o: results[name][o]['champion'] for o in protocol['selection_origins']}, protocol['guardrails'])
            diagnostic = {o: assess_candidate({o: results[name][o]['candidate']},
                                              {o: results[name][o]['champion']}, protocol['guardrails'])
                          for o in protocol['diagnostic_guard_origins']}
            # Diagnostic origins do not require improvement, only per-origin loss limits.
            diagnostic_ok = all(
                all(v <= (protocol['guardrails']['per_origin_smape_ratio_max'] if k == 'smape'
                          else protocol['guardrails']['per_origin_other_metric_ratio_max'])
                    for k, v in record['metric_ratios'].items()) for record in diagnostic.values())
            assessments[name] = dict(selection=assessment, diagnostic=diagnostic,
                                     eligible=assessment['eligible'] and diagnostic_ok)
        summary = dict(protocol_sha256=sha(out / 'protocol.json'), results=results,
                       assessments=assessments, confirmation_executed=False,
                       selected_recipe=None, selection_pending_root_joint_E1_E2=True,
                       seconds=time.perf_counter()-started,
                       peak_sampled_rss_gib=monitor.peak / 2**30,
                       memory_sampling_seconds=.2)
        freeze_record(out / 'development.json', summary)
        return summary
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/events-20261029')
    parser.add_argument('--stage', choices=['freeze', 'development'], required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    protocol = initialize(out)
    if args.stage == 'development':
        summary = development(out, protocol)
        print(json.dumps(summary['assessments'], indent=2), flush=True)
    verify(out)


if __name__ == '__main__':
    main()
