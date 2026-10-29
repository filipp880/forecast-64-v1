"""Frozen regular-demand conditional magnitude calibration on chronological OOF.

October warms calibration, January/March evaluate. The current final event mask
and all nonregular quantities remain unchanged. May is never read by this CLI.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.features import KEY
from app.models.research_model import recipe_prediction
from research_weekly import GUARDRAILS, assess_candidate, fast_metrics, freeze_record, frozen_record, read, sha

SOURCE = ROOT / 'artifacts/improvements-20261028-r1'
WARMUP = '2025-10-19'
EVALUATION = ['2026-01-18', '2026-03-22']
GRID = np.array([.6, .8, 1., 1.2, 1.4])
BINS = [0., 2., 10., None]  # None is +infinity in a strict JSON protocol.
SHRINKAGE_SUPPORT = 200.
MIN_BIN_SUPPORT = 50
CANDIDATES = ['median_ratio', 'smape_ratio']


def demand_bin(frame):
    weekly = frame.mean_28.to_numpy(float) * 7
    return np.searchsorted(np.array([2., 10.]), weekly, side='right')


def source_inventory(origin):
    directory = SOURCE / origin
    complete = read(directory / 'complete.json')
    files = {directory / 'complete.json': sha(directory / 'complete.json')}
    for name in ['evaluation.parquet', 'dynamics-values.pkl']:
        path = directory / name
        if sha(path) != complete['files'][name]:
            raise ValueError('Frozen OOF input changed: ' + str(path))
        files[path] = sha(path)
    return {path.relative_to(ROOT).as_posix(): digest for path, digest in files.items()}


def initialize(out):
    if (out / 'protocol.json').exists():
        return verify(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose empty run directory')
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for origin in [WARMUP, *EVALUATION]:
        hashes.update(source_inventory(origin))
    config = read(ROOT / 'model/config.json')
    if config.get('model_type') != 'research' or config['model_params']['recipe']['source'] != 'dynamics':
        raise ValueError('Expected current research dynamics champion')
    paths = [Path(__file__), ROOT / 'app/models/research_model.py', ROOT / 'app/calibration.py',
             ROOT / 'app/models/base.py', ROOT / 'scripts/research_weekly.py',
             *[ROOT / 'model' / name for name in ['model.pkl', 'preprocessor.pkl', 'config.json']]]
    hashes.update({path.relative_to(ROOT).as_posix(): sha(path) for path in paths})
    protocol = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    warmup_origin=WARMUP, evaluation_origins=EVALUATION,
                    calibration_origins={EVALUATION[0]: [WARMUP], EVALUATION[1]: [WARMUP, EVALUATION[0]]},
                    candidates=CANDIDATES, ratio_grid=GRID.tolist(), weekly_demand_bins=BINS,
                    shrinkage_support=SHRINKAGE_SUPPORT, minimum_bin_support=MIN_BIN_SUPPORT,
                    regular_definition='past n_sale_days_90 >= 45',
                    calibration_scope='regular & current champion prediction>0 & actual>0 & magnitude>0',
                    prediction='q=Poisson_magnitude*calibrated_ratio; fixed champion zero mask; other values unchanged',
                    median_definition='median actual/magnitude ratio; bin ratio shrunk to global median',
                    smape_definition='fixed-grid conditional-positive expected SMAPE; bin risks shrunk to global risks',
                    risk_tie_policy='prefer ratio nearest1, then smaller ratio',
                    guardrails=GUARDRAILS, champion_recipe=config['model_params']['recipe'],
                    files=hashes, confirmation_origin='2026-05-24', confirmation_read=False,
                    previously_inspected_history=True, model_training_executed=False)
    freeze_record(out / 'protocol.json', protocol)
    print('MAGNITUDE PROTOCOL FROZEN', sha(out / 'protocol.json'), flush=True)
    return protocol


def verify(out):
    protocol = frozen_record(out / 'protocol.json')
    for name, digest in protocol['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen magnitude source changed: ' + name)
    return protocol


def load_origin(origin, recipe):
    frame = pd.read_parquet(SOURCE / origin / 'evaluation.parquet')
    values = pickle.loads((SOURCE / origin / 'dynamics-values.pkl').read_bytes())
    if frame.duplicated(['week', *KEY]).any() or not frame.groupby(KEY, observed=True).size().eq(9).all():
        raise ValueError('Expected nine unique weeks per pair')
    frame = frame.copy()
    frame['champion_prediction'] = recipe_prediction({'dynamics': values}, frame, recipe)
    frame['poisson_magnitude'] = values['magnitude']
    frame['oof_origin'] = pd.Timestamp(origin)
    frame['label_end'] = frame.week.max() + pd.Timedelta(days=6)
    return frame


def calibration_rows(frames, evaluation_origin):
    """Entire calibration target horizons must have ended by evaluation origin."""
    evaluation_origin = pd.Timestamp(evaluation_origin)
    permitted = []
    for frame in frames:
        if pd.to_datetime(frame.label_end).max() > evaluation_origin:
            raise ValueError('Calibration labels have not ended by evaluation origin')
        mask = (frame.n_sale_days_90.ge(45) & frame.champion_prediction.gt(0)
                & frame.actual.gt(0) & frame.poisson_magnitude.gt(0))
        selected = frame.loc[mask].copy()
        if not np.isfinite(selected[['actual', 'poisson_magnitude']].to_numpy(float)).all():
            raise ValueError('Nonfinite conditional magnitude calibration')
        permitted.append(selected)
    result = pd.concat(permitted, ignore_index=True)
    if result.empty:
        raise ValueError('No conditional-positive regular calibration rows')
    return result


def choose_risk_ratio(risks):
    minimum = float(np.min(risks))
    tied = np.flatnonzero(np.isclose(risks, minimum, rtol=0, atol=1e-12))
    index = min(tied, key=lambda i: (abs(float(GRID[i]) - 1), float(GRID[i])))
    return float(GRID[index])


def fit_calibration(frame):
    y = frame.actual.to_numpy(float)
    magnitude = frame.poisson_magnitude.to_numpy(float)
    labels = demand_bin(frame)
    ratio = y / magnitude
    global_median = float(np.median(ratio))
    q = magnitude[:, None] * GRID[None, :]
    losses = 200 * np.abs(y[:, None] - q) / (y[:, None] + q)
    global_risks = losses.mean(axis=0)
    bins = {}
    for label in range(3):
        mask = labels == label
        count = int(mask.sum())
        weight = count / (count + SHRINKAGE_SUPPORT) if count >= MIN_BIN_SUPPORT else 0.
        median = float(np.median(ratio[mask])) if count else global_median
        risks = losses[mask].mean(axis=0) if count else global_risks
        shrunk_risks = weight * risks + (1 - weight) * global_risks
        bins[str(label)] = dict(support=count, weight=weight,
                               median_ratio=weight*median+(1-weight)*global_median,
                               smape_ratio=choose_risk_ratio(shrunk_risks),
                               shrunk_conditional_risks=shrunk_risks.tolist())
    return dict(rows=len(frame), global_median_ratio=global_median,
                global_risks=global_risks.tolist(), bins=bins)


def apply_calibration(frame, calibration, candidate):
    prediction = frame.champion_prediction.to_numpy(float).copy()
    scope = frame.n_sale_days_90.ge(45).to_numpy() & (prediction > 0)
    labels = demand_bin(frame)
    ratios = np.array([calibration['bins'][str(label)][candidate] for label in labels])
    prediction[scope] = frame.poisson_magnitude.to_numpy(float)[scope] * ratios[scope]
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid calibrated magnitude')
    if not np.array_equal(prediction > 0, frame.champion_prediction.to_numpy(float) > 0):
        raise ValueError('Magnitude calibration changed champion mask')
    return prediction


def evaluate(frame, prediction):
    y = frame.actual.to_numpy(float)
    champion = frame.champion_prediction.to_numpy(float)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))
    regular = frame.n_sale_days_90.ge(45).to_numpy()
    tp = (y > 0) & (champion > 0)
    loss = lambda q: 200 * np.abs(y-q) / np.maximum(y+q, 1e-20)
    record = dict(champion=fast_metrics(y, champion, codes), candidate=fast_metrics(y, prediction, codes),
                  mask_preserved=bool(np.array_equal(prediction>0, champion>0)),
                  nonregular_preserved=bool(np.array_equal(prediction[~regular], champion[~regular])),
                  changed_rows=int(np.count_nonzero(prediction != champion)),
                  conditional_positive_regular_smape=dict(
                      champion=float(loss(champion)[regular & tp].mean()),
                      candidate=float(loss(prediction)[regular & tp].mean())),
                  positive_quantity_global_contribution_pp=dict(
                      champion=float((loss(champion)*tp).mean()),
                      candidate=float((loss(prediction)*tp).mean())), segments={})
    for name, mask in [('regular', regular), ('nonregular', ~regular),
                       *[(f'past_bin_{i}', demand_bin(frame)==i) for i in range(3)]]:
        if mask.any():
            subcodes, _ = pd.factorize(pd.MultiIndex.from_frame(frame.loc[mask, KEY]))
            record['segments'][name] = dict(rows=int(mask.sum()),
                champion=fast_metrics(y[mask], champion[mask], subcodes),
                candidate=fast_metrics(y[mask], prediction[mask], subcodes))
    return record


def development(out, protocol):
    if (out / 'development.json').exists():
        return frozen_record(out / 'development.json')
    started = time.perf_counter()
    past = {}
    results = {name: {} for name in CANDIDATES}
    for origin in protocol['evaluation_origins']:
        for historical in protocol['calibration_origins'][origin]:
            if historical not in past:
                past[historical] = load_origin(historical, protocol['champion_recipe'])
        allowed = calibration_rows([past[o] for o in protocol['calibration_origins'][origin]], origin)
        calibration = fit_calibration(allowed)
        calibration.update(evaluation_origin=origin, calibration_origins=protocol['calibration_origins'][origin],
                           maximum_calibration_label_end=str(allowed.label_end.max().date()))
        directory = out / origin
        directory.mkdir(exist_ok=True)
        freeze_record(directory / 'calibration.json', calibration)
        frame = load_origin(origin, protocol['champion_recipe'])
        for candidate in CANDIDATES:
            prediction = apply_calibration(frame, calibration, candidate)
            np.save(directory / f'{candidate}.prediction.npy', prediction)
            report = evaluate(frame, prediction)
            results[candidate][origin] = report
            freeze_record(directory / f'{candidate}.json', report)
            print('MAGNITUDE METRICS', origin, candidate, json.dumps(report['candidate']), flush=True)
        # January labels become available for March only through calibration_rows checks.
        past[origin] = frame
    assessments = {name: assess_candidate(
        {o: results[name][o]['candidate'] for o in EVALUATION},
        {o: results[name][o]['champion'] for o in EVALUATION}, protocol['guardrails']) for name in CANDIDATES}
    summary = dict(protocol_sha256=sha(out/'protocol.json'), results=results, assessments=assessments,
                   seconds=time.perf_counter()-started, confirmation_read=False,
                   model_training_executed=False, selection_pending_root_joint_E1_E2_E3=True)
    freeze_record(out / 'development.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/magnitude-20261029')
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
