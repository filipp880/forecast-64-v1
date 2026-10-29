"""Frozen soft reconciliation to independent past-only rolling aggregate forecasts.

This is a small E5 baseline, not learned hierarchical forecasting. Current zero
support is fixed. No May facts, region assumptions or future aggregate totals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.features import KEY
from app.models.research_model import recipe_prediction
from scripts.research_weekly import GUARDRAILS, assess_candidate, fast_metrics, freeze_record, frozen_record, read, sha

ORIGINS = ['2026-01-18', '2026-03-22']
PENALTIES = [.05, .2]
SOURCE = ROOT / 'artifacts/improvements-20261028-r1'


def soft_reconcile(base, history, article, warehouse, penalty, max_iterations=2000):
    """Projected gradient in a diagonal metric; zeros never acquire tiny sales."""
    base, history = np.asarray(base, float), np.asarray(history, float)
    if not np.isfinite(np.r_[base, history]).all() or min(base.min(), history.min()) < 0:
        raise ValueError('Invalid quantity')
    ac, _ = pd.factorize(article, sort=False)
    wc, _ = pd.factorize(warehouse, sort=False)
    at = np.bincount(ac, weights=history)
    wt = np.bincount(wc, weights=history)
    support = base > 0
    weight = np.where(support, base + 1., 0.)
    # Sum of the two normalized block spectral bounds plus the fidelity term.
    lipschitz = 1 + penalty * (np.max(np.bincount(ac, weights=weight)/(at+1))
                              + np.max(np.bincount(wc, weights=weight)/(wt+1)))

    def objective(x):
        ar, wr = np.bincount(ac, weights=x)-at, np.bincount(wc, weights=x)-wt
        value = .5*np.sum((x-base)**2/(base+1))
        value += .5*penalty*(np.sum(ar**2/(at+1))+np.sum(wr**2/(wt+1)))
        return float(value), ar, wr

    x = base.copy()
    before = objective(x)[0]
    for iteration in range(max_iterations):
        loss, ar, wr = objective(x)
        grad = (x-base)/(base+1) + penalty*(ar[ac]/(at[ac]+1)+wr[wc]/(wt[wc]+1))
        updated = np.where(support, np.maximum(x-weight*grad/lipschitz, 1e-9), 0.)
        if np.max(np.abs(updated-x)) <= 1e-7*(1+np.max(x)):
            x = updated
            break
        x = updated
    after, ar, wr = objective(x)
    if after > before + 1e-7*max(before, 1):
        raise ValueError('Reconciliation objective increased')
    return x, dict(iterations=iteration+1, objective_before=before, objective_after=after,
                   article_absolute_residual=float(np.abs(ar).sum()),
                   warehouse_absolute_residual=float(np.abs(wr).sum()),
                   zero_mask_preserved=bool(np.array_equal(x>0, support)))


def initialize(out):
    if (out/'protocol.json').exists():
        protocol = frozen_record(out/'protocol.json')
        for name, expected in protocol['files'].items():
            if sha(ROOT/name) != expected:
                raise ValueError('Frozen reconciliation source changed: '+name)
        return protocol
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty run directory')
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT/'scripts/research_weekly.py', ROOT/'app/features.py',
             ROOT/'app/models/research_model.py', ROOT/'app/calibration.py', ROOT/'model/config.json']
    for origin in ORIGINS:
        directory = SOURCE/origin
        complete = read(directory/'complete.json')
        paths.append(directory/'complete.json')
        for name in ['evaluation.parquet', 'dynamics-values.pkl']:
            path = directory/name
            if sha(path) != complete['files'][name]:
                raise ValueError('OOF cache changed: '+str(path))
            paths.append(path)
    protocol = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_origins=ORIGINS, penalties=PENALTIES, max_iterations=2000,
        targets='article/warehouse sums of past mean_week_4, same known-pair universe',
        objective='weighted bottom fidelity + normalized soft aggregate squared errors',
        support='champion positive support fixed; continuous quantities allowed',
        champion_recipe=read(ROOT/'model/config.json')['model_params']['recipe'],
        guardrails=GUARDRAILS, confirmation_read=False, prior_history_inspected=True,
        limitation='Naive upper-level baseline; no independently trained aggregate models or exact reconciliation',
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    freeze_record(out/'protocol.json', protocol)
    print('RECONCILIATION PROTOCOL FROZEN', sha(out/'protocol.json'), flush=True)
    return protocol


def development(out, protocol):
    if (out/'development.json').exists():
        return frozen_record(out/'development.json')
    results = {str(p): {} for p in PENALTIES}
    champions = {}
    for origin in ORIGINS:
        frame = pd.read_parquet(SOURCE/origin/'evaluation.parquet')
        values = pickle.loads((SOURCE/origin/'dynamics-values.pkl').read_bytes())
        base = recipe_prediction({'dynamics':values}, frame, protocol['champion_recipe'])
        y = frame.actual.to_numpy(float)
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))
        champions[origin] = fast_metrics(y, base, codes)
        for penalty in PENALTIES:
            prediction = base.copy()
            diagnostics = {}
            for week, idx in frame.groupby('week', sort=True).groups.items():
                idx = np.asarray(idx)
                selected = frame.loc[idx]
                prediction[idx], report = soft_reconcile(base[idx], selected.mean_week_4,
                    selected.article, selected.warehouse_name, penalty)
                diagnostics[str(week)] = report
            metrics = fast_metrics(y, prediction, codes)
            results[str(penalty)][origin] = dict(metrics=metrics, diagnostics=diagnostics)
            print('RECONCILIATION METRICS', origin, penalty, json.dumps(metrics), flush=True)
    assessments = {name: assess_candidate({o:r['metrics'] for o,r in reports.items()}, champions,
                                          protocol['guardrails']) for name,reports in results.items()}
    result = dict(protocol_sha256=sha(out/'protocol.json'), results=results, champion=champions,
                  assessments=assessments, confirmation_read=False, model_training_executed=False)
    freeze_record(out/'development.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/reconciliation-20261029')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    protocol = initialize(out)
    result = development(out, protocol)
    initialize(out)
    print(json.dumps(result['assessments'], indent=2))


if __name__ == '__main__':
    main()
