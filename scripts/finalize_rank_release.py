"""Refit the saved weekly rank winner; preserve the production release until checked."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.csv_io import read_csv_checked
from app.features import KEY
from app.models.metric_experts import MetricExpertPart
from app.models.research_model import WeeklyResearchModel, fit_member, recipe_member_names
from app.predictor import Predictor
from scripts.confirm_smape_wape import expert_specs
from scripts.finalize_smape_wape_cv import CACHE, WeeklyLagHistory
from scripts.research_weekly import freeze_record, frozen_record, read, sha
from train import save_artifact

REVIEW = ROOT / 'artifacts/score-review-20261029'


def initialize(out):
    selection = read(REVIEW / 'SELECTED_CONFIGURATION.json')
    comparison = read(REVIEW / 'comparison.json')
    winner = comparison['jan_mar']['ranking'][0]
    if winner['id'] != selection['chosen_id'] or winner['recipe'] != selection['recipe']:
        raise ValueError('Selection differs from the saved rank winner')
    if (out / 'protocol.json').exists():
        protocol = frozen_record(out / 'protocol.json')
        for name, digest in protocol['files'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Frozen refit input changed: ' + name)
        return selection, protocol
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty refit directory')
    manifest = read(CACHE / 'manifest.json')
    paths = [Path(__file__), REVIEW / 'SELECTED_CONFIGURATION.json', REVIEW / 'comparison.json',
             CACHE / 'manifest.json', ROOT / 'scripts/finalize_smape_wape_cv.py',
             ROOT / 'app/models/research_model.py', ROOT / 'app/models/metric_experts.py',
             ROOT / 'app/sequence_features.py', ROOT / 'train.py',
             ROOT / 'data/sales_dataset_hakaton_train.csv', ROOT / 'data/stock_dataset_hakaton_train.csv',
             *[ROOT / 'model' / n for n in ['model.pkl', 'preprocessor.pkl', 'config.json']]]
    for name, digest in manifest.items():
        path = CACHE / name
        if sha(path) != digest:
            raise ValueError('Final training cache changed: ' + name)
        paths.append(path)
    legacy_iterations = ROOT / 'artifacts/improvements-20261028-r1/2026-03-22/iterations.json'
    expert_iterations = ROOT / 'artifacts/smape-wape-confirmation-20261029/iterations.json'
    paths.extend([legacy_iterations, expert_iterations])
    # The latest saved past-only schedule for legacy MAE; no new tuning.
    iterations = {'mae': read(legacy_iterations)['mae'], **frozen_record(expert_iterations)}
    out.mkdir(parents=True)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(), chosen_id=selection['chosen_id'],
        recipe=selection['recipe'], expert_specs=expert_specs(selection['recipe']), iterations=iterations,
        history_cutoff='2026-07-26', training_cache=sorted(manifest),
        evaluation_kind='retrospective_local_rank_selection_on_january_march_NOT_independent_confirmation',
        independent_confirmation=False, model_fit_strategy='reuse identical full-history dynamics/count; fit missing MAE experts on all sixteen permitted cached origins',
        iteration_policy='fixed saved schedules: legacy MAE from latest available March fit; recent MAE from May inner early stopping',
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    freeze_record(out / 'protocol.json', protocol)
    return selection, protocol


def finalize(out, selection, protocol):
    if (out / 'final_training.json').exists():
        return frozen_record(out / 'final_training.json')
    model_dir = out / 'model'
    if model_dir.exists():
        raise FileExistsError('Preserve partial model; choose a new directory')
    sales = read_csv_checked(ROOT / 'data/sales_dataset_hakaton_train.csv',
        usecols=['date', *KEY, 'quantity'], parse_dates=['date'], dtype={'quantity': 'float32'})
    history = WeeklyLagHistory(sales.loc[sales.date <= protocol['history_cutoff']])
    del sales
    gc.collect()
    frames = []
    for name in protocol['training_cache']:
        frame = pd.read_parquet(CACHE / name)
        if (pd.to_datetime(frame.week) + pd.Timedelta(days=6)).max() > pd.Timestamp(protocol['history_cutoff']):
            raise ValueError('Target exceeds authorized training cutoff')
        frames.append(history.add(frame, name[:10]))
    del history
    train = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    if not np.isfinite(train.target).all() or (train.target < 0).any():
        raise ValueError('Invalid training labels')
    base = Predictor(ROOT / 'model').load().model
    model = WeeklyResearchModel(dict(recipe=protocol['recipe'], expert_specs=protocol['expert_specs'],
                                    threads=6, estimators=600, seed=42))
    fitted, reused = [], []
    for name in sorted(recipe_member_names(protocol['recipe'])):
        if name in base.members:
            if name in protocol['expert_specs'] and base.members[name].spec != protocol['expert_specs'][name]:
                raise ValueError('Existing expert specification differs')
            model.members[name] = base.members[name]
            reused.append(name)
            continue
        print('FULL ALLOWED-HISTORY REFIT', name, len(train), protocol['iterations'][name], flush=True)
        if name in protocol['expert_specs']:
            model.members[name] = MetricExpertPart(protocol['expert_specs'][name], 6).fit(
                train, iterations=protocol['iterations'][name])
        else:
            model.members[name] = fit_member(name, train, iterations=protocol['iterations'][name], threads=6)
        fitted.append(name)
        gc.collect()
    model.check_schema()
    counts = {}
    for name, member in model.members.items():
        if name == 'dynamics':
            counts[name] = dict(classifier=member.hurdle.classifier.num_trees(),
                                regressor=member.hurdle.regressor.num_trees(), lgbm=member.lgbm.booster.num_trees())
        else:
            counts[name] = dict(iterations=member.booster.current_iteration(), trees=member.booster.num_trees())
    rows = len(train)
    del train
    gc.collect()
    validation = dict(scheme=protocol['evaluation_kind'], independent_confirmation=False,
        selection_used_all_reported_origins=True, origins=selection['metrics_by_origin'],
        local_rank_score=selection['local_rank_score'], local_rank_candidates=selection['unique_outcome_configurations'],
        official_score_known=False, metrics=selection['metrics_by_origin']['2026-03-22'])
    config = save_artifact(model, model_dir, '2026-07-26', '2026-07-26', rows,
        len(protocol['training_cache']), validation=validation,
        provenance=dict(rank_selection_sha256=sha(REVIEW / 'SELECTED_CONFIGURATION.json'),
                        finalization_protocol_sha256=sha(out / 'protocol.json'), history_cutoff='2026-07-26'))
    Predictor(model_dir).load()
    result = dict(rows=rows, cutoffs=len(protocol['training_cache']), history_cutoff='2026-07-26',
        fitted_members=fitted, reused_identical_full_history_members=reused, actual_tree_counts=counts,
        model_sha256=config['model_artifact_sha256'], preprocessor_sha256=config['preprocessor_sha256'],
        config_sha256=sha(model_dir / 'config.json'), production_replaced=False,
        independent_confirmation=False, evaluation_kind=protocol['evaluation_kind'])
    freeze_record(out / 'final_training.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/rank-winner-final-20261029')
    out = parser.parse_args().out.resolve()
    selection, protocol = initialize(out)
    print(finalize(out, selection, protocol), flush=True)
    initialize(out)


if __name__ == '__main__':
    main()
