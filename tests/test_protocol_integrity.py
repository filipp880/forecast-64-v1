"""Regressions for comparator, acceptance limits and committed research stages."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_weekly as research


CHAMPION = {'source': 'dynamics', 'post': {'medium_gate': .5, 'regular_scale': 1.1,
                                         'round_int': True, 'round_threshold': .5}}


def metrics(value=100., **overrides):
    return {**{name: value for name in research.NAMES}, **overrides}


def replace_record(path, value):
    """Recompute digest as well, to exercise semantic links independently."""
    research.write(path, value)
    path.with_suffix('.sha256').write_text(research.sha(path), encoding='ascii')


@pytest.fixture
def run(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    protocol = {'champion': {'recipe': copy.deepcopy(CHAMPION)},
                'champion_grid_key': 'champion', 'guardrails': dict(research.GUARDRAILS),
                'confirmation_origin': '2026-05-24', 'history_cutoff': '2026-07-26',
                'final_training_end': '2026-07-26'}
    research.freeze_record(out / 'protocol.json', protocol)
    grid = {'champion': CHAMPION, 'candidate': {'source': 'poisson', 'post': {'round_int': True}}}
    research.write(out / 'candidate_grid.json', grid)
    selection = {'selected_id': 'candidate', 'selected_recipe': grid['candidate'],
                 'champion_recipe': copy.deepcopy(CHAMPION),
                 'protocol_sha256': research.sha(out / 'protocol.json'),
                 'grid_sha256': research.sha(out / 'candidate_grid.json')}
    research.freeze_record(out / 'selection.json', selection)
    decision = {'selected_recipe': grid['candidate'], 'champion_recipe': copy.deepcopy(CHAMPION),
                'protocol_sha256': research.sha(out / 'protocol.json'),
                'selection_sha256': research.sha(out / 'selection.json'),
                'confirmation_origin': protocol['confirmation_origin'],
                'candidate': metrics(90.), 'baseline': metrics(), 'promote': True}
    research.freeze_record(out / 'decision.json', decision)
    return out, protocol


def test_initialization_freezes_current_champion_recipe_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    code = tmp_path / 'source.py'
    code.write_text('# source\n', encoding='utf-8')
    monkeypatch.setattr(research, 'source_paths', lambda: [code])
    model = tmp_path / 'model'
    model.mkdir()
    for name in ['model.pkl', 'preprocessor.pkl']:
        (model / name).write_bytes(name.encode())
    config = {'model_type': 'research', 'model_params': {'recipe': CHAMPION},
              'model_file': 'model.pkl', 'preprocessor_file': 'preprocessor.pkl',
              'model_artifact_sha256': research.sha(model / 'model.pkl'),
              'preprocessor_sha256': research.sha(model / 'preprocessor.pkl')}
    research.write(model / 'config.json', config)
    (tmp_path / 'data').mkdir()
    for name in ['sales', 'stock']:
        (tmp_path / f'data/{name}_dataset_hakaton_train.csv').write_text('quantity\n1\n')
    monkeypatch.setattr(research, 'candidate_grid', lambda: {'base': {'source': 'base'}})
    out = tmp_path / 'new-run'
    protocol = research.initialize(out)
    assert protocol['research_version'] == 2
    assert protocol['champion']['recipe'] == CHAMPION
    assert research.read(out / 'candidate_grid.json')[protocol['champion_grid_key']] == CHAMPION
    assert research.verify_protocol(out) == protocol
    (model / 'model.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='champion artifact changed'):
        research.verify_protocol(out)


def test_evaluate_recipes_uses_dynamics_champion_not_legacy_base():
    frame = pd.DataFrame({'warehouse_name': ['w', 'w'], 'article': ['a', 'b'],
                          'actual': [2., 4.], 'mean_28': [2 / 7, 4 / 7],
                          'n_sale_days_90': [10., 10.]})
    values = {'dynamics': {'probability': np.ones(2), 'magnitude': np.array([2., 4.]),
                           'lgbm_raw': np.array([2., 4.])},
              'base': {'raw': np.zeros(2)}}
    champion, candidates = research.evaluate_recipes(
        frame, values, {'legacy': {'source': 'base'}}, CHAMPION)
    assert champion['smape'] == 0
    assert candidates['legacy']['metrics']['smape'] == 200


def test_guardrails_reject_single_metric_harm_hidden_by_weighted_gain():
    # This passed the old score <= .995 and >=3 improving metrics rule.
    candidate = metrics(smape=99., wape=99., rmse=110., bias=95., rmsle=80.)
    assert research.ratio(candidate, metrics()) < .995
    result = research.assess_candidate({'a': candidate}, {'a': metrics()})
    assert result['improved_metrics'] == 4
    assert not result['eligible']


def test_guardrails_check_each_origin_and_mean_smape():
    candidates = {'a': metrics(90.), 'b': metrics(90., smape=104.)}
    champion = {'a': metrics(), 'b': metrics()}
    assert not research.assess_candidate(candidates, champion)['eligible']
    assert not research.assess_candidate({'a': metrics(90., smape=100.1)}, {'a': metrics()})['eligible']
    assert research.assess_candidate({'a': metrics(99.)}, {'a': metrics()})['eligible']
    assert research.ratio(metrics(0.), metrics(0.)) == pytest.approx(1.)


def test_memory_report_without_windows_fields(monkeypatch):
    monkeypatch.setattr(research.psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=2**30)))
    assert research.memory_report() == {'process_memory_gib': 1., 'memory_kind': 'current_rss'}


@pytest.mark.parametrize('field', ['selected_recipe', 'protocol_sha256', 'selection_sha256',
                                   'confirmation_origin', 'champion_recipe', 'promote'])
def test_finalization_rejects_semantically_changed_decision_before_loading_data(run, monkeypatch, field):
    out, protocol = run
    decision = research.read(out / 'decision.json')
    decision[field] = False if field == 'promote' else 'changed'
    replace_record(out / 'decision.json', decision)
    monkeypatch.setattr(research, 'load_data', lambda: pytest.fail('fit/read must not start'))
    with pytest.raises(ValueError):
        research.final_refit(out, protocol)


def test_confirmation_resume_checks_digest_and_never_repeats_fit(run, monkeypatch):
    out, protocol = run
    monkeypatch.setattr(research, 'load_data', lambda: pytest.fail('confirmation cannot repeat'))
    research.confirmation(out, protocol)
    (out / 'decision.sha256').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        research.confirmation(out, protocol)


def test_confirmation_requests_past_only_champion_member_and_commits_decision(run, monkeypatch):
    out, protocol = run
    (out / 'decision.json').unlink()
    (out / 'decision.sha256').unlink()
    frame = pd.DataFrame({'week': pd.to_datetime(['2026-05-25', '2026-05-25']),
                          'warehouse_name': ['w', 'w'], 'article': ['a', 'b'],
                          'actual': [2., 4.], 'mean_28': [1 / 7, 2 / 7],
                          'n_sale_days_90': [10., 10.], 'zero_week_share_13': [.2, .2],
                          'last_week': [1., 2.], 'mean_week_4': [1., 2.],
                          'mean_week_8': [1., 2.], 'mean_week_13': [1., 2.]})
    values = {'dynamics': {'probability': np.ones(2), 'magnitude': np.array([1., 2.]),
                           'lgbm_raw': np.array([1., 2.])},
              'poisson': {'raw': np.array([2., 4.])}}
    calls = []
    monkeypatch.setattr(research, 'load_data', lambda: (pd.DataFrame(), pd.DataFrame()))

    def past_only_fold(out_arg, origin, sales, stock, names):
        calls.append((origin, names))
        return frame, values, {}

    monkeypatch.setattr(research, 'fold', past_only_fold)
    research.confirmation(out, protocol)
    assert calls == [('2026-05-24', ['dynamics', 'poisson'])]
    decision = research.verify_decision(out, protocol)
    assert decision['baseline']['smape'] == pytest.approx(200 / 3)
    assert decision['candidate']['smape'] == 0
    assert decision['promote']


@pytest.fixture
def final_run(run):
    out, protocol = run
    target = out / 'final'
    target.mkdir()
    (target / 'model.pkl').write_bytes(b'model')
    (target / 'preprocessor.pkl').write_bytes(b'preprocessor')
    decision = research.read(out / 'decision.json')
    config = {'model_type': 'research', 'model_file': 'model.pkl',
              'preprocessor_file': 'preprocessor.pkl',
              'model_params': {'recipe': decision['selected_recipe']},
              'model_artifact_sha256': research.sha(target / 'model.pkl'),
              'preprocessor_sha256': research.sha(target / 'preprocessor.pkl'),
              'provenance': {'research_protocol_sha256': research.sha(out / 'protocol.json'),
                             'selection_sha256': research.sha(out / 'selection.json'),
                             'decision_sha256': research.sha(out / 'decision.json'),
                             'history_cutoff': protocol['history_cutoff']}}
    research.write(target / 'config.json', config)
    research.freeze_record(out / 'final_training.json', {
        'config_sha256': research.sha(target / 'config.json'),
        'model_sha256': config['model_artifact_sha256'],
        'preprocessor_sha256': config['preprocessor_sha256']})
    return out, protocol


def test_valid_existing_final_is_checked_without_retraining(final_run, monkeypatch):
    out, protocol = final_run
    monkeypatch.setattr(research, 'load_data', lambda: pytest.fail('must resume without fit'))
    research.final_refit(out, protocol)


@pytest.mark.parametrize('file', ['final/model.pkl', 'final/preprocessor.pkl', 'final/config.json',
                                  'final_training.json'])
def test_existing_final_corruption_rejected_before_fit(final_run, monkeypatch, file):
    out, protocol = final_run
    (out / file).write_text('{}', encoding='utf-8')
    monkeypatch.setattr(research, 'load_data', lambda: pytest.fail('must reject before fit'))
    with pytest.raises((ValueError, KeyError)):
        research.final_refit(out, protocol)


def test_incomplete_final_is_not_silently_overwritten(run, monkeypatch):
    out, protocol = run
    (out / 'final').mkdir()
    (out / 'final/model.pkl').write_bytes(b'partial')
    monkeypatch.setattr(research, 'load_data', lambda: pytest.fail('must reject before fit'))
    with pytest.raises(ValueError, match='Incomplete final artifact'):
        research.final_refit(out, protocol)
