"""Publish the checked weekly rank winner, preserving a complete rollback copy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.promote_cv_release import publish_files
from scripts.research_weekly import frozen_record, read, sha
from scripts.verify_release import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'artifacts/rank-winner-final-20261029')
    parser.add_argument('--expected-model-sha', required=True)
    parser.add_argument('--expected-forecast-sha', required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    if sha(ROOT / 'model/model.pkl') != args.expected_model_sha or sha(ROOT / 'forecast.csv') != args.expected_forecast_sha:
        raise ValueError('Production release changed since preparation')
    protocol = frozen_record(candidate / 'protocol.json')
    fitted = frozen_record(candidate / 'final_training.json')
    selection_path = ROOT / 'artifacts/score-review-20261029/SELECTED_CONFIGURATION.json'
    if sha(selection_path) != protocol['files'][selection_path.relative_to(ROOT).as_posix()]:
        raise ValueError('Rank selection changed')
    selection = read(selection_path)
    model, submission = candidate / 'model', candidate / 'submission'
    config = read(model / 'config.json')
    if config['model_params']['recipe'] != selection['recipe'] or protocol['recipe'] != selection['recipe']:
        raise ValueError('Candidate recipe differs from rank selection')
    if config['provenance']['rank_selection_sha256'] != sha(selection_path):
        raise ValueError('Candidate rank selection provenance differs')
    if config['provenance']['finalization_protocol_sha256'] != sha(candidate / 'protocol.json'):
        raise ValueError('Candidate finalization provenance differs')
    if sha(model / 'model.pkl') != fitted['model_sha256'] or sha(model / 'config.json') != fitted['config_sha256']:
        raise ValueError('Fitted model bundle changed')
    check = verify(submission / 'forecast.csv', submission / 'forecast.pairs.csv',
                   submission / 'forecast.manifest.json', model)
    manifest = read(submission / 'forecast.manifest.json')
    manifest['model_files'] = {p.name: sha(p) for p in model.iterdir() if p.is_file()}
    manifest['output_files'] = dict(forecast=str(ROOT / 'forecast.csv'), catalogue=str(ROOT / 'required_pairs.csv'),
                                    manifest=str(ROOT / 'forecast.manifest.json'))
    manifest['promotion'] = dict(at=datetime.now(timezone.utc).isoformat(), chosen_id=selection['chosen_id'],
        evaluation_kind=protocol['evaluation_kind'], independent_confirmation=False,
        local_rank_score=selection['local_rank_score'], official_score_known=False,
        old_model_sha256=args.expected_model_sha, old_forecast_sha256=args.expected_forecast_sha)
    payloads = {ROOT / 'model' / p.name: p.read_bytes() for p in model.iterdir() if p.is_file()}
    payloads[ROOT / 'forecast.csv'] = (submission / 'forecast.csv').read_bytes()
    payloads[ROOT / 'required_pairs.csv'] = (submission / 'forecast.pairs.csv').read_bytes()
    payloads[ROOT / 'forecast.manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    backup = candidate / 'prior-root-release'
    publish_files(payloads, ROOT, backup, lambda: verify(ROOT / 'forecast.csv', ROOT / 'required_pairs.csv',
                                                       ROOT / 'forecast.manifest.json', ROOT / 'model'))
    result = dict(published_locally=True, network_published=False, backup=str(backup),
        chosen_id=selection['chosen_id'], independent_confirmation=False,
        model_sha256=sha(ROOT / 'model/model.pkl'), forecast_sha256=sha(ROOT / 'forecast.csv'), validation=check)
    (candidate / 'promotion.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
