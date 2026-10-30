"""Publish the verified all40 serving bundle locally with rollback and backup."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.promote_cv_release import publish_files
from scripts.verify_release import verify

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    candidate=ROOT/'artifacts/all40-serving-20261030'
    old_model='278347022a5d2b2dcf92145155f4a9f728a1e5996a7b89517527bfaddd2d318e'
    old_forecast='38b1514811bd3af7d0cb7233b6563b508730c63677c333a9f062bf571246296e'
    if sha(ROOT/'model/model.pkl')!=old_model or sha(ROOT/'forecast.csv')!=old_forecast:
        raise ValueError('Current release differs from the explicitly verified baseline')
    docker=read(candidate/'docker-check/full_http_result.json')
    if not docker['valid'] or not docker['published_forecast_values_equal'] or not docker['deterministic_bytes']:
        raise ValueError('Docker forecast was not verified')
    if not read(candidate/'feature_parity.json')['passed']:raise ValueError('Feature parity failed')
    verified=verify(candidate/'forecast.csv',candidate/'forecast.pairs.csv',candidate/'forecast.manifest.json',candidate/'model')
    config=read(candidate/'model/config.json')
    if config['provenance']['winner']!='blend_8ab4d78a8929_round':raise ValueError('Unexpected selected model')
    manifest=read(candidate/'forecast.manifest.json')
    manifest['model_files']={p.name:sha(p) for p in (candidate/'model').iterdir() if p.is_file()}
    manifest['output_files']={'forecast':str(ROOT/'forecast.csv'),'catalogue':str(ROOT/'required_pairs.csv'),
        'manifest':str(ROOT/'forecast.manifest.json')}
    manifest['promotion']=dict(at=datetime.now(timezone.utc).isoformat(),independent_confirmation=False,
        selected_winner=config['provenance']['winner'],old_model_sha256=old_model,old_forecast_sha256=old_forecast,
        all_five_metrics_lower_on_both_selection_windows=True,docker_offline_verified=True)
    payloads={ROOT/'model'/p.name:p.read_bytes() for p in (candidate/'model').iterdir() if p.is_file()}
    payloads[ROOT/'forecast.csv']=(candidate/'forecast.csv').read_bytes()
    payloads[ROOT/'required_pairs.csv']=(candidate/'forecast.pairs.csv').read_bytes()
    # Manifest is the completion marker and must be published last.
    payloads[ROOT/'forecast.manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    backup=candidate/'prior-root-release'
    publish_files(payloads,ROOT,backup,
        lambda:verify(ROOT/'forecast.csv',ROOT/'required_pairs.csv',ROOT/'forecast.manifest.json',ROOT/'model'))
    report=dict(published_locally=True,network_published=False,training_executed=False,backup=str(backup),
        model_sha256=sha(ROOT/'model/model.pkl'),forecast_sha256=sha(ROOT/'forecast.csv'),validation=verified)
    (candidate/'promotion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
