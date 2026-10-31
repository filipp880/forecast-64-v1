"""Publish the exact reproduced joint submission, keeping the previous release for rollback."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_joint_release import OUT, EXPECTED_FORECAST, EXPECTED_WEEKLY, read, sha, write
from scripts.promote_cv_release import publish_files
from scripts.verify_release import verify


def main():
    assert sha(ROOT / 'model/model.pkl') == EXPECTED_WEEKLY, 'Current model differs from expected release'
    old_forecast = '86e4aa6a95efe99ed13cc05a14a015b9aa542ee7097caefac84a5116c18f4c37'
    assert sha(ROOT / 'forecast.csv') == old_forecast, 'Current forecast changed'
    assert sha(OUT / 'forecast.csv') == EXPECTED_FORECAST
    parity = read(OUT / 'verification.json')
    assert parity['verified'] and parity['generator']['exact_submitted_bytes']
    docker = read(OUT / 'docker-check/full_http_result.json')
    assert docker['valid'] and docker['published_forecast_values_equal'] and docker['deterministic_bytes']
    assert read(OUT / 'docker-check/engine_evidence.json')['network_mode'] == 'none'
    verified = verify(OUT / 'forecast.csv', OUT / 'forecast.pairs.csv', OUT / 'forecast.manifest.json', OUT / 'model')
    manifest = read(OUT / 'forecast.manifest.json')
    manifest['model_files'] = {p.name: sha(p) for p in (OUT / 'model').iterdir() if p.is_file()}
    manifest['output_files'] = dict(forecast=str(ROOT / 'forecast.csv'), catalogue=str(ROOT / 'required_pairs.csv'),
                                  manifest=str(ROOT / 'forecast.manifest.json'))
    manifest['promotion'] = dict(at=datetime.now(timezone.utc).isoformat(), selected_winner='joint_candidate_forecast',
        expected_submitted_forecast_sha256=EXPECTED_FORECAST, exact_submitted_bytes=True,
        old_model_sha256=EXPECTED_WEEKLY, old_forecast_sha256=old_forecast,
        docker_offline_verified=True, independent_confirmation=False,
        external_evidence='user screenshot; scorer-side file SHA not supplied')
    payloads = {ROOT / 'model' / p.name: p.read_bytes() for p in (OUT / 'model').iterdir() if p.is_file()}
    payloads[ROOT / 'forecast.csv'] = (OUT / 'forecast.csv').read_bytes()
    payloads[ROOT / 'required_pairs.csv'] = (OUT / 'forecast.pairs.csv').read_bytes()
    payloads[ROOT / 'forecast.manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    backup = OUT / 'prior-root-release'
    publish_files(payloads, ROOT, backup,
        lambda: verify(ROOT / 'forecast.csv', ROOT / 'required_pairs.csv', ROOT / 'forecast.manifest.json', ROOT / 'model'))
    report = dict(published_locally=True, network_published=False, training_executed=False,
        backup=str(backup), forecast_sha256=sha(ROOT / 'forecast.csv'), model_sha256=sha(ROOT / 'model/model.pkl'),
        validation=verified)
    write(OUT / 'promotion.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
