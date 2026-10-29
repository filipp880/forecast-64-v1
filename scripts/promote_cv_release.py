"""Publish an explicitly adaptive-CV-tuned local release with full rollback."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from scripts.verify_release import verify
from scripts.research_weekly import frozen_record,sha


def publish_files(payloads,root,backup,check=lambda:None):
    root,backup=Path(root).resolve(),Path(backup).resolve()
    if not backup.is_relative_to(root):raise ValueError('Backup must stay inside release workspace')
    paths=[Path(p).resolve() for p in payloads]
    if any(not p.is_relative_to(root) for p in paths):raise ValueError('Publication path escapes release workspace')
    if len(set(paths))!=len(paths):raise ValueError('Publication paths collide')
    backup.mkdir(parents=True,exist_ok=False)
    originals={p:p.read_bytes() if p.exists() else None for p in paths}
    for i,(path,data) in enumerate(originals.items()):
        if data is not None:(backup/f'{i}.before').write_bytes(data)
    (backup/'paths.json').write_text(json.dumps([str(p) for p in paths],indent=2),encoding='utf-8')
    published=[]
    with tempfile.TemporaryDirectory(prefix='publication-',dir=backup) as directory:
        stage=Path(directory).resolve()
        assert stage.is_relative_to(backup)
        for i,(path,data) in enumerate(payloads.items()):(stage/str(i)).write_bytes(data)
        try:
            for i,path in enumerate(paths):
                path.parent.mkdir(parents=True,exist_ok=True)
                os.replace(stage/str(i),path)
                published.append(path)
            check()
        except BaseException:
            for i,path in reversed(list(enumerate(paths))):
                if path not in published:continue
                data=originals[path]
                if data is None:path.unlink(missing_ok=True)
                else:
                    restore=stage/f'restore-{i}';restore.write_bytes(data);os.replace(restore,path)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'artifacts/smape-wape-final-20261029')
    parser.add_argument('--expected-model-sha',required=True)
    parser.add_argument('--expected-forecast-sha',required=True)
    args=parser.parse_args(argv);candidate=args.candidate.resolve()
    if sha(ROOT/'model/model.pkl')!=args.expected_model_sha or sha(ROOT/'forecast.csv')!=args.expected_forecast_sha:
        raise ValueError('Current release differs from the explicitly expected baseline')
    selection=frozen_record(ROOT/'artifacts/smape-wape-adaptive-cv-20261029/selection.json')
    if not selection['accepted']:raise ValueError('No accepted simultaneous-primary CV recipe')
    model=candidate/'model';submission=candidate/'submission'
    check=verify(submission/'forecast.csv',submission/'forecast.pairs.csv',submission/'forecast.manifest.json',model)
    config=json.loads((model/'config.json').read_text(encoding='utf-8'))
    if config['model_params']['recipe']!=selection['recipe']:raise ValueError('Candidate recipe differs from selected recipe')
    if config['provenance']['cv_selection_sha256']!=sha(ROOT/'artifacts/smape-wape-adaptive-cv-20261029/selection.json'):
        raise ValueError('Candidate provenance differs from CV selection')
    manifest=json.loads((submission/'forecast.manifest.json').read_text(encoding='utf-8'))
    manifest['model_files']={p.name:sha(p) for p in model.iterdir() if p.is_file()}
    manifest['output_files']={'forecast':str(ROOT/'forecast.csv'),'catalogue':str(ROOT/'required_pairs.csv'),
                              'manifest':str(ROOT/'forecast.manifest.json')}
    manifest['promotion']={'at':datetime.now(timezone.utc).isoformat(),'evaluation_kind':selection['evaluation_kind'],
        'independent_confirmation':False,'old_model_sha256':args.expected_model_sha,
        'old_forecast_sha256':args.expected_forecast_sha,'cv_primary_metrics_lower_on_all_reported_origins':True}
    payloads={ROOT/'model'/p.name:p.read_bytes() for p in model.iterdir() if p.is_file()}
    payloads[ROOT/'forecast.csv']=(submission/'forecast.csv').read_bytes()
    payloads[ROOT/'required_pairs.csv']=(submission/'forecast.pairs.csv').read_bytes()
    # Completion marker is last; verify the entire committed release before success.
    payloads[ROOT/'forecast.manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    backup=candidate/'prior-root-release'
    publish_files(payloads,ROOT,backup,lambda:verify(ROOT/'forecast.csv',ROOT/'required_pairs.csv',ROOT/'forecast.manifest.json',ROOT/'model'))
    report={'published_locally':True,'network_published':False,'backup':str(backup),
            'evaluation_kind':selection['evaluation_kind'],'independent_confirmation':False,
            'model_sha256':sha(ROOT/'model/model.pkl'),'forecast_sha256':sha(ROOT/'forecast.csv'),
            'validation':check}
    (candidate/'promotion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
