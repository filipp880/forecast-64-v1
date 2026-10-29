"""Explicit adaptive development follow-up; no new fits and no May access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import research_count_distribution as count

ORIGINS=['2026-01-18','2026-03-22']
MULTIPLIERS=[3.,4.,8.,16.]
SOURCE=ROOT/'artifacts/count-distribution-20261029'


def initialize(out):
    if (out/'protocol.json').exists():
        protocol=count.frozen(out/'protocol.json')
        for name,expected in protocol['files'].items():
            if count.sha(ROOT/name)!=expected:raise ValueError('Frozen follow-up input changed: '+name)
        return protocol
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose empty directory')
    original=count.verify(SOURCE)
    count.frozen(SOURCE/'development.json')
    out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'scripts/research_count_distribution.py',ROOT/'app/calibration.py',
           ROOT/'app/metrics.py',SOURCE/'protocol.json',SOURCE/'development.json',SOURCE/'output_manifest.json']
    original_outputs=count.read(SOURCE/'output_manifest.json')['files']
    for origin in ORIGINS:
        for name in ['probability.npy','fit.json','count_model.txt']:
            path=SOURCE/origin/name
            if count.sha(path)!=original_outputs[f'{origin}/{name}']:raise ValueError('Original count output changed')
            paths.append(path)
        directory=count.events.CHAMPION_ROOT/origin
        complete=count.read(directory/'complete.json')
        for name in ['evaluation.parquet','dynamics-values.pkl']:
            path=directory/name
            if count.sha(path)!=complete['files'][name]:raise ValueError('OOF cache changed')
            paths.append(path)
    protocol=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),origins=ORIGINS,
        multipliers=MULTIPLIERS,scopes=count.SCOPES,base_protocol_sha256=count.sha(SOURCE/'protocol.json'),
        adaptation='After initial fixed lambda0/1/2 results: larger MAE penalties may cross Pareto boundary. This is adaptive development, not independent confirmation.',
        base_champion_recipe=original['champion_recipe'],decoder='Unchanged Bayes decoder and class representatives from original fit training',
        balanced=original['balanced'],aggressive=original['aggressive'],
        files={p.relative_to(ROOT).as_posix():count.sha(p) for p in paths},
        new_fits=0,confirmation_read=False,promotion_allowed=False,prior_history_inspected=True)
    count.freeze(out/'protocol.json',protocol)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in paths:
        if path.suffix=='.py':
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
    print('COUNT DECODER FOLLOW-UP FROZEN',count.sha(out/'protocol.json'),flush=True)
    return protocol


def development(out,protocol):
    if (out/'development.json').exists():return count.frozen(out/'development.json')
    results={};champions={}
    for origin in ORIGINS:
        directory=out/origin;directory.mkdir(exist_ok=True)
        fit=count.frozen(SOURCE/origin/'fit.json')
        probability=np.load(SOURCE/origin/'probability.npy')
        frame=pd.read_parquet(count.events.CHAMPION_ROOT/origin/'evaluation.parquet')
        raw=pickle.loads((count.events.CHAMPION_ROOT/origin/'dynamics-values.pkl').read_bytes())
        base=count.champion(frame,raw,protocol['base_champion_recipe'])
        champions[origin]=count.metrics(frame,base)
        volume=7*frame.mean_28.to_numpy(float)
        for multiplier in MULTIPLIERS:
            with threadpool_limits(limits=1,user_api='blas'):
                decoded=count.decode(probability,fit['representatives'],fit['training_mean'],multiplier)
            for scope,(lower,upper) in count.SCOPES.items():
                key=f'{scope}_lambda{multiplier:g}';mask=(volume>=lower)&(volume<upper)
                prediction=np.where(mask,decoded,base)
                np.save(directory/f'{key}.prediction.npy',prediction)
                metrics=count.metrics(frame,prediction)
                actual=frame.actual.to_numpy(float)>0;predicted=prediction>0
                results.setdefault(key,{})[origin]=dict(metrics=metrics,changed_rows=int((prediction!=base).sum()),
                    predicted_volume=float(prediction.sum()),zero_share=float((prediction==0).mean()),
                    event_mismatch=float((actual!=predicted).mean()),scope_rows=int(mask.sum()))
                print('COUNT FOLLOW-UP METRICS',origin,key,json.dumps(metrics),flush=True)
    assessments={key:count.assess({o:r['metrics'] for o,r in per_origin.items()},champions)
                 for key,per_origin in results.items()}
    result=dict(protocol_sha256=count.sha(out/'protocol.json'),results=results,champion=champions,
        assessments=assessments,confirmation_read=False,new_fits=0,promotion_executed=False)
    count.freeze(out/'development.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/count-distribution-decoder-20261029')
    args=parser.parse_args(argv);out=args.out.resolve();protocol=initialize(out)
    result=development(out,protocol);initialize(out)
    files={p.relative_to(out).as_posix():count.sha(p) for p in sorted(out.rglob('*'))
           if p.is_file() and p.name!='output_manifest.json'}
    (out/'output_manifest.json').write_text(json.dumps({'protocol_sha256':count.sha(out/'protocol.json'),
        'files':files,'new_fits':0,'confirmation_read':False},indent=2),encoding='utf-8')
    print(json.dumps(result['assessments'],indent=2),flush=True)


if __name__=='__main__':main()
