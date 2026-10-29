"""Frozen conditional count distribution and SMAPE/MAE Bayes decisions.

Only January/March development is read. May requires a separate parent-selected
confirmation implementation; this runner cannot read it or replace production.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from app.calibration import combine
from app.event_features import FEATURE_COLUMNS
from app.features import CAT_FEATURES, KEY
from app.metrics import compute_all_metrics
import research_events as events

ORIGINS = ['2026-01-18','2026-03-22']
UPPER_EDGES = np.array([1.,2.,3.,4.,7.,15.])
FALLBACK_REPS = np.array([0.,1.,2.,3.,4.,6.,12.,24.])
FIXED_ACTIONS = np.array([0.,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.])
MULTIPLIERS = [0.,1.,2.]
SCOPES = {'low':(.5,2.),'low_medium':(.5,10.)}
PARAMS = dict(objective='multiclass',num_class=8,metric='None',num_leaves=31,max_depth=7,
              reg_alpha=5.,reg_lambda=20.,learning_rate=.05,min_child_samples=100,
              subsample=.9,subsample_freq=1,colsample_bytree=.8,seed=42,num_threads=6,
              deterministic=True,force_col_wise=True,verbosity=-1)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(path,obj):
    path=Path(path)
    if path.exists() or path.with_suffix('.sha256').exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha(path)+'\n',encoding='utf-8')


def frozen(path):
    path=Path(path)
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Frozen record changed')
    return read(path)


def quantity_class(y):
    y=np.asarray(y,float)
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid quantity label')
    return np.where(y==0,0,np.searchsorted(UPPER_EDGES,y,side='left')+1).astype('int8')


def representatives(y):
    y=np.asarray(y,float);labels=quantity_class(y)
    values=FALLBACK_REPS.copy()
    for k in range(1,8):
        if (labels==k).any():values[k]=y[labels==k].mean()
    return values


def risk_table(reps,actions,train_mean,multiplier):
    reps=np.asarray(reps,float);actions=np.asarray(actions,float)
    den=reps[:,None]+actions[None,:]
    smape=np.divide(200*np.abs(reps[:,None]-actions[None,:]),den,
                    out=np.zeros_like(den),where=den>0)
    return .7/60*smape + multiplier*.3/50*100*np.abs(reps[:,None]-actions[None,:])/max(train_mean,1e-9)


def decode(probabilities,reps,train_mean,multiplier):
    probabilities=np.asarray(probabilities,float)
    if probabilities.ndim==1:probabilities=probabilities.reshape(-1,8,order='F')
    if probabilities.shape[1]!=8:raise ValueError('Expected eight class probabilities')
    actions=np.unique(np.r_[FIXED_ACTIONS,np.asarray(reps,float)])
    if not np.isfinite(probabilities).all() or (probabilities<0).any():raise ValueError('Invalid probability')
    risks=risk_table(reps,actions,train_mean,multiplier)
    return actions[np.argmin(probabilities@risks,axis=1)]


def scalar_score(y,q):
    y,q=np.asarray(y,float),np.asarray(q,float)
    d=y+q
    smape=np.divide(200*np.abs(y-q),d,out=np.zeros_like(y),where=d>0).mean()
    wape=100*np.abs(y-q).sum()/max(y.sum(),1e-12)
    return float(.7*smape/60+.3*wape/50)


def initialize(out):
    if (out/'protocol.json').exists():return verify(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose empty run directory')
    out.mkdir(parents=True,exist_ok=True)
    files={};schedules={}
    for origin in ORIGINS:
        schedule,records=events.inventory(origin);schedules[origin]=schedule;files.update(records)
    paths=[Path(__file__),ROOT/'scripts/research_events.py',ROOT/'app/event_features.py',
           ROOT/'app/features.py',ROOT/'app/enhancements.py',ROOT/'app/calibration.py',ROOT/'app/metrics.py',
           *[ROOT/'model'/n for n in ['model.pkl','preprocessor.pkl','config.json']]]
    files.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in paths:
        if path.suffix=='.py':
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
    protocol=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),origins=ORIGINS,
        schedules=schedules,params=PARAMS,feature_columns=FEATURE_COLUMNS,max_iterations=200,
        early_stopping=dict(metric='.7*SMAPE/60+.3*WAPE/50 of Bayes decoder multiplier1',
                            evaluate_every=5,first_iteration=True,patience_iterations=30),
        quantity_classes=['0','(0,1]','(1,2]','(2,3]','(3,4]','(4,7]','(7,15]','(15,infinity)'],
        representative='mean quantity per class on fit training only; fixed fallback if absent',
        missing_class_fallback=FALLBACK_REPS.tolist(),fixed_actions=FIXED_ACTIONS.tolist(),
        action_grid='unique fixed_actions plus fit-train class representatives',
        probability_decision='expected .7/60*SMAPE + multiplier*.3/50*100*absolute_error/train_mean',
        multipliers=MULTIPLIERS,scopes=SCOPES,champion_recipe=read(ROOT/'model/config.json')['model_params']['recipe'],
        balanced=dict(mean_primary_lt=1.,per_origin_primary_max=1.01,mean_auxiliary_max=1.03,
                      per_origin_auxiliary_max=1.08),
        aggressive=dict(mean_primary_lt=1.,per_origin_primary_max=1.03,auxiliary_veto=False),
        files=files,memory_limit_gib=4.,threads=6,seed=42,confirmation_read=False,
        historical_period_previously_inspected=True,promotion_allowed=False)
    freeze(out/'protocol.json',protocol)
    print('COUNT PROTOCOL FROZEN',sha(out/'protocol.json'),flush=True)
    return protocol


def verify(out):
    protocol=frozen(out/'protocol.json')
    for name,digest in protocol['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen count input changed: '+name)
    return protocol


def metrics(frame,q):
    truth=frame[['week',*KEY]].assign(quantity=frame.actual.to_numpy(float))
    prediction=truth.copy();prediction['quantity']=q
    return compute_all_metrics(truth,prediction)


def assess(candidates,champions):
    names=['smape','wape','rmse','bias','rmsle']
    ratios={o:{n:candidates[o][n]/max(champions[o][n],1e-12) for n in names} for o in ORIGINS}
    means={n:float(np.mean([r[n] for r in ratios.values()])) for n in names}
    primary=all(means[n]<1 for n in ['smape','wape'])
    balanced=(primary and all(r[n]<=1.01 for r in ratios.values() for n in ['smape','wape'])
              and all(means[n]<=1.03 for n in names[2:])
              and all(r[n]<=1.08 for r in ratios.values() for n in names[2:]))
    aggressive=primary and all(r[n]<=1.03 for r in ratios.values() for n in ['smape','wape'])
    return dict(mean_ratios=means,per_origin_ratios=ratios,balanced_eligible=bool(balanced),
                aggressive_eligible=bool(aggressive),primary_score=.7*means['smape']+.3*means['wape'])


def champion(frame,raw,recipe):
    # Current champion is deliberately reconstructed without mutable routing code.
    if recipe['source']!='dynamics' or set(recipe)-{'source','post'}:
        raise ValueError('Expected original dynamics champion')
    return combine(raw['probability'],raw['magnitude'],raw['lgbm_raw'],
                   frame.mean_28.to_numpy(float)*7,frame.n_sale_days_90.to_numpy(float),recipe['post'])


def fit(train,valid=None,iterations=None):
    x,categories=events.encode(train)
    y=train.target.to_numpy(float)
    reps=representatives(y);mean=float(y.mean())
    data=lgb.Dataset(x,label=quantity_class(y),categorical_feature=CAT_FEATURES,free_raw_data=True)
    callbacks=[lgb.log_evaluation(25)]
    def budget(env):
        if psutil.Process().memory_info().rss>4*2**30:raise MemoryError('Count fit exceeds 4 GiB')
    callbacks.append(budget)
    kwargs={}
    if valid is not None:
        xv,_=events.encode(valid,categories)
        yy=valid.target.to_numpy(float)
        validation=lgb.Dataset(xv,label=quantity_class(yy),reference=data,categorical_feature=CAT_FEATURES,free_raw_data=True)
        state={'iteration':0,'score':None}
        def evaluator(probability,dataset):
            state['iteration']+=1
            if state['iteration']==1 or state['iteration']%5==0:
                state['score']=scalar_score(yy,decode(probability,reps,mean,1.))
            return 'bayes_smape_wape',state['score'],False
        kwargs.update(valid_sets=[validation],feval=evaluator)
        callbacks.append(lgb.early_stopping(30,first_metric_only=True,verbose=False))
    with threadpool_limits(limits=1,user_api='blas'):
        booster=lgb.train(PARAMS,data,num_boost_round=int(iterations or 200),callbacks=callbacks,**kwargs)
    return booster,categories,reps,mean


def development(out,protocol):
    if (out/'development.json').exists():return frozen(out/'development.json')
    start=time.perf_counter();monitor=events.MemoryMonitor(4.)
    monitor.thread.start();results={};champions={};fits={}
    try:
        for origin in ORIGINS:
            schedule=protocol['schedules'][origin];directory=out/origin;directory.mkdir(exist_ok=True)
            train=events.load_frames(origin,schedule,schedule['inner'],schedule['inner_origin'])
            valid=events.load_frames(origin,schedule,[schedule['valid']],origin)
            monitor.check();print('COUNT INNER',origin,len(train),len(valid),flush=True)
            booster,_,_,_=fit(train,valid)
            iterations=int(booster.best_iteration or booster.current_iteration())
            del booster,train,valid;gc.collect()
            refit=events.load_frames(origin,schedule,schedule['refit'],origin)
            monitor.check();print('COUNT REFIT',origin,len(refit),iterations,flush=True)
            booster,categories,reps,mean=fit(refit,iterations=iterations)
            test=events.cached(origin,schedule['test'],labels=False)
            x,_=events.encode(test,categories)
            probabilities=booster.predict(x,num_threads=6)
            booster.save_model(str(directory/'count_model.txt'))
            np.save(directory/'probability.npy',probabilities)
            freeze(directory/'fit.json',dict(iterations=iterations,categories=categories,
                representatives=reps.tolist(),training_mean=mean,training_rows=len(refit),features=len(FEATURE_COLUMNS)))
            old=pd.read_parquet(events.CHAMPION_ROOT/origin/'evaluation.parquet')
            if not test[['week',*KEY]].astype(str).equals(old[['week',*KEY]].astype(str)):
                raise ValueError('Count predictions/champion keys differ')
            if not np.array_equal(test.actual.to_numpy(float),old.actual.to_numpy(float)):
                raise ValueError('Count/champion facts differ')
            raw=pickle.loads((events.CHAMPION_ROOT/origin/'dynamics-values.pkl').read_bytes())
            base=champion(old,raw,protocol['champion_recipe'])
            champions[origin]=metrics(old,base);volume=old.mean_28.to_numpy(float)*7
            for multiplier in MULTIPLIERS:
                with threadpool_limits(limits=1,user_api='blas'):
                    decoded=decode(probabilities,reps,mean,multiplier)
                for scope,(lower,upper) in SCOPES.items():
                    key=f'{scope}_lambda{multiplier:g}'
                    mask=(volume>=lower)&(volume<upper)
                    q=np.where(mask,decoded,base)
                    np.save(directory/f'{key}.prediction.npy',q)
                    primary=metrics(old,q)
                    pp=old.actual.to_numpy(float)>0;qp=q>0
                    results.setdefault(key,{})[origin]=dict(metrics=primary,changed_rows=int((q!=base).sum()),
                        predicted_volume=float(q.sum()),event_mismatch=float((pp!=qp).mean()),
                        zero_share=float((q==0).mean()),precision=float((pp&qp).sum()/max(qp.sum(),1)),
                        recall=float((pp&qp).sum()/max(pp.sum(),1)),scope_rows=int(mask.sum()))
                    print('COUNT METRICS',origin,key,json.dumps(primary),flush=True)
            fits[origin]=dict(iterations=iterations,refit_rows=len(refit))
            del booster,refit,test,x,probabilities,old,raw;gc.collect();monitor.check()
        assessments={key:assess({o:r['metrics'] for o,r in per_origin.items()},champions)
                     for key,per_origin in results.items()}
        report=dict(protocol_sha256=sha(out/'protocol.json'),results=results,champion=champions,
                    assessments=assessments,fits=fits,seconds=time.perf_counter()-start,
                    peak_sampled_rss_gib=monitor.peak/2**30,confirmation_read=False,
                    selected_recipe=None,promotion_executed=False)
        freeze(out/'development.json',report)
        return report
    finally:
        monitor.stop.set();monitor.thread.join(timeout=1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/count-distribution-20261029')
    parser.add_argument('--stage',choices=['freeze','development'],required=True)
    args=parser.parse_args(argv);out=args.out.resolve();protocol=initialize(out)
    if args.stage=='development':
        report=development(out,protocol);print(json.dumps(report['assessments'],indent=2),flush=True)
    verify(out)


if __name__=='__main__':main()
