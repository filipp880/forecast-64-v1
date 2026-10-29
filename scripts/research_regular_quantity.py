"""Recent regular-demand L1 experts; finite development only, no May reads."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from app.calibration import combine
from app.features import KEY,CAT_FEATURES
from app.event_features import FEATURE_COLUMNS
import research_events as events
from scripts.research_weekly import fast_metrics,freeze_record,frozen_record,sha,read,write

ORIGINS=['2025-10-19','2026-01-18','2026-03-22']
VARIANTS=['recent_mae','recent_balanced_mae']
SCOPES={'medium':(2.,10.),'regular':(10.,50.),'all_regular':(2.,None)}
BLENDS=[.5,1.]
PARAMS=dict(objective='regression_l1',metric='None',num_leaves=31,max_depth=7,
    reg_alpha=5.,reg_lambda=20.,learning_rate=.05,min_child_samples=100,
    subsample=.9,subsample_freq=1,colsample_bytree=.8,seed=42,num_threads=6,
    deterministic=True,force_col_wise=True,verbosity=-1)


def scalar_score(y,q):
    y,q=np.asarray(y,float),np.maximum(np.asarray(q,float),0.)
    den=y+q
    sm=np.divide(200*np.abs(y-q),den,out=np.zeros_like(y),where=den>0).mean()
    wa=100*np.abs(y-q).sum()/max(y.sum(),1e-12)
    return float(.7*sm/60+.3*wa/50)


def training_weights(frame,variant,boundary):
    age=(pd.Timestamp(boundary)-pd.to_datetime(frame.feature_origin)).dt.days.to_numpy(float)
    if (age<0).any():raise ValueError('Training origin exceeds label boundary')
    weights=2.**(-age/180.)
    if variant=='recent_balanced_mae':
        mean=max(float(frame.target.mean()),1e-9)
        weights*=.7/(7*frame.mean_28.to_numpy(float)+1)+.3/mean
    elif variant!='recent_mae':raise ValueError('Unknown expert')
    return weights/weights.mean()


def initialize(out):
    if (out/'protocol.json').exists():return verify(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose empty directory')
    out.mkdir(parents=True,exist_ok=True)
    files={};schedules={}
    for origin in ORIGINS:
        schedules[origin],inventory=events.inventory(origin);files.update(inventory)
    paths=[Path(__file__),ROOT/'app/event_features.py',ROOT/'app/features.py',ROOT/'app/calibration.py',
           ROOT/'scripts/research_events.py',ROOT/'scripts/research_weekly.py',ROOT/'model/config.json']
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in paths:
        key=path.relative_to(ROOT).as_posix();files[key]=sha(path)
        target=snapshot/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    protocol=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),origins=ORIGINS,
        variants=VARIANTS,scopes=SCOPES,blends=BLENDS,params=PARAMS,max_iterations=250,
        early_stopping_rounds=30,early_stopping_metric='.7SMAPE/60+.3WAPE/50 on inner regular validation',
        training_scope='past weekly mean>=2, all nonnegative targets including zeros',
        recency_half_life_days=180,sample_weights='recent_mae: recency; balanced: recency*(.7/(pastweeklymean+1)+.3/trainmean(target))',
        prediction='expert clipped>=0; base zero rows remain zero; positive base may become zero; continuous blend',
        memory_limit_gib=3,schedules=schedules,files=files,confirmation_read=False,
        champion_recipe=read(ROOT/'model/config.json')['model_params']['recipe'],
        aggressive='both mean primary ratios<1; each-origin primary<=1.03',
        balanced='aggressive plus mean auxiliaries<=1.03 and each-origin auxiliaries<=1.08',
        previously_inspected_history=True)
    freeze_record(out/'protocol.json',protocol)
    print('REGULAR QUANTITY PROTOCOL FROZEN',sha(out/'protocol.json'),flush=True)
    return protocol


def verify(out):
    protocol=frozen_record(out/'protocol.json')
    for name,digest in protocol['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    return protocol


def load_regular(origin,names,boundary):
    result=[]
    for name in names:
        frame=events.cached(origin,name)
        if (pd.to_datetime(frame.week)+pd.Timedelta(days=6)).max()>pd.Timestamp(boundary):
            raise ValueError('Labels exceed origin')
        result.append(frame.loc[frame.mean_28.mul(7).ge(2)].copy())
    frame=pd.concat(result,ignore_index=True)
    if frame.empty:raise ValueError('No regular-demand training data')
    return frame


def fit(train,variant,boundary,valid=None,iterations=None):
    x,categories=events.encode(train)
    dataset=lgb.Dataset(x,label=train.target.to_numpy(float),weight=training_weights(train,variant,boundary),
                        categorical_feature=CAT_FEATURES,free_raw_data=True)
    callbacks=[]
    def memory_callback(env):
        if psutil.Process().memory_info().rss>3*2**30:raise MemoryError('Regular expert exceeded 3 GiB')
    callbacks.append(memory_callback)
    kwargs={}
    if valid is not None:
        xv,_=events.encode(valid,categories)
        kwargs['valid_sets']=[lgb.Dataset(xv,label=valid.target.to_numpy(float),reference=dataset,
                                         categorical_feature=CAT_FEATURES)]
        kwargs['feval']=lambda prediction,data:('smape_wape',scalar_score(data.get_label(),prediction),False)
        callbacks.append(lgb.early_stopping(30,first_metric_only=True,verbose=False))
    booster=lgb.train(PARAMS,dataset,num_boost_round=int(iterations or 250),callbacks=callbacks,**kwargs)
    return booster,categories


def champion_for(origin,frame,recipe):
    old=pd.read_parquet(events.CHAMPION_ROOT/origin/'evaluation.parquet')
    if not frame[['week',*KEY]].astype(str).equals(old[['week',*KEY]].astype(str)):
        raise ValueError('Outer keys differ')
    values=pickle.loads((events.CHAMPION_ROOT/origin/'dynamics-values.pkl').read_bytes())
    return old,combine(values['probability'],values['magnitude'],values['lgbm_raw'],
                       old.mean_28.to_numpy(float)*7,old.n_sale_days_90.to_numpy(float),recipe['post'])


def route(base,raw,weekly,lower,upper,blend):
    result=np.asarray(base,float).copy()
    mask=(weekly>=lower)&((weekly<upper) if upper is not None else True)
    mixed=(1-blend)*result+blend*np.maximum(raw,0.)
    result[mask]=np.where(result[mask]>0,mixed[mask],0.)
    return result


def assess(records,champions):
    ratios={o:{k:records[o][k]/max(champions[o][k],1e-12) for k in ['smape','wape','rmse','bias','rmsle']} for o in ORIGINS}
    means={k:float(np.mean([v[k] for v in ratios.values()])) for k in ratios[ORIGINS[0]]}
    aggressive=means['smape']<1 and means['wape']<1 and all(v[k]<=1.03 for v in ratios.values() for k in ['smape','wape'])
    balanced=aggressive and all(means[k]<=1.03 for k in ['rmse','bias','rmsle']) and all(v[k]<=1.08 for v in ratios.values() for k in ['rmse','bias','rmsle'])
    return dict(means=means,per_origin=ratios,primary_score=.7*means['smape']+.3*means['wape'],
                aggressive_eligible=bool(aggressive),balanced_eligible=bool(balanced))


def development(out,protocol):
    if (out/'development.json').exists():return frozen_record(out/'development.json')
    if psutil.virtual_memory().available<2.5*2**30:raise MemoryError('Need at least 2.5 GiB available before fitting')
    started=time.perf_counter();monitor=events.MemoryMonitor(3.);monitor.thread.start()
    results={};champions={}
    try:
        for origin in ORIGINS:
            schedule=protocol['schedules'][origin];directory=out/origin;directory.mkdir(exist_ok=True)
            train=load_regular(origin,schedule['inner'],schedule['inner_origin'])
            valid=load_regular(origin,[schedule['valid']],origin);monitor.check();iterations={}
            for variant in VARIANTS:
                print('REGULAR INNER',origin,variant,len(train),flush=True)
                booster,_=fit(train,variant,schedule['inner_origin'],valid)
                iterations[variant]=booster.best_iteration or booster.current_iteration()
                del booster;gc.collect();monitor.check()
            del train,valid;gc.collect()
            refit=load_regular(origin,schedule['refit'],origin)
            frame=events.cached(origin,schedule['test'],labels=False)
            old,base=champion_for(origin,frame,protocol['champion_recipe'])
            y=old.actual.to_numpy(float);codes,_=pd.factorize(pd.MultiIndex.from_frame(old[KEY]))
            champions[origin]=fast_metrics(y,base,codes)
            for variant in VARIANTS:
                print('REGULAR REFIT',origin,variant,len(refit),iterations[variant],flush=True)
                booster,categories=fit(refit,variant,origin,iterations=iterations[variant]);x,_=events.encode(frame,categories)
                raw=np.maximum(booster.predict(x,num_threads=6),0.)
                booster.save_model(str(directory/f'{variant}.txt'));write(directory/f'{variant}.categories.json',categories)
                np.save(directory/f'{variant}.raw.npy',raw)
                for scope,(lower,upper) in SCOPES.items():
                    for blend in BLENDS:
                        name=f'{variant}_{scope}_{blend}'
                        q=route(base,raw,7*old.mean_28.to_numpy(float),lower,upper,blend)
                        metrics=fast_metrics(y,q,codes);results.setdefault(name,{})[origin]=metrics
                        np.save(directory/f'{name}.prediction.npy',q)
                        print('REGULAR METRICS',origin,name,json.dumps(metrics),flush=True)
                freeze_record(directory/f'{variant}.json',dict(iterations=iterations[variant],train_rows=len(refit),feature_count=len(FEATURE_COLUMNS)))
                del booster,x,raw;gc.collect();monitor.check()
            del refit,frame,old;gc.collect();monitor.check()
        summary=dict(protocol_sha256=sha(out/'protocol.json'),champion=champions,results=results,
            assessments={n:assess(r,champions) for n,r in results.items()},confirmation_read=False,
            promotion_executed=False,seconds=time.perf_counter()-started,peak_sampled_rss_gib=monitor.peak/2**30)
        freeze_record(out/'development.json',summary);return summary
    finally:
        monitor.stop.set();monitor.thread.join(timeout=1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'artifacts/regular-quantity-20261029')
    args=parser.parse_args(argv);out=args.out.resolve();protocol=initialize(out)
    result=development(out,protocol);verify(out);print(json.dumps(result['assessments'],indent=2))


if __name__=='__main__':main()
