"""PyCharm: train daily models, evaluate their calendar-week sums against current weekly champion.

Normal Run trains on the user's computer. --check NEVER trains or installs packages.
Production model/ and forecast.csv are never replaced by this experiment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
CONFIG=dict(seed=42,threads=6,max_iterations=350,early_stopping=35,memory_limit_gib=8.,
    weekly_blend_weights=[.25,.5,.75],blend_top_daily=3,final_if_better=True)
VERSIONS={'numpy':'2.1.1','pandas':'2.3.3','lightgbm':'4.7.0','pyarrow':'25.0.1',
    'scipy':'1.15.3','psutil':'7.2.2','threadpoolctl':'3.6.0','scikit-learn':'1.7.2'}
ORIGINS=['2026-01-18','2026-03-22'];METRICS=['smape','wape','rmse','bias','rmsle'];WEIGHTS=[.5,.2,.15,.1,.05]
NAMES=['event','positive','log'];EVIDENCE=ROOT/'artifacts/all40-0a385edab6b1'
EXPECTED_WEEKLY_MODEL='201c6b747ee351feff7f2017c2c45820862ad53fa0ab9cd9332425a4945d59fd'

def log(*a):print(datetime.now().strftime('%H:%M:%S'),*a,flush=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def parquet(p,frame):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    frame.to_parquet(temp,index=False);temp.replace(p)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def environment(check):
    if not (3,10)<=sys.version_info[:2]<=(3,12):raise RuntimeError('Use Python 3.10-3.12.')
    missing=[]
    for name,version in VERSIONS.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual='missing'
        if actual!=version:missing.append((name,actual,version))
    if not missing:return False
    # Reuse the completed round's environment before downloading anything.
    existing=ROOT/'.venv-all40'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if existing.exists() and Path(sys.executable).resolve()!=existing.resolve():
        subprocess.check_call([str(existing),str(Path(__file__).resolve()),*sys.argv[1:]]);return True
    if check:raise RuntimeError('Missing dependencies: '+repr(missing))
    target=ROOT/'.venv-daily-weekly';python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():subprocess.check_call([sys.executable,'-m','venv',str(target)])
    subprocess.check_call([str(python),'-m','pip','install',*[f'{k}=={v}' for k,v in VERSIONS.items()]])
    if python.resolve()==Path(sys.executable).resolve():raise RuntimeError('Environment setup failed to fix versions')
    subprocess.check_call([str(python),str(Path(__file__).resolve()),*sys.argv[1:]]);return True

@contextmanager
def lock(path):
    with Path(path).open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('Another daily-to-weekly run is active') from e
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

class Experiment:
    def __init__(self):
        if sha(ROOT/'model/model.pkl')!=EXPECTED_WEEKLY_MODEL:raise ValueError('Current weekly champion changed')
        selection=read(EVIDENCE/'SELECTION.json');self.champion=selection['best_mean']
        if read(ROOT/'model/config.json')['provenance']['winner']!=self.champion['id']:
            raise ValueError('Current model and historical benchmark differ')
        bundle=ROOT/'artifacts/colab-sequential-20261029-final/colab_round08'
        self.common=load_module('daily_weekly_common',bundle/'run.py');self.common.verify(bundle)
        self.bundle=bundle
        from scripts.research_events import inventory
        paths=[Path(__file__),ROOT/'app/daily_bridge.py',ROOT/'data/sales_dataset_hakaton_train.csv',
            ROOT/'data/stock_dataset_hakaton_train.csv',ROOT/'model/model.pkl',ROOT/'model/config.json',
            ROOT/'model/preprocessor.pkl',ROOT/'forecast.csv',ROOT/'forecast.manifest.json',EVIDENCE/'SELECTION.json',
            bundle/'MANIFEST.json',ROOT/'scripts/research_events.py',ROOT/'scripts/research_weekly.py']
        paths.extend((ROOT/'app').rglob('*.py'))
        fingerprints={};self.schedules={}
        for origin in [*ORIGINS,'2026-05-24']:
            schedule,found=inventory(origin);self.schedules[origin]=schedule;fingerprints.update(found)
        self.final_manifest=read(ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json')
        paths.append(ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json')
        for name,h in self.final_manifest.items():
            p=ROOT/'artifacts/improvements-20261028-r1/final-cache'/name
            if sha(p)!=h:raise ValueError('Final cache differs: '+name)
            fingerprints[str(p.relative_to(ROOT))]=h
        for origin in ORIGINS:
            paths.append(EVIDENCE/'predictions'/f'{self.champion["id"]}_{origin}.npy')
        fingerprints.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
        self.protocol=dict(version=1,config=CONFIG,files=fingerprints,origins=ORIGINS,
            baseline_id=self.champion['id'],baseline_sha256=EXPECTED_WEEKLY_MODEL,
            purpose='Train on DAILY quantity; evaluate SUMS over nine complete calendar weeks on identical keys and facts',
            final_label_end='2026-07-26',stock_aggregation='sum',negative_stock='preserve',
            historical_final_daily_weights_not_used=True,independent_confirmation=False,
            source_note='Daily architecture adapted to current as-of feature/stock contract, not a replay of obsolete max/clipped stock preprocessing')
        identity=hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()[:12]
        self.out=ROOT/'artifacts'/('daily-to-weekly-'+identity);self.out.mkdir(parents=True,exist_ok=True)
        p=self.out/'protocol.json'
        if p.exists() and read(p)!=self.protocol:raise ValueError('Frozen protocol changed')
        if not p.exists():write(p,self.protocol)
        self.ph=sha(p);self.history=None;self.records=[]
        log('RUN',self.out)

    def ensure_history(self):
        if self.history is None:
            from app.csv_io import read_csv_checked
            from app.daily_bridge import DailyHistory
            log('PREP daily quantity history; no training')
            s=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv',parse_dates=['date'],
                usecols=['date','warehouse_name','article','quantity'])
            self.history=DailyHistory(s,'2026-07-26');del s;gc.collect()
        return self.history

    def verify_inputs(self):
        for name,value in self.protocol['files'].items():
            if sha(ROOT/name)!=value:raise ValueError('An input changed during this run: '+name)

    def weekly(self,origin,name,final=False):
        from app.daily_bridge import DAILY_HISTORY
        from app.features import CAT_FEATURES
        path=(ROOT/'artifacts/improvements-20261028-r1/final-cache'/name if final else
              ROOT/'artifacts/ideas-20261028-r2'/origin/'cache'/name)
        cols=list(dict.fromkeys(CAT_FEATURES+[c for c in DAILY_HISTORY if c!='dow_mean']+['week','h']))
        label='actual' if name.endswith('-False.parquet') else 'target'
        return pd.read_parquet(path,columns=cols+[label])

    def frame(self,origin,name,final=False,evaluation=False):
        path=self.out/'features'/f'{"FINAL" if final else origin}_{name}'
        if path.exists():return pd.read_parquet(path)
        w=self.weekly(origin,name,final);cutoff=pd.Timestamp(name[:10])
        boundary=pd.Timestamp('2026-07-26' if final else origin)
        if not evaluation:w=w.loc[w.week+pd.Timedelta(days=6)<=boundary].copy()
        if w.empty:return None
        d=self.ensure_history().expand(w,cutoff,training=True,
            label_end=(cutoff+pd.Timedelta(weeks=9)) if evaluation else boundary)
        from app.daily_bridge import aggregate_daily
        y=aggregate_daily(d,d.target,w)
        actual=w['actual' if 'actual' in w else 'target'].to_numpy(float)
        if not np.allclose(y,actual,rtol=1e-6,atol=1e-6):raise ValueError('Daily facts do not sum to cached weekly facts')
        parquet(path,d);return d

    def dataset(self,origin,split,final=False):
        if final:
            boundary=pd.Timestamp('2026-05-24' if split=='inner' else '2026-07-26')
            names=list(self.final_manifest) if split!='valid' else ['2026-05-24-False.parquet']
            source_origin='2026-05-24' if split=='valid' else origin
        else:
            schedule=self.schedules[origin];names=schedule[split]
            if isinstance(names,str):names=[names]
            boundary=pd.Timestamp(schedule['inner_origin'] if split=='inner' else origin);source_origin=origin
        frames=[]
        for name in names:
            f=self.frame(source_origin,name,final=final and split!='valid',evaluation=final and split=='valid')
            if f is None:continue
            # Retain complete weeks only; never count a missing future target as zero.
            f=f.loc[f.week+pd.Timedelta(days=6)<=boundary].copy()
            if f.empty:continue
            for c in CAT_FEATURES:f[c]=f[c].astype(str)
            frames.append(f)
        if not frames:raise ValueError('Empty chronological dataset')
        frame=pd.concat(frames,ignore_index=True);del frames;gc.collect()
        if frame.date.max()>boundary:raise ValueError('Target dates leak beyond split')
        return frame

def encode(frame,categories=None):
    x=frame[DAILY_COLUMNS].copy()
    if categories is None:categories={c:sorted(x[c].astype(str).unique()) for c in CAT_FEATURES}
    for c in CAT_FEATURES:x[c]=pd.Categorical(x[c].astype(str),categories=categories[c])
    return x,categories

def checkpoint(folder,name,ph):
    path=folder/name
    if not path.exists():return None
    record=read(path)
    if record['protocol_sha256']!=ph:raise ValueError('Checkpoint protocol differs')
    for f,h in record['files'].items():
        if sha(folder/f)!=h:raise ValueError('Checkpoint file differs: '+str(folder/f))
    return record

def target(frame,name):
    y=frame.target.to_numpy(float)
    return (y>0).astype('int8') if name=='event' else np.log1p(y) if name=='log' else y

def fit_model(e,name,origin,final=False):
    import lightgbm as lgb
    import psutil
    folder=e.out/'models'/name/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    if checkpoint(folder,'complete.json',e.ph):return folder
    params=dict(objective={'event':'binary','positive':'poisson','log':'regression'}[name],
        metric={'event':'binary_logloss','positive':'poisson','log':'rmse'}[name],num_leaves=127 if name=='log' else 63,
        learning_rate=.05,min_data_in_leaf=20 if name=='log' else 50,bagging_fraction=.9,bagging_freq=1,
        feature_fraction=.9,lambda_l2=8.,seed=CONFIG['seed'],num_threads=CONFIG['threads'],
        deterministic=True,force_col_wise=True,verbosity=-1)
    def guard(env):
        if psutil.Process().memory_info().rss>CONFIG['memory_limit_gib']*2**30:
            raise MemoryError('Memory budget exceeded; completed models remain saved')
    inner=checkpoint(folder,'inner.json',e.ph)
    if inner is None:
        train=e.dataset(origin,'inner',final);valid=e.dataset(origin,'valid',final)
        if name=='positive':train=train.loc[train.target>0].copy();valid=valid.loc[valid.target>0].copy()
        if train.empty or valid.empty:raise ValueError('Empty inner training/validation')
        x,categories=encode(train);xv,_=encode(valid,categories)
        schema=dict(columns=DAILY_COLUMNS,categories=categories,kind=name,
            target_end=str(pd.to_datetime(train.date).max().date()),validation_end=str(pd.to_datetime(valid.date).max().date()))
        ds=lgb.Dataset(x,label=target(train,name),categorical_feature=CAT_FEATURES)
        dv=lgb.Dataset(xv,label=target(valid,name),reference=ds,categorical_feature=CAT_FEATURES)
        log('FIT inner',name,origin,'daily rows',len(train))
        model=lgb.train(params,ds,num_boost_round=CONFIG['max_iterations'],valid_sets=[dv],
            callbacks=[guard,lgb.early_stopping(CONFIG['early_stopping'],verbose=False)])
        iterations=int(model.best_iteration or model.current_iteration())
        temp=folder/'inner_model.txt.tmp';model.save_model(str(temp),num_iteration=iterations);temp.replace(folder/'inner_model.txt')
        write(folder/'inner_schema.json',schema)
        write(folder/'inner.json',dict(protocol_sha256=e.ph,iterations=iterations,
            files={n:sha(folder/n) for n in ['inner_model.txt','inner_schema.json']}))
        del train,valid,x,xv,ds,dv,model;gc.collect();inner=read(folder/'inner.json')
    train=e.dataset(origin,'refit',final)
    if name=='positive':train=train.loc[train.target>0].copy()
    x,categories=encode(train);ds=lgb.Dataset(x,label=target(train,name),categorical_feature=CAT_FEATURES)
    schema=dict(columns=DAILY_COLUMNS,categories=categories,kind=name,training_rows=len(train),
        target_end=str(pd.to_datetime(train.date).max().date()))
    log('FIT refit',name,origin,len(train),'iterations',inner['iterations'])
    model=lgb.train(params,ds,num_boost_round=inner['iterations'],callbacks=[guard])
    temp=folder/'refit_model.txt.tmp';model.save_model(str(temp));temp.replace(folder/'refit_model.txt')
    write(folder/'refit_schema.json',schema)
    write(folder/'complete.json',dict(protocol_sha256=e.ph,iterations=inner['iterations'],
        files={n:sha(folder/n) for n in ['inner.json','inner_model.txt','inner_schema.json','refit_model.txt','refit_schema.json']}))
    del train,x,ds,model;gc.collect();return folder

def predict_model(folder,frame):
    import lightgbm as lgb
    schema=read(folder/'refit_schema.json');x,_=encode(frame,schema['categories'])
    model=lgb.Booster(model_file=str(folder/'refit_model.txt'));p=model.predict(x,num_threads=CONFIG['threads'])
    if schema['kind']=='log':p=np.expm1(np.clip(p,-20,20))
    return np.maximum(p,0.)

def policies():
    result=[]
    for gate in [.35,.5,.65]:
        for weight in [0.,.3,.6,1.]:
            for rounding in ['daily','weekly','none']:
                result.append(dict(id=f'g{gate}_w{weight}_{rounding}',mode='gated',gate=gate,weight=weight,rounding=rounding))
    for weight in [.3,1.]:
        for rounding in ['weekly','none']:
            result.append(dict(id=f'expected_w{weight}_{rounding}',mode='expected',weight=weight,rounding=rounding))
    return result

def compose(frame,parts,weekly,policy):
    event=parts['event'];positive=parts['positive'];regression=parts['log']
    hurdle=event*positive if policy['mode']=='expected' else np.where(event>=policy['gate'],positive,0.)
    raw=policy['weight']*hurdle+(1-policy['weight'])*regression
    if policy['rounding']=='daily':raw=np.rint(raw)
    result=aggregate_daily(frame,raw,weekly)
    return np.rint(result) if policy['rounding']=='weekly' else result

def evaluate(e,name,predictions,details,loaded):
    record=dict(id=name,details=details,metrics={},per_origin_ratio={},independent_confirmation=False)
    folder=e.out/'predictions';folder.mkdir(exist_ok=True)
    for origin,p in predictions.items():
        f,b=loaded[origin];y=f.actual.to_numpy(float);code=e.common.codes(f)
        metrics=e.common.metrics(y,p,code);base=e.common.metrics(y,b,code)
        record['metrics'][origin]=metrics
        record['per_origin_ratio'][origin]=float(sum(w*e.common.ratio(metrics[k],base[k]) for w,k in zip(WEIGHTS,METRICS)))
        np.save(folder/f'{name}_{origin}.npy',p)
    record['weighted_ratio']=float(np.mean(list(record['per_origin_ratio'].values())))
    record['worst_window_ratio']=max(record['per_origin_ratio'].values())
    write(e.out/'evaluations'/f'{name}.json',record);e.records.append(record)
    log('RESULT',name,round(record['weighted_ratio'],8));return record

def run(e):
    loaded={};daily={};components={}
    for origin in ORIGINS:
        f,_,_=e.common.load(e.bundle,origin)
        base=np.load(EVIDENCE/'predictions'/f'{e.champion["id"]}_{origin}.npy',allow_pickle=False)
        # Verify that the reference is the same weekly champion published in model/.
        measured=e.common.metrics(f.actual,base,e.common.codes(f))
        for k in METRICS:
            if not np.isclose(measured[k],e.champion['metrics'][origin][k],rtol=1e-11):raise ValueError('Weekly benchmark metrics differ')
        loaded[origin]=(f,base)
        d=e.frame(origin,e.schedules[origin]['test'],evaluation=True)
        daily[origin]=d
        parquet(e.out/'evaluation_data'/f'{origin}_daily.parquet',d)
        parquet(e.out/'evaluation_data'/f'{origin}_weekly.parquet',f[['week',*KEY,'actual']])
        components[origin]={name:predict_model(fit_model(e,name,origin),d) for name in NAMES}
        np.savez_compressed(e.out/'evaluation_data'/f'{origin}_components.npz',**components[origin])
    evaluate(e,'weekly_baseline',{o:loaded[o][1] for o in ORIGINS},dict(kind='baseline'),loaded)
    for policy in policies():
        p={o:compose(daily[o],components[o],loaded[o][0],policy) for o in ORIGINS}
        evaluate(e,policy['id'],p,dict(kind='daily',policy=policy),loaded)
    # A finite grid over completed standalone recipes; resume cannot create blends of blends.
    solo=sorted([r for r in e.records if r['details']['kind']=='daily'],key=lambda r:(r['weighted_ratio'],r['id']))
    distinct=[];seen=set()
    for r in solo:
        signature=tuple(round(r['metrics'][o][k],8) for o in ORIGINS for k in METRICS)
        if signature not in seen:distinct.append(r);seen.add(signature)
    for r in distinct[:CONFIG['blend_top_daily']]:
        for weight in CONFIG['weekly_blend_weights']:
            for rounded in [False,True]:
                p={o:(1-weight)*loaded[o][1]+weight*np.load(e.out/'predictions'/f'{r["id"]}_{o}.npy',allow_pickle=False) for o in ORIGINS}
                if rounded:p={o:np.rint(q) for o,q in p.items()}
                name=f'blend_{r["id"]}_{weight}_{int(rounded)}'
                evaluate(e,name,p,dict(kind='blend',daily_id=r['id'],policy=r['details']['policy'],weight=weight,round_int=rounded),loaded)
    eligible=[r for r in e.records if r['worst_window_ratio']<=1.01]
    best=min(eligible,key=lambda r:(r['weighted_ratio'],r['worst_window_ratio'],r['id']))
    accepted=best['details']['kind']!='baseline' and best['weighted_ratio']<1-1e-8
    selection=dict(best=best,best_daily=solo[0],accepted=accepted,records=sorted(e.records,key=lambda r:r['weighted_ratio']),
        metric_weights=dict(zip(METRICS,WEIGHTS)),baseline=e.champion,independent_confirmation=False,
        note='Same January/March windows and weekly facts; daily SMAPE is not used for selection; ranks are not official scores')
    write(e.out/'SELECTION.json',selection)
    if accepted and CONFIG['final_if_better']:
        e.verify_inputs()
        models={n:fit_model(e,n,'2026-07-26',final=True) for n in NAMES}
        final_forecast(e,models,best['details'])
    write(e.out/'summary.json',dict(completed=True,accepted=accepted,production_changed=False,
        best=best,best_daily=solo[0],independent_confirmation=False,
        final_models_trained=bool(accepted and CONFIG['final_if_better']),
        baseline_model_sha256=EXPECTED_WEEKLY_MODEL))
    package(e)

def final_forecast(e,models,details):
    from app.csv_io import read_csv_checked
    from app.features import build_panel,make_features,truncate_history
    from app.coverage import complete_required_forecast
    from app.predictor import Predictor
    from app.pipeline import postprocess_fallback
    from app.submission import validate_forecast
    log('PREP final calendar weeks; training targets end July 26')
    s=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv');st=read_csv_checked(ROOT/'data/stock_dataset_hakaton_train.csv')
    s,st=truncate_history(s,st,'2026-07-26')
    panel=build_panel(s,st,history_end='2026-07-26')
    weekly=make_features(panel,'2026-07-26');del panel;gc.collect()
    daily=e.ensure_history().expand(weekly,'2026-07-26')
    parts={name:predict_model(folder,daily) for name,folder in models.items()}
    policy=details['policy'];q=compose(daily,parts,weekly,policy)
    observed=weekly[['week',*KEY]].copy();observed['week']=observed.week.dt.strftime('%Y-%m-%d');observed['quantity']=q
    champion=Predictor(ROOT/'model').load();base=read_csv_checked(ROOT/'forecast.csv')
    completed,pairs,_=complete_required_forecast(observed,s,'2026-07-26','2026-07-27',base[KEY],
        fallback_transform=lambda f:postprocess_fallback(champion.model,champion.config,f))
    # The few pairs without joint history retain the same fallback as the weekly baseline.
    keys=['week',*KEY];candidate=completed.merge(base,on=keys,suffixes=('_daily','_weekly'),validate='one_to_one')
    if len(candidate)!=len(base):raise ValueError('Final daily catalogue differs from weekly baseline')
    if details['kind']=='blend':
        value=(1-details['weight'])*candidate.quantity_weekly.to_numpy(float)+details['weight']*candidate.quantity_daily.to_numpy(float)
        if details['round_int']:value=np.rint(value)
    else:value=candidate.quantity_daily.to_numpy(float)
    forecast=candidate[keys].copy();forecast['quantity']=value
    forecast=forecast.sort_values(keys).reset_index(drop=True);validate_forecast(forecast,pairs,'2026-07-27')
    forecast.to_csv(e.out/'daily_candidate_forecast.csv',index=False)
    pairs.to_csv(e.out/'daily_candidate_pairs.csv',index=False)
    parquet(e.out/'evaluation_data/final_daily.parquet',daily)
    write(e.out/'FINAL_MODELS.json',dict(label_end='2026-07-26',details=details,production_changed=False,
        serving_integration_required=True,forecast_sha256=sha(e.out/'daily_candidate_forecast.csv'),
        baseline_model_sha256=EXPECTED_WEEKLY_MODEL,model_folders={k:str(v.relative_to(e.out)) for k,v in models.items()}))

def package(e):
    e.verify_inputs()
    export=e.out/'export';export.mkdir(exist_ok=True)
    for name in ['models','predictions','evaluations','evaluation_data']:
        if (e.out/name).exists():shutil.copytree(e.out/name,export/name,dirs_exist_ok=True)
    for p in e.out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.txt']:shutil.copy2(p,export/p.name)
    shutil.copy2(Path(__file__),export/'train_daily_to_weekly.py')
    shutil.copy2(ROOT/'app/daily_bridge.py',export/'daily_bridge.py')
    files={p.relative_to(export).as_posix():sha(p) for p in export.rglob('*') if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    write(export/'OUTPUT_MANIFEST.json',dict(protocol_sha256=e.ph,files=files,production_changed=False))
    archive=e.out/'daily_to_weekly_results.zip';temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in export.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(export))
    with zipfile.ZipFile(temp) as z:
        if z.testzip():raise ValueError('Corrupt result archive')
    temp.replace(archive);log('DONE. SEND THIS FILE:',archive)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if environment(args.check):return
    global np,pd,DAILY_COLUMNS,CAT_FEATURES,KEY,aggregate_daily
    import numpy as np
    import pandas as pd
    from app.daily_bridge import DAILY_COLUMNS,aggregate_daily
    from app.features import CAT_FEATURES,KEY
    e=Experiment()
    if args.check:
        for origin in ORIGINS:
            f=e.frame(origin,e.schedules[origin]['test'],evaluation=True)
            assert set(DAILY_COLUMNS)<=set(f) and len(f)==len(e.weekly(origin,e.schedules[origin]['test']))*7
        v=e.dataset('2026-07-26','valid',final=True)
        assert v.date.min()==pd.Timestamp('2026-05-25') and v.date.max()==pd.Timestamp('2026-07-26')
        partial=e.frame('2026-07-26','2026-06-28-True.parquet',final=True)
        assert partial.date.max()==pd.Timestamp('2026-07-26') and partial.h.max()==28
        log('CHECK PASSED. No models trained.');return
    with lock(e.out/'.run.lock'):
        if (e.out/'daily_to_weekly_results.zip').exists():
            with zipfile.ZipFile(e.out/'daily_to_weekly_results.zip') as z:
                manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
                if manifest['protocol_sha256']!=e.ph:raise ValueError('Result protocol changed')
                for name,h in manifest['files'].items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Result changed: '+name)
            log('ALREADY COMPLETED. SEND:',e.out/'daily_to_weekly_results.zip');return
        try:run(e)
        except BaseException:
            (e.out/'LAST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            log('STOPPED. Checkpoints preserved. Send LAST_ERROR.txt from',e.out);raise

if __name__=='__main__':main()
