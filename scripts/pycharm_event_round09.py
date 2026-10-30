"""Prepare without fitting locally; user runs the self-contained PyCharm training project."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/pycharm-round09-12-20261030/forecast_round09'
ORIGINS=['2026-01-18','2026-03-22']
NAMES=['smape','wape','rmse','bias','rmsle']
WEIGHTS=[.5,.2,.15,.1,.05]
CAT=['warehouse_name','article','category_lvl_1','category_lvl_2','category_lvl_3','characteristic']
PARAMS=dict(objective='binary',metric='binary_logloss',learning_rate=.05,num_leaves=31,max_depth=7,
    min_data_in_leaf=100,bagging_fraction=.9,bagging_freq=1,feature_fraction=.8,
    lambda_l1=5.,lambda_l2=20.,seed=42,num_threads=6,deterministic=True,force_col_wise=True,verbosity=-1)


def check_environment():
    if not ((3,10)<=sys.version_info[:2]<=(3,12)):raise RuntimeError('Use Python 3.10–3.12')
    versions={'numpy':'2.1.1','pandas':'2.3.3','lightgbm':'4.7.0','scikit-learn':'1.7.2',
              'scipy':'1.15.3','pyarrow':'25.0.1','threadpoolctl':'3.6.0','psutil':'7.2.2'}
    for name,expected in versions.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual='missing'
        if actual!=expected:raise RuntimeError(f'{name}: expected {expected}, found {actual}. Run python -m pip install -r requirements.txt')


@contextmanager
def exclusive_run(project):
    with (project/'.training.lock').open('a+b') as f:
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
        except OSError as error:raise RuntimeError('Another process is using this project') from error
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)


def common_module(project):
    sys.path.insert(0,str(project/'common'))
    spec=importlib.util.spec_from_file_location('round1_common',project/'common/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def extra_features(frame):
    """Uses only named past inputs, never target, actual or future observations."""
    lag=frame[[f'weekly_lag_{i}' for i in range(1,11)]].to_numpy(float)
    known=np.isfinite(lag)
    recent=np.divide(np.nansum(lag[:,:2],axis=1),known[:,:2].sum(axis=1),
        out=np.full(len(frame),np.nan),where=known[:,:2].sum(axis=1)>0)
    past=np.divide(np.nansum(lag[:,2:],axis=1),known[:,2:].sum(axis=1),
        out=np.full(len(frame),np.nan),where=known[:,2:].sum(axis=1)>=4)
    change=(recent+.1)/(past+.1)
    result=pd.DataFrame(index=frame.index)
    result['regime_change_ratio']=change
    result['regime_absolute_change']=recent-past
    result['regime_recent_support']=known[:,:2].sum(axis=1)
    result['regime_prior_support']=known[:,2:].sum(axis=1)
    result['regime_id']=np.where(~np.isfinite(change),-1,np.where(change<.7,0,np.where(change>1.3,2,1)))
    stock=frame.stock_last.to_numpy(float);missing=~np.isfinite(stock)
    positive=np.where(missing,np.nan,(stock>0).astype(float))
    zero=frame.stock_zero_share_90.to_numpy(float)
    age=frame.event_last_positive_age.to_numpy(float)
    streak=frame.event_recent_zero_streak.to_numpy(float)
    rate=frame.event_rate_13.to_numpy(float)
    result['stock_event_has_stock_x_age']=positive*age
    result['stock_event_no_stock_x_streak']=(1-positive)*streak
    result['stock_event_zero_fraction_x_rate']=zero*rate
    result['stock_event_zero_fraction_x_h']=zero*frame.h.to_numpy(float)
    result['stock_event_last_to_mean']=stock/np.maximum(np.abs(frame.stock_mean_28.to_numpy(float)),1.)
    result['stock_event_cover_x_rate']=frame.days_of_cover.to_numpy(float)*rate
    result['stock_event_negative_x_gap']=np.where(missing,np.nan,(stock<0).astype(float))*age
    result['stock_event_known']=np.isfinite(stock).astype(float)
    return result.replace([np.inf,-np.inf],np.nan).astype('float32')


def variants(base_columns):
    regime=['regime_change_ratio','regime_absolute_change','regime_recent_support','regime_prior_support','regime_id']
    stock=['stock_event_has_stock_x_age','stock_event_no_stock_x_streak','stock_event_zero_fraction_x_rate',
        'stock_event_zero_fraction_x_h','stock_event_last_to_mean','stock_event_cover_x_rate',
        'stock_event_negative_x_gap','stock_event_known']
    return [dict(id='09_resumption',task=9,features=base_columns,heads=['resume']),
            dict(id='10_regime_heads',task=10,features=base_columns+regime,heads=['regime0','regime1','regime2']),
            dict(id='11_stock_interactions',task=11,features=base_columns+stock,heads=['all']),
            dict(id='12_horizon_heads',task=12,features=base_columns,heads=['h0','h1','h2']),
            dict(id='23_regime_features_control',task=23,features=base_columns+regime,heads=['all'])]


def mask(frame,head):
    if head=='all':return np.ones(len(frame),bool)
    if head=='resume':return frame.weekly_lag_1.to_numpy(float)==0
    if head.startswith('regime'):return frame.regime_id.to_numpy(float)==int(head[-1])
    if head.startswith('h'):return (frame.h.to_numpy(int)-1)//3==int(head[-1])
    raise ValueError('Unknown head '+head)


def verify(project):
    manifest=read(project/'INPUT_MANIFEST.json')
    for name,digest in manifest['files'].items():
        path=(project/name).resolve()
        if not path.is_relative_to(project.resolve()) or not path.is_file() or sha(path)!=digest:
            raise ValueError('Input changed: '+name)
    protocol=read(project/'protocol.json')
    common_module(project).verify(project/'common')
    return protocol


def load_split(project,protocol,origin,split,features,head):
    schedule=protocol['schedules'][origin]
    boundary=schedule['inner_origin'] if split=='inner' else origin
    columns=list(dict.fromkeys(features+CAT+['week','feature_origin','h','target','weekly_lag_1','regime_id']))
    frames=[]
    for name in schedule[split] if isinstance(schedule[split],list) else [schedule[split]]:
        frame=pd.read_parquet(project/name,columns=columns)
        if (pd.to_datetime(frame.week)+pd.Timedelta(days=6)).max()>pd.Timestamp(boundary):
            raise ValueError('Training labels exceed boundary: '+name)
        frames.append(frame.loc[mask(frame,head)].copy())
    return pd.concat(frames,ignore_index=True)


def encode(frame,features,categories=None):
    x=frame[features].copy()
    if categories is None:categories={c:sorted(x[c].astype(str).unique()) for c in CAT}
    for c in CAT:x[c]=pd.Categorical(x[c].astype(str),categories=categories[c])
    return x,categories


def valid_checkpoint(folder,name,protocol_sha):
    path=folder/name
    if not path.exists():return None
    record=read(path)
    if record['protocol_sha256']!=protocol_sha:raise ValueError('Resume protocol differs')
    for name,digest in record['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Resume output changed: '+name)
    return record


def fit_head(project,protocol,variant,origin,head):
    """Fitting boundary: called only by the user's Run, never by prepare/check."""
    import lightgbm as lgb
    import psutil
    directory=project/'training_results'/variant['id']/origin/head;directory.mkdir(parents=True,exist_ok=True)
    protocol_sha=sha(project/'protocol.json')
    saved=valid_checkpoint(directory,'complete.json',protocol_sha)
    if saved:return saved
    features=variant['features'];inner=valid_checkpoint(directory,'inner.json',protocol_sha)
    params={**protocol['params'],'num_threads':min(6,os.cpu_count() or 2)}
    def memory_guard(env):
        if psutil.Process().memory_info().rss>protocol['max_process_memory_gib']*2**30:
            raise MemoryError('Memory limit exceeded. Send training log; do not change the grid silently.')
    if inner is None:
        train=load_split(project,protocol,origin,'inner',features,head)
        valid=load_split(project,protocol,origin,'valid',features,head)
        y=(train.target.to_numpy(float)>0).astype('int8');yv=(valid.target.to_numpy(float)>0).astype('int8')
        if len(train)<1000 or len(valid)<100 or len(np.unique(y))<2 or len(np.unique(yv))<2:
            record=dict(protocol_sha256=protocol_sha,skipped=True,reason='insufficient past support; retain baseline probability',files={})
            write(directory/'complete.json',record);return record
        x,categories=encode(train,features);xv,_=encode(valid,features,categories)
        dataset=lgb.Dataset(x,label=y,categorical_feature=CAT,free_raw_data=True)
        validation=lgb.Dataset(xv,label=yv,reference=dataset,categorical_feature=CAT,free_raw_data=True)
        print('INNER',variant['id'],origin,head,len(train),len(valid),flush=True)
        booster=lgb.train(params,dataset,num_boost_round=protocol['maximum_iterations'],valid_sets=[validation],
            callbacks=[memory_guard,lgb.early_stopping(protocol['early_stopping_rounds'],verbose=False)])
        iterations=int(booster.best_iteration or booster.current_iteration())
        booster.save_model(str(directory/'inner_model.txt'),num_iteration=iterations)
        write(directory/'inner_schema.json',dict(features=features,categories=categories,params=params))
        oof=valid[['week','warehouse_name','article','h','feature_origin','target']].copy()
        oof['probability']=booster.predict(xv,num_iteration=iterations,num_threads=params['num_threads'])
        oof.to_parquet(directory/'inner_validation_predictions.parquet',index=False)
        inner=dict(protocol_sha256=protocol_sha,iterations=iterations,train_rows=len(train),valid_rows=len(valid),
            train_label_end=str((pd.to_datetime(train.week)+pd.Timedelta(days=6)).max().date()),
            valid_label_end=str((pd.to_datetime(valid.week)+pd.Timedelta(days=6)).max().date()),
            files={name:sha(directory/name) for name in ['inner_model.txt','inner_schema.json','inner_validation_predictions.parquet']})
        write(directory/'inner.json',inner)
        del train,valid,x,xv,dataset,validation,booster,oof;gc.collect()
    train=load_split(project,protocol,origin,'refit',features,head)
    x,categories=encode(train,features);y=(train.target.to_numpy(float)>0).astype('int8')
    print('REFIT',variant['id'],origin,head,len(train),inner['iterations'],flush=True)
    dataset=lgb.Dataset(x,label=y,categorical_feature=CAT,free_raw_data=True)
    booster=lgb.train(params,dataset,num_boost_round=inner['iterations'],callbacks=[memory_guard])
    booster.save_model(str(directory/'model.txt'))
    write(directory/'schema.json',dict(features=features,categories=categories,params=params))
    record=dict(protocol_sha256=protocol_sha,skipped=False,iterations=inner['iterations'],rows=len(train),
        label_end=str((pd.to_datetime(train.week)+pd.Timedelta(days=6)).max().date()),
        files={name:sha(directory/name) for name in ['inner.json','inner_model.txt','inner_schema.json',
            'inner_validation_predictions.parquet','model.txt','schema.json']})
    write(directory/'complete.json',record)
    del train,x,y,dataset,booster;gc.collect()
    return record


def train(project):
    import lightgbm as lgb
    protocol=verify(project);common=common_module(project);started=time.perf_counter()
    output=project/'training_results';output.mkdir(exist_ok=True)
    result_path=project/'round09_results.zip'
    if result_path.exists():
        validate_result(project,result_path);print('Already finished. SEND:',result_path);return
    records=[]
    for variant in protocol['variants']:
        dest=output/variant['id'];record=dict(id=variant['id'],task=variant['task'],metrics={},baselines={},heads={},probability_metrics={})
        for origin in ORIGINS:
            frame,values,arrays=common.load(project/'common',origin)
            evaluation=pd.read_parquet(project/protocol['schedules'][origin]['test'])
            if not frame[common.KEY].astype(str).equals(evaluation[common.KEY].astype(str)) or not np.array_equal(frame.actual,evaluation.actual):
                raise ValueError('Evaluation keys/truth differ from fixed baseline')
            probability=values['dynamics']['probability'].copy();head_records={};applied=np.zeros(len(frame),bool)
            for head in variant['heads']:
                info=fit_head(project,protocol,variant,origin,head);head_records[head]=info
                if info['skipped']:continue
                folder=dest/origin/head;schema=read(folder/'schema.json')
                booster=lgb.Booster(model_file=str(folder/'model.txt'))
                if booster.feature_name()!=schema['features']:raise ValueError('Booster feature schema differs')
                m=mask(evaluation,head)
                if not m.any():continue
                x,_=encode(evaluation.loc[m],schema['features'],schema['categories'])
                probability[m]=booster.predict(x,num_threads=min(6,os.cpu_count() or 2))
                applied[m]=True
                del x,booster;gc.collect()
            v=dict(values);v['dynamics']={**values['dynamics'],'probability':probability}
            from app.models.research_model import recipe_prediction
            prediction=recipe_prediction(v,frame,protocol['recipe']);code=common.codes(frame)
            record['metrics'][origin]=common.metrics(frame.actual,prediction,code)
            record['baselines'][origin]=common.metrics(frame.actual,arrays['baseline'],code)
            record['heads'][origin]=head_records
            actual=(frame.actual.to_numpy(float)>0).astype(float)
            from sklearn.metrics import log_loss
            record['probability_metrics'][origin]=dict(brier=float(np.mean((probability-actual)**2)),
                logloss=float(log_loss(actual,np.clip(probability,1e-8,1-1e-8),labels=[0,1])))
            np.save(dest/f'{origin}.probability.npy',probability);np.save(dest/f'{origin}.prediction.npy',prediction)
            write(dest/f'{origin}.diagnostics.json',common.diagnostics(frame,prediction))
            # Same trained event model can improve the hard low-volume segment:
            # replace count P(y>0), retaining its conditional positive distribution.
            old=arrays['count_probability'];changed=old.copy()
            positive=old[:,1:];mass=positive.sum(axis=1,keepdims=True)
            conditional=np.divide(positive,mass,out=np.full_like(positive,1/positive.shape[1]),where=mass>0)
            changed[applied,0]=1-probability[applied]
            changed[applied,1:]=conditional[applied]*probability[applied,None]
            if not np.allclose(changed.sum(axis=1),1.,atol=1e-10):raise ValueError('Count probabilities do not sum to one')
            fit=read(project/'common/data'/f'{origin}.count.json')
            count_values=dict(values)
            for name in values:
                if name.startswith('metric_count_lambda'):
                    multiplier=float(name.removeprefix('metric_count_lambda'))
                    count_values[name]={'raw':common.bayes_quantities(changed,fit['representatives'],fit['training_mean'],multiplier)}
            route_values={'dynamics':v,'count_zero':count_values,'both':{**count_values,'dynamics':v['dynamics']}}
            for route,rv in route_values.items():
                q=prediction if route=='dynamics' else recipe_prediction(rv,frame,protocol['recipe'])
                route_record=record.setdefault('routes',{}).setdefault(route,{'metrics':{},'baselines':{}})
                route_record['metrics'][origin]=common.metrics(frame.actual,q,code)
                route_record['baselines'][origin]=record['baselines'][origin]
                if route!='dynamics':
                    np.save(dest/f'{origin}.{route}.prediction.npy',q)
                    write(dest/f'{origin}.{route}.diagnostics.json',common.diagnostics(frame,q))
        ratios={o:float(sum(w*common.ratio(record['metrics'][o][k],record['baselines'][o][k]) for w,k in zip(WEIGHTS,NAMES))) for o in ORIGINS}
        record.update(per_origin_weighted_ratio=ratios,weighted_ratio=float(np.mean(list(ratios.values()))),
            protocol_sha256=sha(project/'protocol.json'),independent_confirmation=False)
        for route,r in record['routes'].items():
            r['per_origin_weighted_ratio']={o:float(sum(w*common.ratio(r['metrics'][o][k],r['baselines'][o][k])
                for w,k in zip(WEIGHTS,NAMES))) for o in ORIGINS}
            r['weighted_ratio']=float(np.mean(list(r['per_origin_weighted_ratio'].values())))
        record['best_route']=min(record['routes'],key=lambda name:record['routes'][name]['weighted_ratio'])
        write(dest/'result.json',record);records.append(record)
        print('RESULT',variant['id'],{name:r['weighted_ratio'] for name,r in record['routes'].items()},flush=True)
    write(output/'summary.json',dict(records=records,protocol_sha256=sha(project/'protocol.json'),production_changed=False))
    write(output/'run_manifest.json',dict(protocol_sha256=sha(project/'protocol.json'),runner_sha256=sha(Path(__file__)),
        input_manifest_sha256=sha(project/'INPUT_MANIFEST.json'),python=platform.python_version(),platform=platform.platform(),
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','lightgbm','scikit-learn','pyarrow','psutil']},
        seconds=time.perf_counter()-started,training_executed=True,independent_confirmation=False))
    write(output/'protocol.json',protocol)
    files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    write(output/'OUTPUT_MANIFEST.json',dict(protocol_sha256=sha(project/'protocol.json'),files=files))
    temporary=result_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(output))
    temporary.replace(result_path);validate_result(project,result_path)
    print('DONE. SEND THIS FILE:',result_path,flush=True)


def validate_result(project,path):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
        if manifest['protocol_sha256']!=sha(project/'protocol.json'):raise ValueError('Result protocol differs')
        if z.testzip() is not None:raise ValueError('Corrupt result ZIP')
        for name,digest in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Result changed: '+name)


def prepare(project):
    if project.exists():raise FileExistsError('Preserve prior export; choose another directory')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    from scripts import research_events as events
    from app.event_features import make_event_features,FEATURE_COLUMNS
    project.mkdir(parents=True)
    source=ROOT/'artifacts/colab-sequential-20261029-final/colab_round08'
    shutil.copytree(source,project/'common')
    common=common_module(project);common.verify(project/'common')
    schedules={};source_files={};exported={}
    for origin in ORIGINS:
        schedule,inventory=events.inventory(origin);source_files.update(inventory);mapped={}
        for name in sorted(set(schedule['inner']+schedule['refit']+[schedule['valid'],schedule['test']])):
            labels=name!=schedule['test'];raw=events.cached(origin,name,labels=labels)
            x=make_event_features(raw)
            for column in ['week','feature_origin','target' if labels else 'actual']:x[column]=raw[column].to_numpy()
            x=pd.concat([x,extra_features(x)],axis=1)
            if (pd.to_datetime(x.feature_origin)>pd.to_datetime(x.week)).any():raise ValueError('Invalid temporal order')
            if labels and (pd.to_datetime(x.week)+pd.Timedelta(days=6)).max()>pd.Timestamp(origin):raise ValueError('Labels exceed origin')
            rel=f'data/{origin}/{name}'
            path=project/rel;path.parent.mkdir(parents=True,exist_ok=True);x.to_parquet(path,index=False)
            mapped[name]=rel;exported[rel]=dict(rows=len(x),columns=len(x.columns))
            del raw,x;gc.collect()
        schedules[origin]=dict(inner=[mapped[n] for n in schedule['inner']],valid=mapped[schedule['valid']],
            refit=[mapped[n] for n in schedule['refit']],test=mapped[schedule['test']],inner_origin=schedule['inner_origin'])
        print('EXPORTED',origin,len(mapped),flush=True)
    shutil.copy2(Path(__file__),project/'train_round09.py')
    protocol=dict(version=1,origins=ORIGINS,tasks=[9,10,11,12],additional_partial_task=23,
        baseline_protocol_sha256=sha(project/'common/protocol.json'),recipe=read(project/'common/protocol.json')['recipe'],
        params=PARAMS,maximum_iterations=400,early_stopping_rounds=40,max_process_memory_gib=6.,
        variants=variants(list(FEATURE_COLUMNS)),schedules=schedules,source_files=source_files,exported_data=exported,
        label_boundaries='inner labels <= inner_origin; valid/refit labels <= outer origin',
        temporal_split='same fixed purged historical schedules as baseline studies',
        task8_decision='calibration rejected; original probabilities retained as baseline',
        source_model_sha256=sha(ROOT/'model/model.pkl'),independent_confirmation=False,
        fit_location='user PyCharm only; preparation performs no model fitting',
        event31_cached_control='available in common/data/*.npz for paired old-feature control',
        event_output_routes=['dynamics probability only','count P(y>0) only, fixed positive conditional probabilities','both'],
        group_boundaries='regime ratio recent2/prior8, additive0.1; decline<0.7, growth>1.3; missing history retains base')
    write(project/'protocol.json',protocol)
    print('PREPARED',project,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','check','train'],default='train')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    project=args.out.resolve() if args.out else (Path(__file__).resolve().parent if Path(__file__).name=='train_round09.py' else DEFAULT)
    if args.stage=='prepare':prepare(project)
    elif args.check or args.stage=='check':
        check_environment()
        p=verify(project)
        for origin in ORIGINS:
            frame=pd.read_parquet(project/p['schedules'][origin]['test'])
            if frame.groupby(['warehouse_name','article'],observed=True).size().ne(9).any():raise ValueError('Incomplete evaluation horizon')
            if not np.isfinite(frame.actual).all():raise ValueError('Invalid evaluation facts')
        print('CHECK ONLY OK: immutable inputs and temporal data; NO FITTING EXECUTED')
    else:
        check_environment()
        with exclusive_run(project):train(project)


if __name__=='__main__':main()
