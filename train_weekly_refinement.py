"""PyCharm: refine current weekly ensemble, seven declared improvements.

Normal Run trains on the user's computer. --check and --screen never train.
The production model and forecast remain unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
CONFIG=dict(seed=42,threads=6,max_count_iterations=180,early_stopping=35,memory_limit_gib=8.,
    residual_prefix_iterations=64,residual_max_iterations=200,final_if_better=True)
VERSIONS={'numpy':'2.1.1','pandas':'2.3.3','lightgbm':'4.7.0','pyarrow':'25.0.1',
    'scipy':'1.15.3','psutil':'7.2.2','threadpoolctl':'3.6.0','scikit-learn':'1.7.2'}
FINE=[.5,1.,1.5,2.,3.,4.,6.,8.,12.,20.,40.]
VARIANTS=[dict(id='count_metric_stop',edges=FINE,weighting='uniform'),
    dict(id='count_equal_series',edges=FINE,weighting='equal_series'),
    dict(id='count_rare_weight',edges=FINE,weighting='rare'),
    dict(id='count_equal_rare',edges=FINE,weighting='equal_rare'),
    dict(id='count_small_bins',edges=[.25,.5,.75,1.,1.5,2.,3.,4.,6.,8.,12.,20.,40.],weighting='uniform'),
    dict(id='count_compact_bins',edges=[.5,1.,2.,3.,5.,8.,16.,32.],weighting='uniform')]
SIDE=['prefix_prediction','count_action','count_zero_probability','count_expectation','refreshed_prediction','expert_disagreement']

def log(*a):print(datetime.now().strftime('%H:%M:%S'),*a,flush=True)
def environment(check):
    if not (3,10)<=sys.version_info[:2]<=(3,12):raise RuntimeError('Use Python 3.10-3.12')
    missing=[]
    for name,version in VERSIONS.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual='missing'
        if actual!=version:missing.append((name,actual,version))
    if not missing:return False
    entry=str(Path(__file__).resolve());existing=ROOT/'.venv-all40'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if existing.exists() and existing.resolve()!=Path(sys.executable).resolve():
        subprocess.check_call([str(existing),entry,*sys.argv[1:]]);return True
    if check:raise RuntimeError('Missing dependencies: '+repr(missing))
    folder=ROOT/'.venv-weekly-refine';python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():subprocess.check_call([sys.executable,'-m','venv',str(folder)])
    subprocess.check_call([str(python),'-m','pip','install',*[f'{n}=={v}' for n,v in VERSIONS.items()]])
    if python.resolve()==Path(sys.executable).resolve():raise RuntimeError('Environment did not resolve dependencies')
    subprocess.check_call([str(python),entry,*sys.argv[1:]]);return True

def initialize():
    global np,pd,shared,write,read,sha,CAT,KEY,WK,METRICS,WEIGHTS,ORIGINS
    import numpy as np
    import pandas as pd
    from scripts import weekly_refinement_common as shared
    write,read,sha=shared.write,shared.read,shared.sha
    from app.features import CAT_FEATURES as CAT,KEY
    WK=shared.WK;METRICS=shared.METRICS;WEIGHTS=shared.WEIGHTS;ORIGINS=shared.ORIGINS

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
        except OSError as exc:raise RuntimeError('Another weekly refinement run is active') from exc
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def checkpoint(folder,name,ph):
    p=Path(folder)/name
    if not p.exists():return None
    d=read(p)
    if d['protocol_sha256']!=ph:raise ValueError('Checkpoint protocol differs')
    for name,h in d['files'].items():
        if sha(Path(folder)/name)!=h:raise ValueError('Checkpoint file differs: '+str(Path(folder)/name))
    return d

def labels(y,edges):return np.where(y==0,0,np.searchsorted(np.asarray(edges),y,side='left')+1).astype('int32')
def schema_for(frame,variant,categories):
    y=frame.target.to_numpy(float);lab=labels(y,variant['edges']);reps=np.r_[0.,variant['edges'],2*variant['edges'][-1]].astype(float)
    for k in range(1,len(reps)):
        if (lab==k).any():reps[k]=y[lab==k].mean()
    return dict(spec=dict(kind='count',**variant),columns=list(COLUMNS),categories=categories,
        representatives=reps.tolist(),training_mean=float(y.mean()),label_end=str((pd.to_datetime(frame.week)+pd.Timedelta(days=6)).max().date()),
        early_stopping='five final prediction errors in affected demand range 0.5-10; combined 75/25 and rounded',
        training_rows=len(frame))

def sample_weights(frame,kind):
    weight=np.ones(len(frame))
    if kind in ['equal_series','equal_rare']:
        codes,_=pd.factorize(pd.MultiIndex.from_frame(frame[KEY]));counts=np.bincount(codes)
        weight/=counts[codes]
    if kind in ['rare','equal_rare']:weight*=np.where(frame.mean_28.to_numpy(float)*7<2,1.5,1.)
    return weight/weight.mean()

def encode(frame,columns,categories=None):
    x=frame[columns].copy()
    if categories is None:categories={c:sorted(x[c].astype(str).unique()) for c in CAT if c in columns}
    for c,values in categories.items():x[c]=pd.Categorical(x[c].astype(str),categories=values)
    return x,categories

def prefix_part(name,origin,frame,inner=False,final=False):
    """Fixed first 64 iterations: validation-based early stopping cannot affect this prefix."""
    import lightgbm as lgb
    folder=shared.RUN/'models'/name/('FINAL' if final else origin);prefix='inner' if inner else 'refit'
    schema=read(folder/f'{prefix}_schema.json');model=lgb.Booster(model_file=str(folder/f'{prefix}_model.txt'))
    n=CONFIG['residual_prefix_iterations']
    if model.current_iteration()<n:raise ValueError('Required fixed model prefix is unavailable; do not adapt prefix length to labels')
    x,_=encode(frame,schema['columns'],schema['categories'])
    p=model.predict(x,num_iteration=n,num_threads=CONFIG['threads'])
    if schema['spec']['kind']=='count':return p,schema
    if schema['spec'].get('log_target'):p=np.expm1(np.clip(p,-20,20))
    return np.maximum(p,0.),schema

def shadow(frame,origin,inner=False,final=False,prefix=False):
    """Exact current mixing inside 0.5-10; original legacy branch is not needed there."""
    from app.models.research_model import recipe_prediction
    if prefix:
        probability,schema=prefix_part('count_fine',origin,frame,inner,final)
        values={}
        for name in shared.PARTS[1:]:
            p,s=prefix_part(name,origin,frame,inner,final)
            values[name]=shared.decode(p,s,3.) if s['spec']['kind']=='count' else p
        members={'dynamics':dict(probability=values['core_event'],magnitude=values['core_positive'],lgbm_raw=values['core_log']),
            'mae':dict(raw=values['core_mae']),'metric_count_lambda3':dict(raw=values['core_count']),
            'metric_regular_recent_mae':dict(raw=values['core_recent'])}
        refreshed=recipe_prediction(members,frame,read(ROOT/'model/config.json')['model_params']['recipe'])
    else:
        folder=shared.RUN/'models/count_fine'/('FINAL' if final else origin)
        probability,schema=shared.probabilities(folder,frame,inner)
        refreshed=shared.component_values(origin,frame,inner,final)
    action=shared.decode(probability,schema,3.);base=np.rint(.75*action+.25*refreshed)
    features=frame.copy()
    features['prefix_prediction']=base;features['count_action']=action
    features['count_zero_probability']=probability[:,0]
    features['count_expectation']=probability@np.asarray(schema['representatives'])
    features['refreshed_prediction']=refreshed;features['expert_disagreement']=action-refreshed
    for c in SIDE:features[c]=features[c].astype('float32')
    return features,base,refreshed

class Experiment:
    def __init__(self,contexts,screen):
        self.reference=contexts;self.screen=screen
        files=dict(contexts.protocol['files']);files[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        files[str((contexts.out/'SCREEN_RESULTS.json').relative_to(ROOT))]=sha(contexts.out/'SCREEN_RESULTS.json')
        self.protocol=dict(version=1,files=files,config=CONFIG,variants=VARIANTS,
            reference_model_sha256=shared.MODEL_SHA,screen_protocol_sha256=contexts.ph,scope=list(range(1,8)),
            residual_scope='past weekly mean 0.5-10; fixed 64-iteration ensemble, clean prefix predictions after its training boundary',
            residual_meta_split='first six weeks train, last three validate, refit all nine before outer origin',
            count_early_stopping='five error ratios of decoded/combined/rounded output on affected rows',
            independent_confirmation=False,production_changed=False)
        identity=hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()[:12]
        self.out=ROOT/'artifacts'/('weekly-refinement-'+identity);self.out.mkdir(parents=True,exist_ok=True)
        p=self.out/'protocol.json'
        if p.exists() and read(p)!=self.protocol:raise ValueError('Frozen training protocol changed')
        if not p.exists():write(p,self.protocol)
        self.ph=sha(p);self.records=[];log('WEEKLY REFINEMENT',self.out)

    def verify_inputs(self):
        for name,h in self.protocol['files'].items():
            if sha(ROOT/name)!=h:raise ValueError('An input changed: '+name)

    def dataset(self,origin,split,final=False):
        if final:
            boundary=pd.Timestamp('2026-05-24' if split=='inner' else '2026-07-26')
            names=list(self.reference.final_manifest) if split!='valid' else ['2026-05-24-False.parquet']
            source='2026-05-24' if split=='valid' else origin
        else:
            schedule=self.reference.schedules[origin];names=schedule[split]
            if isinstance(names,str):names=[names]
            boundary=pd.Timestamp(schedule['inner_origin'] if split=='inner' else origin);source=origin
        frames=[]
        for name in names:
            f=self.reference.frame(source,name,final=final and split!='valid').rename(columns={'actual':'target'})
            f=f.loc[f.week+pd.Timedelta(days=6)<=boundary].copy()
            if len(f):frames.append(f)
        if not frames:raise ValueError('Empty chronological data')
        result=pd.concat(frames,ignore_index=True);del frames;gc.collect()
        if (result.week+pd.Timedelta(days=6)).max()>boundary:raise ValueError('Target leakage beyond split')
        return result

    def record(self,name,predictions,details):
        result=dict(id=name,details=details,**self.reference.measure(predictions),independent_confirmation=False)
        folder=self.out/'predictions';folder.mkdir(exist_ok=True)
        for origin,p in predictions.items():np.save(folder/f'{name}_{origin}.npy',p)
        write(self.out/'evaluations'/f'{name}.json',result);self.records.append(result)
        log('RESULT',name,round(result['weighted_ratio'],8));return result

def fit_count(e,variant,origin,final=False):
    import lightgbm as lgb
    import psutil
    folder=e.out/'models'/variant['id']/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    if checkpoint(folder,'complete.json',e.ph):return folder
    params=dict(objective='multiclass',metric='None',num_class=len(variant['edges'])+2,num_leaves=31,max_depth=7,
        learning_rate=.05,min_data_in_leaf=100,bagging_fraction=.9,bagging_freq=1,feature_fraction=.8,
        lambda_l1=5.,lambda_l2=20.,seed=42,num_threads=CONFIG['threads'],deterministic=True,force_col_wise=True,verbosity=-1)
    def guard(env):
        if psutil.Process().memory_info().rss>CONFIG['memory_limit_gib']*2**30:raise MemoryError('Memory budget exceeded; checkpoints preserved')
    inner=checkpoint(folder,'inner.json',e.ph)
    if inner is None:
        train=e.dataset(origin,'inner',final);valid=e.dataset(origin,'valid',final)
        x,categories=encode(train,COLUMNS);xv,_=encode(valid,COLUMNS,categories)
        schema=schema_for(train,variant,categories)
        _,incumbent,refreshed=shadow(valid,origin,inner=True,final=final)
        r=valid.mean_28.to_numpy(float)*7;mask=(r>=.5)&(r<10)
        yv=valid.target.to_numpy(float)[mask];codes=e.reference.common.codes(valid.loc[mask])
        base_metrics=e.reference.common.metrics(yv,incumbent[mask],codes)
        ds=lgb.Dataset(x,label=labels(train.target.to_numpy(float),variant['edges']),
            weight=sample_weights(train,variant['weighting']),categorical_feature=CAT)
        dv=lgb.Dataset(xv,label=labels(valid.target.to_numpy(float),variant['edges']),reference=ds,categorical_feature=CAT)
        def evaluate(probability,data):
            count=shared.decode(probability,schema,3.);q=np.rint(.75*count+.25*refreshed)[mask]
            metrics=e.reference.common.metrics(yv,q,codes)
            score=sum(w*e.reference.common.ratio(metrics[k],base_metrics[k]) for w,k in zip(WEIGHTS,METRICS))
            return 'weekly_final_component_score',float(score),False
        log('FIT count inner',variant['id'],origin,len(train))
        model=lgb.train(params,ds,num_boost_round=CONFIG['max_count_iterations'],valid_sets=[dv],feval=evaluate,
            callbacks=[guard,lgb.early_stopping(CONFIG['early_stopping'],verbose=False)])
        iterations=int(model.best_iteration or model.current_iteration())
        temp=folder/'inner_model.txt.tmp';model.save_model(str(temp),num_iteration=iterations);temp.replace(folder/'inner_model.txt')
        write(folder/'inner_schema.json',schema)
        write(folder/'inner.json',dict(protocol_sha256=e.ph,iterations=iterations,
            files={n:sha(folder/n) for n in ['inner_model.txt','inner_schema.json']}))
        del train,valid,x,xv,ds,dv,model,incumbent,refreshed,yv;gc.collect();inner=read(folder/'inner.json')
    train=e.dataset(origin,'refit',final);x,categories=encode(train,COLUMNS);schema=schema_for(train,variant,categories)
    ds=lgb.Dataset(x,label=labels(train.target.to_numpy(float),variant['edges']),
        weight=sample_weights(train,variant['weighting']),categorical_feature=CAT)
    log('FIT count refit',variant['id'],origin,len(train),inner['iterations'])
    model=lgb.train(params,ds,num_boost_round=inner['iterations'],callbacks=[guard])
    temp=folder/'refit_model.txt.tmp';model.save_model(str(temp));temp.replace(folder/'refit_model.txt')
    write(folder/'refit_schema.json',schema)
    write(folder/'complete.json',dict(protocol_sha256=e.ph,iterations=inner['iterations'],
        files={n:sha(folder/n) for n in ['inner.json','inner_model.txt','inner_schema.json','refit_model.txt','refit_schema.json']}))
    del train,x,ds,model;gc.collect();return folder

def residual_apply(prefix,delta):
    limit=np.maximum(1.,.25*np.maximum(prefix,0.))
    return np.maximum(prefix+np.clip(delta,-limit,limit),0.)

def fit_residual(e,origin,final=False):
    import lightgbm as lgb
    folder=e.out/'models/residual'/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    if checkpoint(folder,'complete.json',e.ph):return folder
    valid=e.dataset(origin,'valid',final)
    features,prefix,_=shadow(valid,origin,inner=True,final=final,prefix=True)
    r=valid.mean_28.to_numpy(float)*7;mask=(r>=.5)&(r<10)
    f=features.loc[mask].reset_index(drop=True);actual=valid.target.to_numpy(float)[mask];base=prefix[mask]
    weeks=sorted(pd.to_datetime(f.week).unique())
    if len(weeks)!=9:raise ValueError('Residual meta-window must contain nine complete weeks')
    train_mask=pd.to_datetime(f.week)<=weeks[5];columns=list(COLUMNS)+SIDE
    x,categories=encode(f.loc[train_mask],columns);xv,_=encode(f.loc[~train_mask],columns,categories)
    residual=actual-base
    params=dict(objective='regression_l1',metric='None',num_leaves=15,max_depth=4,learning_rate=.03,
        min_data_in_leaf=200,lambda_l1=10.,lambda_l2=50.,feature_fraction=.8,seed=42,num_threads=6,
        deterministic=True,force_col_wise=True,verbosity=-1)
    ds=lgb.Dataset(x,label=residual[train_mask]);dv=lgb.Dataset(xv,label=residual[~train_mask],reference=ds)
    code=e.reference.common.codes(f.loc[~train_mask]);truth=actual[~train_mask];b=base[~train_mask]
    baseline=e.reference.common.metrics(truth,b,code)
    def evaluate(delta,data):
        q=np.rint(residual_apply(b,delta));metrics=e.reference.common.metrics(truth,q,code)
        score=sum(w*e.reference.common.ratio(metrics[k],baseline[k]) for w,k in zip(WEIGHTS,METRICS))
        return 'weekly_residual_score',float(score),False
    log('FIT residual inner',origin,'honest fixed-prefix forecasts',len(f))
    model=lgb.train(params,ds,num_boost_round=CONFIG['residual_max_iterations'],valid_sets=[dv],feval=evaluate,
        callbacks=[lgb.early_stopping(25,verbose=False)])
    iterations=int(model.best_iteration or model.current_iteration())
    x,categories=encode(f,columns);del ds,dv,xv,model;gc.collect()
    log('FIT residual refit',origin,len(f),iterations)
    model=lgb.train(params,lgb.Dataset(x,label=residual),num_boost_round=iterations)
    model.save_model(str(folder/'model.txt'))
    write(folder/'schema.json',dict(columns=columns,categories=categories,prefix_iterations=64,
        scope_min=.5,scope_max=10.,iterations=iterations,label_end=str((f.week+pd.Timedelta(days=6)).max().date()),
        source='first fixed 64 iterations of inner boosters; target rows absent from their gradient training; same fixed prefix used on inference',
        correction='clip(delta, -max(1,.25*prefix), max(1,.25*prefix)); floor quantity at zero'))
    write(folder/'complete.json',dict(protocol_sha256=e.ph,iterations=iterations,
        files={n:sha(folder/n) for n in ['model.txt','schema.json']}))
    return folder

def residual_prediction(folder,features):
    import lightgbm as lgb
    schema=read(Path(folder)/'schema.json');x,_=encode(features,schema['columns'],schema['categories'])
    return lgb.Booster(model_file=str(Path(folder)/'model.txt')).predict(x,num_threads=6)

def candidate_policies(screen):
    result=[shared.baseline_policy(),screen['best']['policy']]
    for value in [1.,3.,6.,12.]:
        p=copy.deepcopy(screen['best']['policy']);p['lambdas']=np.full((5,3),value).tolist();result.append(p)
    seen=set();out=[]
    for p in result:
        key=json.dumps(p,sort_keys=True)
        if key not in seen:out.append(p);seen.add(key)
    return out

def run(e):
    e.record('current_baseline',{o:c['base'] for o,c in e.reference.contexts.items()},dict(kind='baseline'))
    policy=e.screen['best']['policy']
    e.record('calibrated_existing',{o:shared.apply_policy(c,policy) for o,c in e.reference.contexts.items()},dict(kind='policy',policy=policy))
    policies=candidate_policies(e.screen)
    for variant in VARIANTS:
        loaded={}
        for origin,c in e.reference.contexts.items():
            folder=fit_count(e,variant,origin);loaded[origin]=shared.probabilities(folder,c['frame'])
        for p in policies:
            name=variant['id']+'_'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:10]
            prediction={o:shared.apply_policy(c,p,*loaded[o]) for o,c in e.reference.contexts.items()}
            e.record(name,prediction,dict(kind='count',variant=variant,policy=p))
    prefix_predictions={};corrected={}
    for origin,c in e.reference.contexts.items():
        features,prefix,_=shadow(c['frame'],origin,prefix=True)
        prefix_predictions[origin]=prefix
        folder=fit_residual(e,origin);delta=residual_prediction(folder,features)
        corrected[origin]=residual_apply(prefix,delta)
    for weight in [.1,.25,.5,1.]:
        for kind,source in [('prefix_control',prefix_predictions),('residual',corrected)]:
            predictions={}
            for origin,c in e.reference.contexts.items():
                r,h=c['cells'];mask=(r==1)|(r==2);p=c['base'].copy()
                p[mask]=np.rint((1-weight)*p[mask]+weight*source[origin][mask]);predictions[origin]=p
            e.record(f'{kind}_{weight}',predictions,dict(kind=kind,weight=weight))
    # Test combinations only after standalone components have complete forecasts.
    standalone=sorted(e.records,key=lambda r:r['weighted_ratio'])[:3]
    for i,a in enumerate(standalone):
        for b in standalone[i+1:]:
            for weight in [.25,.5,.75]:
                p={o:np.rint((1-weight)*np.load(e.out/'predictions'/f'{a["id"]}_{o}.npy',allow_pickle=False)+
                    weight*np.load(e.out/'predictions'/f'{b["id"]}_{o}.npy',allow_pickle=False)) for o in ORIGINS}
                name='blend_'+hashlib.sha256((a['id']+'|'+b['id']+'|'+str(weight)).encode()).hexdigest()[:12]
                e.record(name,p,dict(kind='blend',left=a['id'],right=b['id'],weight=weight))
    eligible=[r for r in e.records if r['worst_window_ratio']<=1.01]
    best=min(eligible,key=lambda r:(r['weighted_ratio'],r['worst_window_ratio'],r['id']))
    accepted=best['weighted_ratio']<1-1e-8 and best['details']['kind']!='baseline'
    write(e.out/'SELECTION.json',dict(best=best,accepted=accepted,records=sorted(e.records,key=lambda r:r['weighted_ratio']),
        metric_weights=dict(zip(METRICS,WEIGHTS)),independent_confirmation=False,production_changed=False))
    if accepted and CONFIG['final_if_better']:final_forecast(e,best)
    write(e.out/'summary.json',dict(completed=True,scope=list(range(1,8)),accepted=accepted,best=best,
        independent_confirmation=False,production_changed=False,final_dependencies_prepared=bool(accepted and CONFIG['final_if_better'])))
    package(e)

def final_forecast(e,best):
    from app.csv_io import read_csv_checked
    from app.features import truncate_history
    from app.pipeline import prepare_panel,feature_frame,postprocess_fallback
    from app.advanced_features import augment_advanced_features
    from app.predictor import Predictor
    from app.coverage import complete_required_forecast
    from app.submission import validate_forecast
    e.verify_inputs();log('PREP final candidate, production remains unchanged')
    s=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv');st=read_csv_checked(ROOT/'data/stock_dataset_hakaton_train.csv')
    s,st=truncate_history(s,st,'2026-07-26');model=Predictor(ROOT/'model').load().model
    panel=prepare_panel(s,st,'research',model.params,history_end='2026-07-26')
    f=feature_frame(panel,'2026-07-26','research',model_params=model.params);del panel;gc.collect()
    f=augment_advanced_features(s,st,f,'2026-07-26')
    probability,schema=shared.probabilities(shared.RUN/'models/count_fine/FINAL',f)
    current=model.predict(f);context=dict(frame=f,base=current,legacy=model.baseline.predict(f),
        refreshed=shared.component_values('2026-07-26',f,final=True),probability=probability,schema=schema,cells=shared.cells(f),decoded={})
    if not np.array_equal(shared.apply_policy(context,shared.baseline_policy()),current):raise ValueError('Final baseline replay differs')
    lookup={r['id']:r for r in e.records};lookup[best['id']]=best;memo={};dependencies={}
    def predict_record(record):
        if record['id'] in memo:return memo[record['id']]
        d=record['details'];kind=d['kind']
        if kind=='baseline':p=current
        elif kind=='policy':p=shared.apply_policy(context,d['policy'])
        elif kind=='count':
            folder=fit_count(e,d['variant'],'2026-07-26',final=True);prob,sch=shared.probabilities(folder,f)
            p=shared.apply_policy(context,d['policy'],prob,sch);dependencies[d['variant']['id']]=str(folder.relative_to(e.out))
        elif kind in ['residual','prefix_control']:
            features,prefix,_=shadow(f,'2026-07-26',final=True,prefix=True);q=prefix
            if kind=='residual':
                folder=fit_residual(e,'2026-07-26',final=True)
                q=residual_apply(prefix,residual_prediction(folder,features));dependencies['residual']=str(folder.relative_to(e.out))
            r,h=context['cells'];mask=(r==1)|(r==2);p=current.copy()
            p[mask]=np.rint((1-d['weight'])*p[mask]+d['weight']*q[mask])
        elif kind=='blend':p=np.rint((1-d['weight'])*predict_record(lookup[d['left']])+d['weight']*predict_record(lookup[d['right']]))
        else:raise ValueError('Unknown final graph component')
        memo[record['id']]=p;return p
    q=predict_record(best);observed=f[WK].copy();observed['week']=observed.week.dt.strftime('%Y-%m-%d');observed['quantity']=q
    base=read_csv_checked(ROOT/'forecast.csv')
    completed,pairs,_=complete_required_forecast(observed,s,'2026-07-26','2026-07-27',base[KEY],
        fallback_transform=lambda frame:postprocess_fallback(model,read(ROOT/'model/config.json'),frame))
    validate_forecast(completed,pairs,'2026-07-27');completed.to_csv(e.out/'candidate_forecast.csv',index=False)
    pairs.to_csv(e.out/'candidate_pairs.csv',index=False)
    parquet=e.out/'final_features.parquet';f.to_parquet(parquet,index=False)
    write(e.out/'FINAL_MODELS.json',dict(best=best,dependencies=dependencies,baseline_model_sha256=shared.MODEL_SHA,
        label_end='2026-07-26',production_changed=False,serving_integration_required=True,
        forecast_sha256=sha(e.out/'candidate_forecast.csv')))

def package(e):
    e.verify_inputs();export=e.out/'export';export.mkdir(exist_ok=True)
    for name in ['models','predictions','evaluations']:
        if (e.out/name).exists():shutil.copytree(e.out/name,export/name,dirs_exist_ok=True)
    for p in e.out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.parquet']:shutil.copy2(p,export/p.name)
    shutil.copy2(Path(__file__),export/Path(__file__).name)
    shutil.copy2(ROOT/'scripts/weekly_refinement_common.py',export/'weekly_refinement_common.py')
    shutil.copy2(e.reference.out/'SCREEN_RESULTS.json',export/'SCREEN_RESULTS.json')
    files={p.relative_to(export).as_posix():sha(p) for p in export.rglob('*') if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    write(export/'OUTPUT_MANIFEST.json',dict(protocol_sha256=e.ph,files=files,production_changed=False))
    archive=e.out/'weekly_refinement_results.zip';temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in export.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(export))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip():raise ValueError('Corrupt result archive')
    temporary.replace(archive);log('DONE. SEND THIS FILE:',archive)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    parser.add_argument('--screen',action='store_true',help='No model fitting, only existing-weight policy probes')
    args=parser.parse_args()
    if environment(args.check or args.screen):return
    initialize();reference=shared.Contexts()
    global COLUMNS
    COLUMNS=list(reference.contexts[ORIGINS[0]]['schema']['columns'])
    if args.check:
        for origin in ORIGINS:
            f=reference.frame(origin,reference.schedules[origin]['valid'])
            scope=(f.mean_28*7>=.5)&(f.mean_28*7<10)
            features,prefix,_=shadow(f,origin,inner=True,prefix=True)
            assert scope.any() and np.isfinite(prefix).all() and set(COLUMNS+SIDE)<=set(features)
            for variant in VARIANTS:
                train=f.rename(columns={'actual':'target'})
                w=sample_weights(train,variant['weighting']);assert np.isfinite(w).all() and (w>0).all()
        screen=reference.screen();e=Experiment(reference,screen)
        valid=e.dataset('2026-07-26','valid',final=True)
        assert (valid.week+pd.Timedelta(days=6)).max()==pd.Timestamp('2026-07-26')
        features,prefix,_=shadow(valid,'2026-07-26',inner=True,final=True,prefix=True)
        assert np.isfinite(prefix).all() and set(COLUMNS+SIDE)<=set(features)
        partial=reference.frame('2026-07-26','2026-06-28-True.parquet',final=True)
        assert (partial.week+pd.Timedelta(days=6)).max()<=pd.Timestamp('2026-07-26')
        x,categories=encode(partial.iloc[:700],COLUMNS)
        import lightgbm as lgb
        for variant in VARIANTS:
            schema=schema_for(partial,variant,categories)
            lab=labels(partial.target.to_numpy(float),variant['edges'])
            assert lab.max()<len(schema['representatives'])
        dataset=lgb.Dataset(x,label=partial.target.iloc[:700],categorical_feature=CAT,
            params=dict(min_data_in_leaf=5,verbosity=-1)).construct()
        assert dataset.num_feature()==len(COLUMNS)
        log('CHECK PASSED. No models trained.');return
    screen=reference.screen();log('SCREEN',screen['candidate_count'],'best ratio',screen['best']['weighted_ratio'])
    if args.screen:return
    e=Experiment(reference,screen)
    with lock(e.out/'.run.lock'):
        archive=e.out/'weekly_refinement_results.zip'
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
                if manifest['protocol_sha256']!=e.ph:raise ValueError('Result protocol differs')
                for name,h in manifest['files'].items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Result file differs: '+name)
            log('ALREADY COMPLETED. SEND:',archive);return
        try:run(e)
        except BaseException:
            (e.out/'LAST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            log('STOPPED. Checkpoints preserved. Send LAST_ERROR.txt from',e.out);raise

if __name__=='__main__':main()
