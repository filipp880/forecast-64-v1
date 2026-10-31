"""PyCharm round 2: daily predictions trained on weekly-sum loss, with matched daily control.

No model training is performed by --check. Production weights/CSV are never replaced.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('daily_round1_shared',ROOT/'train_daily_to_weekly.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
read,write,sha,log,parquet=old.read,old.write,old.sha,old.log,old.parquet
CONFIG=dict(seed=42,threads=6,max_iterations=250,early_stopping=30,memory_limit_gib=8.,
    blend_weights=[.05,.1,.25],final_if_better=True,early_stop_weekly_gate=.5)
VARIANTS=[dict(id='daily_poisson_control',loss='poisson',weekly_weight=0.,daily_weight=1.),
    dict(id='weekly_poisson_aux',loss='poisson',weekly_weight=1.,daily_weight=.1),
    dict(id='weekly_huber_aux',loss='huber',weekly_weight=1.,daily_weight=.05,delta=2.)]
EXTRA=['stock_negative_share_90','stock_reserved','last_week','mean_week_4','mean_week_8','mean_week_13',
    'zero_week_share_13','mean_all','history_days','ewma_week_4','ewma_week_13','std_week_13','positive_cv2_13',
    'adi_13','week_trend_4_13','tsb_week','sba_week','article_week_mean','warehouse_week_mean','category_week_mean','category_share']
WEEK_KEY=['week','warehouse_name','article']
ORIGINS=old.ORIGINS;METRICS=old.METRICS;WEIGHTS=old.WEIGHTS

def environment(check):
    """Always relaunch THIS round, never the imported round1 entry point."""
    import importlib.metadata
    import os
    import subprocess
    if not (3,10)<=sys.version_info[:2]<=(3,12):raise RuntimeError('Use Python 3.10-3.12')
    missing=[]
    for name,version in old.VERSIONS.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual='missing'
        if actual!=version:missing.append((name,actual,version))
    if not missing:return False
    entry=str(Path(__file__).resolve())
    existing=ROOT/'.venv-all40'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if existing.exists() and Path(sys.executable).resolve()!=existing.resolve():
        subprocess.check_call([str(existing),entry,*sys.argv[1:]]);return True
    if check:raise RuntimeError('Missing dependencies: '+repr(missing))
    target=ROOT/'.venv-daily-weekly';python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():subprocess.check_call([sys.executable,'-m','venv',str(target)])
    subprocess.check_call([str(python),'-m','pip','install',*[f'{k}=={v}' for k,v in old.VERSIONS.items()]])
    if python.resolve()==Path(sys.executable).resolve():raise RuntimeError('Environment setup did not resolve versions')
    subprocess.check_call([str(python),entry,*sys.argv[1:]]);return True

def initialize():
    global np,pd,CAT_FEATURES,KEY,DAILY_COLUMNS,COLUMNS,aggregate_daily
    import numpy as np
    import pandas as pd
    from app.features import CAT_FEATURES,KEY
    from app.daily_bridge import DAILY_COLUMNS,aggregate_daily
    COLUMNS=list(dict.fromkeys(DAILY_COLUMNS+EXTRA+[f'bridge_lag_{i}' for i in range(1,14)]+
        ['bridge_support_13','bridge_event_rate_13','bridge_week_horizon']))
    for name,value in [('np',np),('pd',pd),('CAT_FEATURES',CAT_FEATURES),('KEY',KEY),
            ('DAILY_COLUMNS',DAILY_COLUMNS),('aggregate_daily',aggregate_daily)]:setattr(old,name,value)

def group_targets(frame):
    """Every forecast-origin/pair/week has exactly seven consecutive rows."""
    if not len(frame) or len(frame)%7:raise ValueError('Incomplete seven-day training blocks')
    for column in ['feature_origin','week',*KEY]:
        a=frame[column].to_numpy().reshape(-1,7)
        if not (a==a[:,:1]).all():raise ValueError('Mixed origins/pairs/weeks inside a training block')
    dates=pd.to_datetime(frame.date).to_numpy().reshape(-1,7)
    if not ((dates-dates[:,:1])/np.timedelta64(1,'D')==np.arange(7)).all():raise ValueError('Non-consecutive daily targets')
    if (pd.DatetimeIndex(dates[:,0]).weekday!=0).any():raise ValueError('Weekly target must start on Monday')
    return frame.target.to_numpy(float).reshape(-1,7).sum(axis=1)

def offset(frame):
    # Known demand and weekday profile give a non-negative, past-only initial mean.
    return np.log(np.maximum(.5*frame.mean_28.to_numpy(float)+.5*frame.dow_mean.to_numpy(float),.02))

def gradients(z,y,variant):
    """Exact gradient of the declared loss; positive curvature approximation.

    Cross-day Hessian entries are not representable by LightGBM. Poisson uses
    a Fisher row-sum approximation; Huber uses a positive block majorizer.
    """
    mu=np.exp(np.clip(np.asarray(z,float),-15,15));y=np.asarray(y,float)
    expected=mu.reshape(-1,7).sum(axis=1);actual=y.reshape(-1,7).sum(axis=1)
    a=variant['weekly_weight'];b=variant['daily_weight']
    if variant['loss']=='poisson':
        term=1-actual/np.maximum(expected,1e-12)
        grad=a*mu*np.repeat(term,7)+b*(mu-y)
        hess=(a+b)*mu
    else:
        delta=variant['delta'];error=expected-actual;t=np.sqrt(1+(error/delta)**2)
        derivative=error/t;curvature=1/t**3
        grad=a*mu*np.repeat(derivative,7)+b*(mu-y)
        hess=a*mu*np.repeat(np.abs(derivative)+expected*curvature,7)+b*mu
    return grad,np.maximum(hess,1e-6)

def loss_value(z,y,variant):
    mu=np.exp(np.clip(np.asarray(z,float),-15,15));y=np.asarray(y,float)
    expected=mu.reshape(-1,7).sum(axis=1);actual=y.reshape(-1,7).sum(axis=1)
    daily=(mu-y*np.log(mu)).sum()
    if variant['loss']=='poisson':weekly=(expected-actual*np.log(expected)).sum()
    else:
        delta=variant['delta'];weekly=(delta**2*(np.sqrt(1+((expected-actual)/delta)**2)-1)).sum()
    return float(variant['weekly_weight']*weekly+variant['daily_weight']*daily)

class Experiment:
    def __init__(self):
        self.first=old.Experiment()
        expected=ROOT/'artifacts/daily-to-weekly-0eba813e6d26'
        if self.first.out!=expected:raise ValueError('Round1 inputs/source changed; review before reusing its results')
        archive=expected/'daily_to_weekly_results.zip'
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
            if manifest['protocol_sha256']!=self.first.ph:raise ValueError('Source result protocol differs')
            for name,h in manifest['files'].items():
                if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Source archive file differs: '+name)
        files=dict(self.first.protocol['files'])
        files[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        files[str(archive.relative_to(ROOT))]=sha(archive)
        event_root=ROOT/'artifacts/all40-0a385edab6b1/models/core_event'
        for origin in [*ORIGINS,'FINAL']:
            for phase in ['inner','refit']:
                for suffix in ['model.txt','schema.json']:
                    p=event_root/origin/f'{phase}_{suffix}';files[str(p.relative_to(ROOT))]=sha(p)
        self.protocol=dict(version=1,config=CONFIG,variants=VARIANTS,files=files,columns=COLUMNS,
            purpose='Matched enriched daily control vs daily outputs optimized for weekly totals',
            weekly_loss_groups='forecast_origin + warehouse + article + full Monday-Sunday week',
            offset='log(max(0.5 * past mean28 + 0.5 * past weekday mean, 0.02))',
            positive_hessian_approximation=True,independent_confirmation=False,
            early_stopping='five weekly metrics on an earlier nine-week window, fixed weekly event gate 0.5',
            weekly_event_source='verified existing classifiers; inner model only on later validation window, refit only on later outer window')
        identity=hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()[:12]
        self.out=ROOT/'artifacts'/('daily-weekloss-'+identity);self.out.mkdir(parents=True,exist_ok=True)
        p=self.out/'protocol.json'
        if p.exists() and read(p)!=self.protocol:raise ValueError('Frozen protocol changed')
        if not p.exists():write(p,self.protocol)
        self.ph=sha(p);self.records=[];log('ROUND2',self.out)

    def weekly_source(self,origin,name,final=False):
        path=(ROOT/'artifacts/improvements-20261028-r1/final-cache'/name if final else
              ROOT/'artifacts/ideas-20261028-r2'/origin/'cache'/name)
        return pd.read_parquet(path)

    def event_probability(self,origin,weekly,inner=False,final=False):
        from app.models.all40_model import NativePart
        folder=ROOT/'artifacts/all40-0a385edab6b1/models/core_event'/('FINAL' if final else origin)
        prefix='inner' if inner else 'refit'
        schema=read(folder/f'{prefix}_schema.json')
        part=NativePart((folder/f'{prefix}_model.txt').read_text(encoding='utf-8'),schema,CONFIG['threads'])
        return part.predict(weekly)

    def enrich(self,daily,weekly,cutoff):
        d=daily.copy();w=weekly.copy()
        for f in [d,w]:
            f['week']=pd.to_datetime(f.week)
            for k in KEY:f[k]=f[k].astype(str)
        index=pd.MultiIndex.from_frame(w[WEEK_KEY]);codes=index.get_indexer(pd.MultiIndex.from_frame(d[WEEK_KEY]))
        if (codes<0).any() or not index.is_unique:raise ValueError('Daily/weekly feature alignment differs')
        for c in EXTRA:d[c]=w[c].to_numpy(dtype='float32')[codes]
        history=self.first.ensure_history();pairs=pd.MultiIndex.from_frame(w[KEY].drop_duplicates())
        paircode=pairs.get_indexer(pd.MultiIndex.from_frame(d[KEY]))
        cut=pd.Timestamp(cutoff);end=cut-pd.Timedelta(days=(cut.weekday()+1)%7)
        dates=pd.date_range(end-pd.Timedelta(days=90),end)
        values=history.daily.reindex(index=pairs,columns=dates).fillna(0).to_numpy(float).reshape(len(pairs),13,7).sum(axis=2)
        starts=pd.date_range(dates[0],periods=13,freq='7D')
        first=history.first.reindex(pairs).to_numpy(dtype='datetime64[ns]')
        known=(starts.to_numpy()[None,:]>=first[:,None])&(starts.to_numpy()[None,:]>=history.start.to_datetime64())
        values[~known]=np.nan
        for i in range(1,14):d[f'bridge_lag_{i}']=values[paircode,-i].astype('float32')
        support=known.sum(axis=1);rates=np.divide((values>0).sum(axis=1),support,out=np.full(len(pairs),np.nan),where=support>0)
        d['bridge_support_13']=support[paircode].astype('float32');d['bridge_event_rate_13']=rates[paircode].astype('float32')
        d['bridge_week_horizon']=((d.week-(cut-pd.Timedelta(days=cut.weekday())+pd.Timedelta(days=7))).dt.days/7+1).astype('float32')
        keep=list(dict.fromkeys(['date','week','feature_origin',*COLUMNS]+(['target'] if 'target' in d else [])))
        d=d[keep]
        for c in CAT_FEATURES:d[c]=d[c].astype('category')
        return d

    def frame(self,origin,name,final=False,evaluation=False):
        path=self.out/'features'/f'{"FINAL" if final else origin}_{name}'
        if path.exists():return pd.read_parquet(path)
        daily=self.first.frame(origin,name,final=final,evaluation=evaluation)
        if daily is None:return None
        weekly=self.weekly_source(origin,name,final)
        index=pd.MultiIndex.from_frame(daily[WEEK_KEY].drop_duplicates())
        weekly=weekly.loc[pd.MultiIndex.from_frame(weekly[WEEK_KEY]).isin(index)].copy()
        d=self.enrich(daily,weekly,name[:10]);group_targets(d);parquet(path,d);return d

    def dataset(self,origin,split,final=False):
        if final:
            boundary=pd.Timestamp('2026-05-24' if split=='inner' else '2026-07-26')
            names=list(self.first.final_manifest) if split!='valid' else ['2026-05-24-False.parquet']
            source='2026-05-24' if split=='valid' else origin
        else:
            schedule=self.first.schedules[origin];names=schedule[split]
            if isinstance(names,str):names=[names]
            boundary=pd.Timestamp(schedule['inner_origin'] if split=='inner' else origin);source=origin
        frames=[]
        for name in names:
            f=self.frame(source,name,final=final and split!='valid',evaluation=final and split=='valid')
            if f is None:continue
            f=f.loc[f.week+pd.Timedelta(days=6)<=boundary].copy()
            if len(f):frames.append(f)
        if not frames:raise ValueError('No legal training rows')
        # Align categorical domains using the permitted training frames only.
        for c in CAT_FEATURES:
            domain=sorted(set().union(*(set(f[c].astype(str).unique()) for f in frames)))
            for f in frames:f[c]=pd.Categorical(f[c].astype(str),categories=domain)
        result=pd.concat(frames,ignore_index=True);del frames;gc.collect();group_targets(result)
        if result.date.max()>boundary:raise ValueError('Target leakage beyond split')
        return result

    def validation_weekly(self,origin,final=False):
        source='2026-05-24' if final else origin
        name='2026-05-24-False.parquet' if final else self.first.schedules[origin]['valid']
        return self.weekly_source(source,name)

    def verify_inputs(self):
        for name,h in self.protocol['files'].items():
            if sha(ROOT/name)!=h:raise ValueError('An input changed: '+name)

def encode(frame,categories=None):
    x=frame[COLUMNS].copy()
    if categories is None:categories={c:sorted(x[c].astype(str).unique()) for c in CAT_FEATURES}
    for c in CAT_FEATURES:x[c]=pd.Categorical(x[c].astype(str),categories=categories[c])
    return x,categories

def fit_model(e,variant,origin,final=False):
    import lightgbm as lgb
    import psutil
    folder=e.out/'models'/variant['id']/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    if old.checkpoint(folder,'complete.json',e.ph):return folder
    params=dict(metric='None',num_leaves=31,max_depth=7,learning_rate=.05,min_data_in_leaf=100,
        feature_fraction=.8,bagging_fraction=1.,lambda_l1=5.,lambda_l2=20.,max_delta_step=1.,
        boost_from_average=False,seed=42,num_threads=CONFIG['threads'],deterministic=True,force_col_wise=True,verbosity=-1)
    # No row subsampling: each objective evaluation must see all seven days of every group.
    def guard(env):
        if psutil.Process().memory_info().rss>CONFIG['memory_limit_gib']*2**30:raise MemoryError('Memory budget exceeded; checkpoints preserved')
    inner=old.checkpoint(folder,'inner.json',e.ph)
    if inner is None:
        train=e.dataset(origin,'inner',final);valid=e.dataset(origin,'valid',final)
        y=train.target.to_numpy(float);base=offset(train);vy=group_targets(valid);vbase=offset(valid)
        vw=e.validation_weekly(origin,final)
        first_rows=valid.iloc[::7]
        ix=pd.MultiIndex.from_frame(vw[WEEK_KEY]);lookup=ix.get_indexer(pd.MultiIndex.from_frame(first_rows[WEEK_KEY]))
        if (lookup<0).any():raise ValueError('Validation keys differ')
        event=e.event_probability(origin,vw,inner=True,final=final)[lookup]
        codes=e.first.common.codes(first_rows)
        x,categories=encode(train);xv,_=encode(valid,categories)
        ds=lgb.Dataset(x,label=y,categorical_feature=CAT_FEATURES);dv=lgb.Dataset(xv,label=valid.target.to_numpy(float),reference=ds,categorical_feature=CAT_FEATURES)
        def objective(raw,data):return gradients(raw+base,y,variant)
        def evaluate(raw,data):
            q=np.exp(np.clip(raw+vbase,-15,15)).reshape(-1,7).sum(axis=1)
            q=np.rint(np.where(event>=CONFIG['early_stop_weekly_gate'],q,0.))
            metrics=e.first.common.metrics(vy,q,codes)
            constants=[60.,50.,3.,75.,.5]
            score=sum(w*metrics[k]/c for w,k,c in zip(WEIGHTS,METRICS,constants))
            return 'weekly_five_error_proxy',float(score),False
        log('FIT weekly-aware inner',variant['id'],origin,len(train),'daily rows')
        model=lgb.train({**params,'objective':objective},ds,num_boost_round=CONFIG['max_iterations'],
            valid_sets=[dv],feval=evaluate,callbacks=[guard,lgb.early_stopping(CONFIG['early_stopping'],verbose=False)])
        iterations=int(model.best_iteration or model.current_iteration())
        temporary=folder/'inner_model.txt.tmp';model.save_model(str(temporary),num_iteration=iterations);temporary.replace(folder/'inner_model.txt')
        write(folder/'inner_schema.json',dict(columns=COLUMNS,categories=categories,variant=variant,
            prediction='exp(clip(raw_booster_score + past_only_offset, -15, 15))',
            target_end=str(train.date.max().date()),validation_end=str(valid.date.max().date()),weekly_early_stopping=True))
        write(folder/'inner.json',dict(protocol_sha256=e.ph,iterations=iterations,
            files={n:sha(folder/n) for n in ['inner_model.txt','inner_schema.json']}))
        del train,valid,x,xv,y,base,vy,vbase,ds,dv,model;gc.collect();inner=read(folder/'inner.json')
    train=e.dataset(origin,'refit',final);y=train.target.to_numpy(float);base=offset(train)
    x,categories=encode(train);ds=lgb.Dataset(x,label=y,categorical_feature=CAT_FEATURES)
    def objective(raw,data):return gradients(raw+base,y,variant)
    log('FIT weekly-aware refit',variant['id'],origin,len(train),'iterations',inner['iterations'])
    model=lgb.train({**params,'objective':objective},ds,num_boost_round=inner['iterations'],callbacks=[guard])
    temporary=folder/'refit_model.txt.tmp';model.save_model(str(temporary));temporary.replace(folder/'refit_model.txt')
    write(folder/'refit_schema.json',dict(columns=COLUMNS,categories=categories,variant=variant,
        prediction='exp(clip(raw_booster_score + past_only_offset, -15, 15))',target_end=str(train.date.max().date()),training_rows=len(train)))
    write(folder/'complete.json',dict(protocol_sha256=e.ph,iterations=inner['iterations'],
        files={n:sha(folder/n) for n in ['inner.json','inner_model.txt','inner_schema.json','refit_model.txt','refit_schema.json']}))
    del train,x,y,base,ds,model;gc.collect();return folder

def predict_model(folder,frame):
    import lightgbm as lgb
    schema=read(folder/'refit_schema.json');x,_=encode(frame,schema['categories'])
    model=lgb.Booster(model_file=str(folder/'refit_model.txt'))
    raw=model.predict(x,raw_score=True,num_threads=CONFIG['threads'])
    return np.exp(np.clip(raw+offset(frame),-15,15))

def quantity(weekly_sum,event,policy):
    q=weekly_sum.copy()
    if policy['gate'] is not None:q=np.where(event>=policy['gate'],q,0.)
    return np.rint(q) if policy['round_int'] else q

def evaluate(e,name,predictions,details,loaded):
    return old.evaluate(e,name,predictions,details,loaded)

def run(e):
    loaded={};all_predictions={};events={}
    for origin in ORIGINS:
        weekly,_,_=e.first.common.load(e.first.bundle,origin)
        base=np.load(e.first.out/'predictions'/f'weekly_baseline_{origin}.npy',allow_pickle=False)
        loaded[origin]=(weekly,base)
        d=e.frame(origin,e.first.schedules[origin]['test'],evaluation=True)
        w=e.weekly_source(origin,e.first.schedules[origin]['test'])
        event=e.event_probability(origin,w)
        mapping=pd.MultiIndex.from_frame(w[WEEK_KEY]).get_indexer(pd.MultiIndex.from_frame(weekly[WEEK_KEY]))
        if (mapping<0).any():raise ValueError('Outer evaluation keys differ')
        events[origin]=event[mapping]
        parquet(e.out/'evaluation_data'/f'{origin}_daily.parquet',d)
        parquet(e.out/'evaluation_data'/f'{origin}_weekly.parquet',weekly[['week',*KEY,'actual']])
        np.save(e.out/'evaluation_data'/f'{origin}_weekly_event.npy',events[origin])
        for variant in VARIANTS:
            folder=fit_model(e,variant,origin);raw=predict_model(folder,d)
            all_predictions[(variant['id'],origin)]=aggregate_daily(d,raw,weekly)
            np.save(e.out/'evaluation_data'/f'{origin}_{variant["id"]}_daily_prediction.npy',raw)
    # old.evaluate uses e.common, whose formula and ratio normalization are frozen.
    evaluate(e,'weekly_baseline',{o:loaded[o][1] for o in ORIGINS},dict(kind='baseline'),loaded)
    for variant in VARIANTS:
        for gate in [None,.35,.5,.65]:
            for rounded in [False,True]:
                policy=dict(gate=gate,round_int=rounded)
                p={o:quantity(all_predictions[(variant['id'],o)],events[o],policy) for o in ORIGINS}
                evaluate(e,f'{variant["id"]}_g{gate}_r{int(rounded)}',p,
                    dict(kind='daily_weekloss',variant=variant,policy=policy),loaded)
    solo=sorted([r for r in e.records if r['details']['kind']=='daily_weekloss'],key=lambda r:r['weighted_ratio'])
    # Compare the best recipe from each loss family, including weak daily admixtures.
    chosen=[next(r for r in solo if r['details']['variant']['id']==v['id']) for v in VARIANTS]
    for r in chosen:
        for weight in CONFIG['blend_weights']:
            for scope in ['all','regular','low']:
                p={}
                for origin,(weekly,base) in loaded.items():
                    q=np.load(e.out/'predictions'/f'{r["id"]}_{origin}.npy',allow_pickle=False)
                    demand=weekly.mean_28.to_numpy(float)*7
                    mask=np.ones(len(base),bool) if scope=='all' else (demand>=2) if scope=='regular' else ((demand>=.5)&(demand<10))
                    value=base.copy();value[mask]=np.rint((1-weight)*base[mask]+weight*q[mask]);p[origin]=value
                evaluate(e,f'blend_{r["id"]}_{weight}_{scope}',p,
                    dict(kind='blend',variant=r['details']['variant'],policy=r['details']['policy'],weight=weight,scope=scope),loaded)
    eligible=[r for r in e.records if r['worst_window_ratio']<=1.01]
    best=min(eligible,key=lambda r:(r['weighted_ratio'],r['worst_window_ratio'],r['id']))
    accepted=best['details']['kind']!='baseline' and best['weighted_ratio']<1-1e-8
    write(e.out/'SELECTION.json',dict(best=best,accepted=accepted,records=sorted(e.records,key=lambda r:r['weighted_ratio']),
        independent_confirmation=False,control_description='identical enriched features and trees; daily Poisson vs weekly Poisson/Huber with daily auxiliary loss'))
    if accepted and CONFIG['final_if_better']:
        e.verify_inputs();folder=fit_model(e,best['details']['variant'],'2026-07-26',final=True)
        final_forecast(e,folder,best['details'])
    write(e.out/'summary.json',dict(completed=True,accepted=accepted,best=best,production_changed=False,
        final_models_trained=bool(accepted and CONFIG['final_if_better']),independent_confirmation=False))
    package(e)

def final_forecast(e,folder,details):
    from app.csv_io import read_csv_checked
    from app.features import build_panel,truncate_history
    from app.pipeline import feature_frame,prepare_panel,postprocess_fallback
    from app.predictor import Predictor
    from app.coverage import complete_required_forecast
    from app.submission import validate_forecast
    s=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv');st=read_csv_checked(ROOT/'data/stock_dataset_hakaton_train.csv')
    s,st=truncate_history(s,st,'2026-07-26');champion=Predictor(ROOT/'model').load()
    panel=prepare_panel(s,st,'research',champion.model.params,history_end='2026-07-26')
    weekly=feature_frame(panel,'2026-07-26','research',model_params=champion.model.params);del panel;gc.collect()
    daily=e.first.ensure_history().expand(weekly,'2026-07-26');daily=e.enrich(daily,weekly,'2026-07-26')
    raw=predict_model(folder,daily);total=aggregate_daily(daily,raw,weekly)
    event=e.event_probability('2026-07-26',weekly,final=True)
    q=quantity(total,event,details['policy']);keys=['week',*KEY]
    observed=weekly[keys].copy();observed['week']=observed.week.dt.strftime('%Y-%m-%d');observed['quantity']=q
    base=read_csv_checked(ROOT/'forecast.csv')
    completed,pairs,_=complete_required_forecast(observed,s,'2026-07-26','2026-07-27',base[KEY],
        fallback_transform=lambda f:postprocess_fallback(champion.model,champion.config,f))
    if details['kind']=='blend':
        mix=completed.merge(base,on=keys,suffixes=('_daily','_weekly'),validate='one_to_one')
        means=weekly.set_index(KEY).mean_28.groupby(level=KEY).first()*7
        demand=pd.MultiIndex.from_frame(mix[KEY]).map(means).to_numpy(float)
        mask=np.ones(len(mix),bool) if details['scope']=='all' else (demand>=2) if details['scope']=='regular' else ((demand>=.5)&(demand<10))
        result=mix.quantity_weekly.to_numpy(float).copy()
        result[mask]=np.rint((1-details['weight'])*result[mask]+details['weight']*mix.quantity_daily.to_numpy(float)[mask])
        completed=mix[keys].copy();completed['quantity']=result
    validate_forecast(completed,pairs,'2026-07-27');completed.to_csv(e.out/'daily_candidate_forecast.csv',index=False)
    pairs.to_csv(e.out/'daily_candidate_pairs.csv',index=False);parquet(e.out/'evaluation_data/final_daily.parquet',daily)
    # The weekly event model is a required dependency, never omitted from export.
    event_source=ROOT/'artifacts/all40-0a385edab6b1/models/core_event/FINAL'
    target=e.out/'weekly_event_FINAL';target.mkdir(exist_ok=True)
    for n in ['refit_model.txt','refit_schema.json']:shutil.copy2(event_source/n,target/n)
    write(e.out/'FINAL_MODELS.json',dict(details=details,label_end='2026-07-26',weekly_event_model='weekly_event_FINAL',
        quantity_model=str(folder.relative_to(e.out)),forecast_sha256=sha(e.out/'daily_candidate_forecast.csv'),
        production_changed=False,serving_integration_required=True))

def package(e):
    e.verify_inputs();export=e.out/'export';export.mkdir(exist_ok=True)
    for n in ['models','predictions','evaluations','evaluation_data','weekly_event_FINAL']:
        if (e.out/n).exists():shutil.copytree(e.out/n,export/n,dirs_exist_ok=True)
    for p in e.out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv']:shutil.copy2(p,export/p.name)
    shutil.copy2(Path(__file__),export/Path(__file__).name)
    files={p.relative_to(export).as_posix():sha(p) for p in export.rglob('*') if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    write(export/'OUTPUT_MANIFEST.json',dict(protocol_sha256=e.ph,files=files,production_changed=False))
    archive=e.out/'daily_weekloss_results.zip';temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in export.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(export))
    with zipfile.ZipFile(temp) as z:
        if z.testzip():raise ValueError('Corrupt result archive')
    temp.replace(archive);log('DONE. SEND THIS FILE:',archive)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if environment(args.check):return
    initialize();e=Experiment();e.common=e.first.common
    if args.check:
        for origin in ORIGINS:
            f=e.frame(origin,e.first.schedules[origin]['test'],evaluation=True);assert set(COLUMNS)<=set(f)
            w=e.weekly_source(origin,e.first.schedules[origin]['test'])
            probability=e.event_probability(origin,w)
            assert len(probability)==len(f)//7 and np.isfinite(probability).all()
        valid=e.dataset('2026-07-26','valid',final=True)
        assert valid.date.max()==pd.Timestamp('2026-07-26')
        partial=e.frame('2026-07-26','2026-06-28-True.parquet',final=True)
        assert partial.date.max()==pd.Timestamp('2026-07-26') and partial.h.max()==28
        import lightgbm as lgb
        sample=valid.iloc[:700].copy();group_targets(sample)
        x,categories=encode(sample)
        dataset=lgb.Dataset(x,label=sample.target.to_numpy(float),categorical_feature=CAT_FEATURES,
            params=dict(min_data_in_leaf=5,verbosity=-1)).construct()
        assert dataset.num_feature()==len(COLUMNS)
        for variant in VARIANTS:
            g,h=gradients(offset(sample),sample.target.to_numpy(float),variant)
            assert np.isfinite(g).all() and np.isfinite(h).all() and (h>0).all()
        log('CHECK PASSED. No model fitting executed.');return
    with old.lock(e.out/'.run.lock'):
        archive=e.out/'daily_weekloss_results.zip'
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
                if manifest['protocol_sha256']!=e.ph:raise ValueError('Result protocol differs')
                for name,h in manifest['files'].items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Result changed: '+name)
            log('ALREADY COMPLETED. SEND:',archive);return
        try:run(e)
        except BaseException:
            (e.out/'LAST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            log('STOPPED. Checkpoints preserved. Send LAST_ERROR.txt from',e.out);raise

if __name__=='__main__':main()
