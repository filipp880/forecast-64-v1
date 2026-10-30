"""One user-run entry point for the forty-item weekly forecasting research plan.

Run in PyCharm from this repository. --check never fits models.
This script DOES NOT replace model/ or forecast.csv and does not use OpenAI APIs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

# Freeze these settings before the first launch. Changing them creates a new run.
CONFIG = dict(seed=42, threads=6, max_iterations=350, count_iterations=180,
              early_stopping=35, memory_limit_gib=8., cnn_epochs=8,
              cnn_batch_size=1024, final_refit=True,
              auto_install=True, feature_top_k=[40,80], dense_step_days=14)
VERSIONS = {'numpy':'2.1.1','pandas':'2.3.3','scikit-learn':'1.7.2','scipy':'1.15.3',
            'lightgbm':'4.7.0','pyarrow':'25.0.1','threadpoolctl':'3.6.0','psutil':'7.2.2','torch':'2.13.0'}
ORIGINS = ['2026-01-18','2026-03-22']
METRICS = ['smape','wape','rmse','bias','rmsle']
WEIGHTS = [.5,.2,.15,.1,.05]
KEY = ['warehouse_name','article']
ROWKEY = ['week',*KEY]
CAT = KEY+['category_lvl_1','category_lvl_2','category_lvl_3','characteristic']
BASE_MODEL_SHA = '278347022a5d2b2dcf92145155f4a9f728a1e5996a7b89517527bfaddd2d318e'
DENSE_RESUME_SOURCE_SHA = '6a3db03c1cd1d904638fe014eafd92289ac9f0d10b1107540ab412d4c72ceb62'


def root_directory():
    for p in [Path(__file__).resolve().parent,*Path(__file__).resolve().parents]:
        if (p/'app/features.py').is_file() and (p/'model/config.json').is_file():return p
    raise RuntimeError('Place train_all_improvements.py in the forecast repository root.')


ROOT=root_directory()
sys.path.insert(0,str(ROOT))


def log(*args):print(datetime.now().strftime('%H:%M:%S'),*args,flush=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def write_frame(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    frame.to_parquet(temporary,index=False);temporary.replace(path)


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def dependencies(check):
    if not ((3,10)<=sys.version_info[:2]<=(3,12)):raise RuntimeError('Use Python 3.10–3.12.')
    missing=[]
    for name,version in VERSIONS.items():
        try:actual=importlib.metadata.version(name).split('+')[0]
        except importlib.metadata.PackageNotFoundError:actual='missing'
        if actual!=version:missing.append((name,version,actual))
    if not missing:return False
    if check or not CONFIG['auto_install']:
        raise RuntimeError('Missing/incompatible dependencies: '+repr(missing)+'. Normal Run can install them in an isolated .venv-all40.')
    env=ROOT/'.venv-all40';python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():
        log('Creating isolated environment',env)
        subprocess.check_call([sys.executable,'-m','venv',str(env)])
    subprocess.check_call([str(python),'-m','pip','install','--upgrade','pip'])
    plain=[f'{n}=={v}' for n,v in VERSIONS.items() if n!='torch']
    subprocess.check_call([str(python),'-m','pip','install',*plain])
    subprocess.check_call([str(python),'-m','pip','install','torch=='+VERSIONS['torch'],
                           '--index-url','https://download.pytorch.org/whl/cpu'])
    if Path(sys.executable).resolve()==python.resolve():raise RuntimeError('Environment setup did not resolve package versions.')
    code=subprocess.call([str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    if code:raise RuntimeError('Child run failed; inspect the last error above.')
    return True


@contextmanager
def run_lock(path):
    with Path(path).open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('Another all40 run is active.') from e
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def check_zip(path,protocol_sha=None):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('Corrupt ZIP: '+str(path))
        manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
        if protocol_sha and manifest['protocol_sha256']!=protocol_sha:raise ValueError('Wrong result protocol')
        for name,h in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Result checksum differs: '+name)
        return json.loads(z.read('summary.json'))


class History:
    """As-of numerical feature bank. No fitting and no future labels are read."""
    def __init__(self):
        from app.csv_io import read_csv_checked
        log('Preparing sales/stock numerical histories')
        s=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv',usecols=['date',*KEY,'quantity'],
            parse_dates=['date'],dtype={'warehouse_name':'category','article':'category','quantity':'float32'})
        s=s.loc[s.date<=pd.Timestamp('2026-07-26')].copy()
        self.first=s.groupby(KEY,observed=True).date.min();self.first.index=self.first.index.set_levels(
            [self.first.index.levels[i].astype(str) for i in range(2)])
        self.start=s.date.min();s['week']=s.date-pd.to_timedelta(s.date.dt.weekday,unit='D')
        self.weekly=s.groupby([*KEY,'week'],observed=True).quantity.sum().unstack('week')
        self.daily=s.groupby([*KEY,'date'],observed=True).quantity.sum().unstack('date')
        del s;gc.collect()
        stock_path=ROOT/'data/stock_dataset_hakaton_train.csv'
        headers=read_csv_checked(stock_path,nrows=0).columns
        stock_article='char_article' if 'char_article' in headers else 'article'
        st=read_csv_checked(stock_path,usecols=['date','warehouse_name',stock_article,'stock'],
            parse_dates=['date'],dtype={'warehouse_name':'category',stock_article:'category','stock':'float32'})
        st=st.loc[st.date<=pd.Timestamp('2026-07-26')].rename(columns={stock_article:'article'})
        self.stock=st.groupby([*KEY,'date'],observed=True).stock.sum().unstack('date')
        for table in [self.weekly,self.daily,self.stock]:
            table.index=table.index.set_levels([table.index.levels[i].astype(str) for i in range(2)])
        del st;gc.collect()

    def add(self,frame,cutoff):
        cutoff=pd.Timestamp(cutoff);base=frame.drop_duplicates(KEY).copy()
        for k in KEY:base[k]=base[k].astype(str)
        pairs=pd.MultiIndex.from_frame(base[KEY]);first=self.first.reindex(pairs).to_numpy(dtype='datetime64[ns]')
        end=cutoff-pd.Timedelta(days=(cutoff.weekday()+1)%7)
        weeks=pd.date_range(end-pd.Timedelta(weeks=52)+pd.Timedelta(days=1),end-pd.Timedelta(days=6),freq='7D')
        y=self.weekly.reindex(index=pairs,columns=weeks).fillna(0).to_numpy(dtype='float32')
        known=(weeks.to_numpy()[None,:]>=first[:,None])&(weeks.to_numpy()[None,:]>=self.start.to_datetime64())
        y[~known]=np.nan
        extra=pd.DataFrame(index=pairs)
        for lag in range(14,53):extra[f'long_lag_{lag}']=y[:,-lag]
        for n in [26,52]:
            part=y[:,-n:];support=np.isfinite(part).sum(axis=1)
            extra[f'long_support_{n}']=support
            extra[f'long_mean_{n}']=np.divide(np.nansum(part,axis=1),support,out=np.full(len(y),np.nan),where=support>0)
        recent=np.nan_to_num(y[:,-2:],nan=0).sum(axis=1)/np.maximum(np.isfinite(y[:,-2:]).sum(axis=1),1)
        previous=np.nan_to_num(y[:,-10:-2],nan=0).sum(axis=1)/np.maximum(np.isfinite(y[:,-10:-2]).sum(axis=1),1)
        extra['change_ratio_2_8']=(recent+.1)/(previous+.1)
        extra['change_absolute_2_8']=recent-previous
        extra['life_age_days']=(cutoff.to_datetime64()-first)/np.timedelta64(1,'D')
        extra['life_positive_weeks_13']=(y[:,-13:]>0).sum(axis=1)
        extra['life_positive_weeks_52']=(y>0).sum(axis=1)
        extra['life_recent_share']=(np.nansum(y[:,-4:],axis=1)+.1)/(np.nansum(y,axis=1)+.1)
        dates=pd.date_range(cutoff-pd.Timedelta(days=89),cutoff)
        stock=self.stock.reindex(index=pairs,columns=dates).to_numpy(dtype='float32')
        quantity=self.daily.reindex(index=pairs,columns=dates).fillna(0).to_numpy(dtype='float32')
        for n in [28,90]:
            st=stock[:,-n:];q=quantity[:,-n:];obs=np.isfinite(st);available=obs&(st>0)
            extra[f'avail_observed_days_{n}']=obs.sum(axis=1)
            extra[f'avail_positive_days_{n}']=available.sum(axis=1)
            extra[f'avail_fraction_{n}']=np.divide(available.sum(axis=1),obs.sum(axis=1),out=np.full(len(y),np.nan),where=obs.sum(axis=1)>0)
            extra[f'avail_sales_per_positive_day_{n}']=np.divide((q*available).sum(axis=1),available.sum(axis=1),
                out=np.full(len(y),np.nan),where=available.sum(axis=1)>0)
        positive=np.isfinite(stock)&(stock>0);zero=np.isfinite(stock)&(stock<=0)
        extra['availability_positive_streak']=np.cumprod(positive[:,::-1],axis=1).sum(axis=1)
        extra['availability_nonpositive_streak']=np.cumprod(zero[:,::-1],axis=1).sum(axis=1)
        transitions=positive[:,1:]&zero[:,:-1]
        extra['availability_restock_transitions_28']=transitions[:,-28:].sum(axis=1)
        extra['availability_known_last_day']=np.isfinite(stock[:,-1]).astype(float)
        # Deterministic, label-free neighbor profiles; no KMeans fit is hidden here.
        profile=base[KEY+['category_lvl_2','mean_week_4','mean_week_13','zero_week_share_13']].copy()
        article=profile.groupby('article',observed=True).agg(mean4=('mean_week_4','mean'),mean13=('mean_week_13','mean'),
            zeros=('zero_week_share_13','mean'),category=('category_lvl_2','first'))
        a=article[['mean4','mean13','zeros']].to_numpy(float);a[:,:2]=np.log1p(np.maximum(a[:,:2],0))
        a=np.nan_to_num(a);a=(a-np.median(a,axis=0))/np.maximum(np.std(a,axis=0),.1)
        distance=((a[:,None,:]-a[None,:,:])**2).sum(axis=2);np.fill_diagonal(distance,np.inf)
        categories=article.category.astype(str).to_numpy();neighbor=[]
        for i in range(len(article)):
            eligible=np.flatnonzero((categories==categories[i])&(np.arange(len(article))!=i))
            if len(eligible)<3:eligible=np.flatnonzero(np.arange(len(article))!=i)
            chosen=eligible[np.argsort(distance[i,eligible],kind='stable')[:5]]
            neighbor.append(float(article.mean4.iloc[chosen].mean()) if len(chosen) else np.nan)
        lookup=pd.Series(neighbor,index=article.index)
        extra['neighbors_article_mean4']=pairs.get_level_values('article').map(lookup).to_numpy(float)
        own=base.set_index(KEY).mean_week_4.reindex(pairs)
        total=profile.groupby('article',observed=True).mean_week_4.sum();count=profile.groupby('article',observed=True).size()
        extra['neighbors_other_warehouses_mean4']=(pairs.get_level_values('article').map(total).to_numpy(float)-own.to_numpy(float))/np.maximum(pairs.get_level_values('article').map(count).to_numpy(float)-1,1)
        extra['neighbors_other_warehouses_support']=pairs.get_level_values('article').map(count).to_numpy(float)-1
        wh_profile=profile.pivot_table(index='warehouse_name',columns='category_lvl_2',values='mean_week_4',
            aggfunc='sum',observed=True).fillna(0)
        wh_profile=wh_profile.div(wh_profile.sum(axis=1).replace(0,1),axis=0)
        distance_wh=((wh_profile.to_numpy()[:,None,:]-wh_profile.to_numpy()[None,:,:])**2).sum(axis=2)
        np.fill_diagonal(distance_wh,np.inf)
        demand=profile.pivot_table(index='article',columns='warehouse_name',values='mean_week_4',aggfunc='mean',observed=True)
        estimates={};supports={}
        for i,warehouse in enumerate(wh_profile.index):
            nearest=np.argsort(distance_wh[i],kind='stable')[:min(3,max(len(wh_profile)-1,0))]
            columns=wh_profile.index[nearest];a=demand.reindex(columns=columns).to_numpy(float)
            weights=1/(distance_wh[i,nearest]+.1);valid=np.isfinite(a)
            numerator=(np.nan_to_num(a)*weights).sum(axis=1);denominator=(valid*weights).sum(axis=1)
            estimates[str(warehouse)]=pd.Series(np.divide(numerator,denominator,out=np.full(len(a),np.nan),where=denominator>0),index=demand.index)
            supports[str(warehouse)]=pd.Series(valid.sum(axis=1),index=demand.index)
        extra['neighbors_similar_warehouse_mean4']=[estimates.get(str(wh),pd.Series(dtype=float)).get(str(article),np.nan) for wh,article in pairs]
        extra['neighbors_similar_warehouse_support']=[supports.get(str(wh),pd.Series(dtype=float)).get(str(article),0) for wh,article in pairs]
        extra['neighbors_relative_demand']=(own.to_numpy(float)+.1)/(extra.neighbors_similar_warehouse_mean4.to_numpy(float)+.1)
        augmented=frame.copy()
        for k in KEY:augmented[k]=augmented[k].astype(str)
        augmented=augmented.merge(extra.reset_index(),on=KEY,how='left',validate='many_to_one',sort=False)
        for lag in [26,52]:
            values=np.full(len(augmented),np.nan)
            for week in augmented.week.unique():
                old=pd.Timestamp(week)-pd.Timedelta(weeks=lag)
                if old+pd.Timedelta(days=6)>cutoff:continue
                selected=augmented.week.eq(week);index=pd.MultiIndex.from_frame(augmented.loc[selected,KEY])
                raw=self.weekly.reindex(index=index,columns=[old]).fillna(0).iloc[:,0].to_numpy(float)
                f=self.first.reindex(index).to_numpy(dtype='datetime64[ns]')
                raw[(old.to_datetime64()<f)|(old<self.start)]=np.nan;values[selected]=raw
            augmented[f'year_analog_{lag}']=values
            augmented[f'year_analog_{lag}_known']=np.isfinite(values).astype(float)
        # Category/calendar interactions use historical observations only.
        category_code,category_labels=pd.factorize(base.category_lvl_2.astype(str),sort=True)
        category_by_pair=pd.Series(category_code,index=pairs)
        row_categories=pd.MultiIndex.from_frame(augmented[KEY]).map(category_by_pair).to_numpy(int)
        month=pd.to_datetime(augmented.week).dt.month.to_numpy(int)
        augmented['season_month_sin']=np.sin(2*np.pi*month/12)
        augmented['season_month_cos']=np.cos(2*np.pi*month/12)
        seasonal_mean=np.full(len(augmented),np.nan);seasonal_support=np.zeros(len(augmented))
        for m in np.unique(month):
            historical=y[:,weeks.month==m]
            sums=np.bincount(category_code,weights=np.nansum(historical,axis=1),minlength=len(category_labels))
            support=np.bincount(category_code,weights=np.isfinite(historical).sum(axis=1),minlength=len(category_labels))
            means=np.divide(sums,support,out=np.full(len(sums),np.nan),where=support>0)
            rows=month==m;seasonal_mean[rows]=means[row_categories[rows]];seasonal_support[rows]=support[row_categories[rows]]
        augmented['season_category_month_mean']=seasonal_mean
        augmented['season_category_month_support']=seasonal_support
        for c in list(extra.columns)+[f'year_analog_{n}{s}' for n in [26,52] for s in ['', '_known']]:
            augmented[c]=augmented[c].replace([np.inf,-np.inf],np.nan).astype('float32')
        return augmented


class Workbench:
    def __init__(self,check=False):
        self.bundle=ROOT/'artifacts/colab-sequential-20261029-final/colab_round08'
        self.event=ROOT/'artifacts/pycharm-round09-12-20261030/forecast_round09'
        if not self.bundle.is_dir() or not self.event.is_dir():raise RuntimeError('Required round08/round09 input folders are missing.')
        sys.path.insert(0,str(ROOT/'scripts'))
        self.common=module_at('all40_common',self.bundle/'run.py')
        self.event_module=module_at('all40_events',ROOT/'scripts/pycharm_event_round09.py')
        self.common.verify(self.bundle);self.event_protocol=self.event_module.verify(self.event)
        if sha(ROOT/'model/model.pkl')!=BASE_MODEL_SHA:raise RuntimeError('Production baseline changed; do not silently resume against another model.')
        self.recipe=read(self.bundle/'protocol.json')['recipe']
        paths=[ROOT/'data/sales_dataset_hakaton_train.csv',ROOT/'data/stock_dataset_hakaton_train.csv',
               self.bundle/'MANIFEST.json',self.event/'INPUT_MANIFEST.json',Path(__file__),
               ROOT/'scripts/pycharm_event_round09.py',ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json']
        paths.extend((ROOT/'app').rglob('*.py'))
        paths.extend(ROOT/'scripts'/name for name in ['research_events.py','research_weekly.py','finalize_smape_wape_cv.py'])
        paths.extend(ROOT/'artifacts/cold-pairs-20261029'/f'held_pairs_{o}.csv' for o in ORIGINS)
        paths.extend(p for p in (self.bundle/'screening').glob('*') if p.is_file())
        old8=ROOT/'artifacts/pycharm-round08-20261029/forecast_round08/round08_results.zip'
        choices9=[ROOT/'artifacts/pycharm-round09-12-20261030/forecast_round09_pycharm/forecast_round09/round09_results.zip',self.event/'round09_results.zip']
        old9=next((p for p in choices9 if p.exists()),None)
        if not old8.exists() or old9 is None:raise RuntimeError('Both returned round08/round09 ZIP files are required.')
        paths.extend([old8,old9])
        from scripts.research_events import inventory
        cache_fingerprints={}
        for origin in [*ORIGINS,'2026-05-24']:
            _,found=inventory(origin);cache_fingerprints.update(found)
        final_manifest=read(ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json')
        for name,digest in final_manifest.items():
            path=ROOT/'artifacts/improvements-20261028-r1/final-cache'/name
            if sha(path)!=digest:raise ValueError('Final history cache changed: '+name)
            cache_fingerprints[str(path.relative_to(ROOT))]=digest
        fingerprints={str(p.relative_to(ROOT)):sha(p) for p in paths}
        fingerprints.update(cache_fingerprints)
        self.protocol=dict(version=1,config=CONFIG,files=fingerprints,baseline_sha256=BASE_MODEL_SHA,origins=ORIGINS,
            scope=list(range(1,41)),independent_confirmation=False,production_changed=False,
            objective='fixed weighted ratios 50/20/15/10/5; also report naive-normalized and fixed-pool rank',
            training_location='user local Python; never invoked during assistant preparation')
        # This narrowly scoped repair can reuse the original run: inputs, settings
        # and every other source hash must still match. Record the repaired source
        # separately instead of invalidating already completed training.
        repaired_resume=False
        legacy_path=ROOT/'artifacts/all40-0a385edab6b1/protocol.json'
        source_key=str(Path(__file__).relative_to(ROOT))
        if legacy_path.exists():
            legacy=read(legacy_path)
            if legacy['files'].get(source_key)==DENSE_RESUME_SOURCE_SHA:
                compatible=json.loads(json.dumps(self.protocol))
                compatible['files'][source_key]=DENSE_RESUME_SOURCE_SHA
                if compatible==legacy:self.protocol=legacy;repaired_resume=True
        identity=hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()[:12]
        self.out=ROOT/'artifacts'/('all40-'+identity);self.out.mkdir(parents=True,exist_ok=True)
        if (self.out/'protocol.json').exists():
            if read(self.out/'protocol.json')!=self.protocol:raise RuntimeError('Frozen protocol changed')
        else:write(self.out/'protocol.json',self.protocol)
        if repaired_resume:
            patch=dict(patch='dense_available_horizons_and_lazy_history_v2',source_sha256=sha(Path(__file__)),
                original_source_sha256=DENSE_RESUME_SOURCE_SHA,
                reason='Request complete available target weeks and initialize history on uncached dense resume; retain valid completed checkpoints.')
            patch_path=self.out/'COMPATIBLE_CODE_FIX.json'
            if patch_path.exists() and read(patch_path)!=patch:
                previous=read(patch_path)
                if not (previous.get('patch')=='dense_available_horizons_v1'
                        and previous.get('source_sha256')=='b4c5fd8e208161a50c9a14000e8835e927b9909eef3a544f0ea9f4c2077bc121'
                        and previous.get('original_source_sha256')==DENSE_RESUME_SOURCE_SHA):
                    raise RuntimeError('Resume repair source changed; review compatibility before continuing.')
                write(self.out/'COMPATIBLE_CODE_FIX_v1.json',previous)
            write(patch_path,patch)
        self.ph=sha(self.out/'protocol.json');self.history=None;self.panel=None;self.frames={};self.records=[]
        from app.event_features import FEATURE_COLUMNS
        from app.enhancements import feature_columns,CALENDAR_COLUMNS,MONEY_COLUMNS
        self.event_columns=list(FEATURE_COLUMNS);self.base_columns=list(dict.fromkeys([*FEATURE_COLUMNS,*feature_columns()]))
        self.groups={'base':self.base_columns,'event':self.event_columns,'legacy49':feature_columns('dynamics'),
                     'legacy68':feature_columns(),'money':list(MONEY_COLUMNS),'calendar':list(CALENDAR_COLUMNS)}
        self.protocol_native=dict(objective='regression_l1',metric='l1',learning_rate=.05,num_leaves=31,max_depth=7,
            min_data_in_leaf=100,bagging_fraction=.9,bagging_freq=1,feature_fraction=.8,lambda_l1=5.,lambda_l2=20.,
            seed=CONFIG['seed'],num_threads=CONFIG['threads'],deterministic=True,force_col_wise=True,verbosity=-1)
        self.progress={str(i):dict(status='pending') for i in range(1,41)}
        if (self.out/'PROGRESS_40.json').exists():self.progress=read(self.out/'PROGRESS_40.json')['tasks']
        self.mark(40,'ongoing','fixed baseline; both windows retained, no independent-holdout claim')
        if check:log('CHECK ONLY: input hashes and packages verified. No fitting.',self.out)

    def mark(self,task,status,note=''):
        self.progress[str(task)]=dict(status=status,note=note)
        write(self.out/'PROGRESS_40.json',dict(tasks=self.progress,production_changed=False))

    def feature_frame(self,origin,name,final=False):
        identifier=('final' if final else origin)+'_'+name
        path=self.out/'features'/(identifier+'.parquet')
        if path.exists():return pd.read_parquet(path)
        if self.history is None:self.history=History()
        from app.event_features import make_event_features
        from app.enhancements import feature_columns
        if final:
            source=ROOT/'artifacts/improvements-20261028-r1/final-cache'/name
            frame=pd.read_parquet(source)
            from scripts.finalize_smape_wape_cv import WeeklyLagHistory
            # The History bank already has the same weekly table; add the known 13 lags directly.
            cutoff=pd.Timestamp(name[:10]);end=cutoff-pd.Timedelta(days=(cutoff.weekday()+1)%7)
            weeks=pd.date_range(end-pd.Timedelta(days=90),end-pd.Timedelta(days=6),freq='7D')
            index=pd.MultiIndex.from_frame(frame[KEY].drop_duplicates())
            y=self.history.weekly.reindex(index=index,columns=weeks).fillna(0).to_numpy(dtype='float32')
            first=self.history.first.reindex(index).to_numpy(dtype='datetime64[ns]')
            valid=(weeks.to_numpy()[None,:]>=first[:,None])&(weeks.to_numpy()[None,:]>=self.history.start.to_datetime64())
            y[~valid]=np.nan;extra=pd.DataFrame(index=index)
            for lag in range(1,14):extra[f'weekly_lag_{lag}']=y[:,-lag]
            extra['weekly_support_13']=np.isfinite(y).sum(axis=1)
            frame=frame.merge(extra.reset_index(),on=KEY,how='left',validate='many_to_one');frame['feature_origin']=cutoff
        else:
            source=ROOT/'artifacts/ideas-20261028-r2'/origin/'cache'/name
            frame=pd.read_parquet(source)
        x=frame[list(dict.fromkeys([*self.base_columns,*feature_columns(),'week','feature_origin',
            'target' if 'target' in frame else 'actual']))].copy() if all(c in frame for c in self.base_columns) else frame.copy()
        event=make_event_features(frame)
        for c in event:
            if c not in x:x[c]=event[c].to_numpy()
        # Drop obsolete, unselected research columns; build new feature groups from as-of inputs.
        meta=['week','feature_origin','target' if 'target' in frame else 'actual']
        x=x[list(dict.fromkeys(self.base_columns+meta))].copy()
        x=self.history.add(x,name[:10])
        additions=self.event_module.extra_features(x)
        for c in additions:
            if c not in x:x[c]=additions[c].to_numpy()
        for c in x.select_dtypes(include=[np.number]):x[c]=x[c].astype('float32')
        write_frame(path,x)
        return x

    def discover_groups(self):
        f=self.feature_frame(ORIGINS[0],self.schedule(ORIGINS[0])['test'])
        self.groups.update(
            availability=[c for c in f if c.startswith('avail_')],
            availability_dynamics=[c for c in f if c.startswith('availability_')],
            long26=[c for c in f if (c.startswith('long_lag_') and int(c.rsplit('_',1)[1])<=26) or c in ['long_mean_26','long_support_26']],
            long52=[c for c in f if c.startswith('long_')],
            regime=[c for c in f if c.startswith(('change_','regime_'))],
            lifecycle=[c for c in f if c.startswith('life_')],
            season=[c for c in f if c.startswith(('year_analog_','season_'))],
            neighbors=[c for c in f if c.startswith('neighbors_')],
            stock_interactions=[c for c in f if c.startswith('stock_event_')])
        self.groups['all']=list(dict.fromkeys(self.base_columns+sum([v for k,v in self.groups.items()
            if k not in ['base','event','legacy49','legacy68']],[])))
        write(self.out/'feature_groups.json',self.groups)

    def schedule(self,origin):
        from scripts.research_events import split_names
        manifest=read(ROOT/'artifacts/ideas-20261028-r2'/origin/'cache_manifest.json')
        return split_names(origin,manifest)

    def dataset(self,origin,split,spec,final=False):
        if spec.get('dense'):return dense_dataset(self,origin,split,spec,final=final)
        if final:
            manifest=read(ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json')
            names=sorted(manifest);boundary=pd.Timestamp('2026-05-24' if split=='inner' else '2026-07-26')
            if split=='valid':
                frame=self.feature_frame('2026-05-24','2026-05-24-False.parquet').rename(columns={'actual':'target'})
                frame=frame.loc[(pd.to_datetime(frame.week)+pd.Timedelta(days=6))<=boundary].copy()
                if spec.get('positive_only'):frame=frame.loc[frame.target>0]
                if spec.get('regular_only'):frame=frame.loc[frame.mean_28*7>=2]
                if spec.get('head_filter'):frame=frame.loc[self.event_module.mask(frame,spec['head_filter'])]
                return frame
        else:
            schedule=self.schedule(origin);names=schedule[split] if isinstance(schedule[split],list) else [schedule[split]]
            boundary=pd.Timestamp(schedule['inner_origin'] if split=='inner' else origin)
        frames=[]
        for name in names:
            frame=self.feature_frame(origin,name,final=final)
            if final:frame=frame.loc[(pd.to_datetime(frame.week)+pd.Timedelta(days=6))<=boundary].copy()
            elif (pd.to_datetime(frame.week)+pd.Timedelta(days=6)).max()>boundary:raise ValueError('Target leakage in '+name)
            if frame.empty:continue
            if spec.get('window_days') and (boundary-pd.Timestamp(name[:10])).days>spec['window_days']:continue
            if spec.get('positive_only'):frame=frame.loc[frame.target>0]
            if spec.get('regular_only'):frame=frame.loc[frame.mean_28*7>=2]
            if spec.get('head_filter'):frame=frame.loc[self.event_module.mask(frame,spec['head_filter'])]
            if spec.get('complete_only'):
                count=frame.groupby(['feature_origin',*KEY],observed=True).h.transform('count');frame=frame.loc[count==9]
            frames.append(frame)
        if not frames:raise ValueError('No eligible frames for '+str(spec))
        result=pd.concat(frames,ignore_index=True);del frames;gc.collect()
        return result

    def columns(self,spec):
        groups=spec.get('groups',['all']);return list(dict.fromkeys(sum([self.groups[g] for g in groups],[])))

    def evaluate(self,name,predictions,tasks,details=None):
        result=dict(id=name,tasks=tasks,metrics={},baseline={},per_origin_ratio={},details=details or {},independent_confirmation=False)
        folder=self.out/'predictions';folder.mkdir(exist_ok=True)
        for origin,p in predictions.items():
            f,values,arrays=self.common.load(self.bundle,origin);m=self.common.metrics(f.actual,p,self.common.codes(f))
            b=self.common.metrics(f.actual,arrays['baseline'],self.common.codes(f))
            result['metrics'][origin]=m;result['baseline'][origin]=b
            result['per_origin_ratio'][origin]=float(sum(w*self.common.ratio(m[k],b[k]) for w,k in zip(WEIGHTS,METRICS)))
            np.save(folder/(name+'_'+origin+'.npy'),p)
        result['weighted_ratio']=float(np.mean(list(result['per_origin_ratio'].values())))
        result['worst_window_ratio']=max(result['per_origin_ratio'].values())
        write(self.out/'evaluations'/(name+'.json'),result);self.records.append(result)
        log('RESULT',name,round(result['weighted_ratio'],8))
        return result

    def resume_evaluations(self):
        self.records=[read(p) for p in sorted((self.out/'evaluations').glob('*.json'))] if (self.out/'evaluations').exists() else []

    def saved_prediction(self,name,origin):return np.load(self.out/'predictions'/(name+'_'+origin+'.npy'),allow_pickle=False)

    def prior_rounds(self):
        screening=read(self.bundle/'screening/results.json')
        short={o:np.load(self.bundle/'screening'/f'{o}.shortlist.npz',allow_pickle=False) for o in ORIGINS}
        for r in screening['best_by_task'].values():
            predictions={o:short[o][r['id']] for o in ORIGINS}
            self.evaluate('cached_'+r['id'],predictions,[r['task']],dict(policy=r['policy'],source='frozen round01 screening'))
        for i in list(range(1,8))+[13,14,18,28]:self.mark(i,'completed_cached_screen','saved forecasts and declared finite grid')
        source8=ROOT/'artifacts/pycharm-round08-20261029/forecast_round08/round08_results.zip'
        if not source8.exists():raise RuntimeError('Completed round08 archive is required; do not repeat it silently.')
        eight=check_zip(source8,sha(self.bundle/'protocol.json'))
        self.mark(8,'completed_rejected','all four calibrators retained in source archive; baseline probabilities kept')
        choices=[ROOT/'artifacts/pycharm-round09-12-20261030/forecast_round09_pycharm/forecast_round09/round09_results.zip',
                 self.event/'round09_results.zip']
        source9=next((p for p in choices if p.exists()),None)
        if source9 is None:raise RuntimeError('Completed round09 archive is required.')
        nine=check_zip(source9,sha(self.event/'protocol.json'))
        with zipfile.ZipFile(source9) as z:
            for r in nine['records']:
                for route in r['routes']:
                    suffix='.prediction.npy' if route=='dynamics' else '.'+route+'.prediction.npy'
                    predictions={o:np.load(io.BytesIO(z.read(r['id']+'/'+o+suffix)),allow_pickle=False) for o in ORIGINS}
                    self.evaluate('returned_'+r['id']+'_'+route,predictions,[r['task']],
                        dict(source_zip=str(source9.relative_to(ROOT)),variant=r['id'],route=route))
        for i in [9,10,11,12]:self.mark(i,'completed_user_training','validated round09 models and all three output routes reused')
        for z in short.values():z.close()


def native_specs():
    specs=[dict(id='quantity_control',tasks=[17],kind='quantity',groups=['base'],objective='regression_l1'),
           dict(id='quantity_all',tasks=[17,32],kind='quantity',groups=['all'],objective='regression_l1',half_life=180),
           dict(id='count8',tasks=[14,15],kind='count',groups=['all'],edges=[1,2,3,4,7,15]),
           dict(id='count_fine',tasks=[15],kind='count',groups=['all'],edges=[.5,1,1.5,2,3,4,6,8,12,20,40])]
    mapping={'availability':[20],'availability_dynamics':[21],'long26':[22],'long52':[22],
             'regime':[23],'lifecycle':[24],'money':[25],'calendar':[26],'season':[26],'neighbors':[27]}
    for group,tasks in mapping.items():
        # Event-only base excludes money/calendar; adding them is an actual ablation.
        specs.append(dict(id='feature_'+group,tasks=tasks,kind='quantity',groups=['event',group],objective='regression_l1'))
    specs.extend([
        dict(id='money_event',tasks=[25],kind='binary',groups=['event','money'],objective='binary'),
        dict(id='money_count',tasks=[25],kind='count',groups=['event','money'],edges=[1,2,3,4,7,15]),
        dict(id='money_positive',tasks=[25],kind='quantity',groups=['event','money'],objective='poisson',positive_only=True),
        dict(id='money_recent',tasks=[25],kind='quantity',groups=['event','money'],objective='regression_l1',regular_only=True,half_life=180),
        dict(id='season_event',tasks=[26],kind='binary',groups=['event','calendar','season'],objective='binary')])
    for alpha in [.25,.5,.75]:specs.append(dict(id=f'quantile_{alpha}',tasks=[16],kind='quantity',groups=['all'],objective='quantile',alpha=alpha))
    for loss in ['huber','poisson','tweedie']:
        specs.append(dict(id='loss_'+loss,tasks=[17],kind='quantity',groups=['all'],objective=loss))
    for days in [90,365]:specs.append(dict(id=f'recency_{days}',tasks=[35],kind='quantity',groups=['all'],objective='regression_l1',half_life=days))
    for days in [180,365]:specs.append(dict(id=f'window_{days}',tasks=[35],kind='quantity',groups=['all'],objective='regression_l1',window_days=days))
    specs.extend([
        dict(id='equal_horizons',tasks=[37],kind='quantity',groups=['all'],objective='regression_l1',equal_horizons=True),
        dict(id='complete_horizons',tasks=[37],kind='quantity',groups=['all'],objective='regression_l1',complete_only=True),
        dict(id='core_event',tasks=[38],kind='binary',groups=['legacy49'],objective='binary'),
        dict(id='core_positive',tasks=[38],kind='quantity',groups=['legacy49'],objective='poisson',positive_only=True),
        dict(id='core_log',tasks=[38],kind='quantity',groups=['legacy49'],objective='regression',log_target=True),
        dict(id='core_mae',tasks=[38],kind='quantity',groups=['legacy68'],objective='regression_l1'),
        dict(id='core_count',tasks=[38],kind='count',groups=['event'],edges=[1,2,3,4,7,15]),
        dict(id='core_recent',tasks=[38],kind='quantity',groups=['event'],objective='regression_l1',regular_only=True,half_life=180),
        dict(id='total9',tasks=[19],kind='total',groups=['all'],objective='regression',log_target=True),
        dict(id='profile9',tasks=[19],kind='profile',groups=['all'],objective='regression_l1'),
        dict(id='parent_article',tasks=[33],kind='parent_article',groups=['all'],objective='regression_l1'),
        dict(id='parent_warehouse',tasks=[33],kind='parent_warehouse',groups=['all'],objective='regression_l1')])
    return specs


def transform_target_frame(frame,spec,predict=False):
    kind=spec['kind'];frame=frame.copy()
    if kind=='total':
        group=['feature_origin',*KEY];count=frame.groupby(group,observed=True).h.transform('count')
        frame=frame.loc[count==9].sort_values('h',kind='stable')
        total=frame.groupby(group,observed=True).target.sum() if not predict else None
        frame=frame.groupby(group,observed=True,sort=False).head(1).copy()
        if not predict:frame['target']=pd.MultiIndex.from_frame(frame[group]).map(total).to_numpy(float)
    elif kind=='profile' and not predict:
        group=['feature_origin',*KEY];size=frame.groupby(group,observed=True).h.transform('count')
        frame=frame.loc[size==9].copy();total=frame.groupby(group,observed=True).target.transform('sum')
        keep=total>0;frame=frame.loc[keep].copy();total=total.loc[keep]
        frame['target']=frame.target/total
    elif kind.startswith('parent_'):
        parent='article' if kind=='parent_article' else 'warehouse_name'
        group=['feature_origin','week','h',parent]
        amounts=['mean_week_4','mean_week_13','mean_28','stock_last','stock_mean_28']
        average=['zero_week_share_13','stock_zero_share_90','month','week_of_year','year']
        agg={c:'sum' for c in amounts};agg.update({c:'mean' for c in average})
        if not predict:agg['target']='sum'
        count=frame.groupby(group,observed=True).size().rename('parent_group_size')
        frame=frame.groupby(group,observed=True,as_index=False).agg(agg)
        frame=frame.merge(count.reset_index(),on=group,how='left',validate='one_to_one')
    return frame.reset_index(drop=True)


def model_columns(w,spec,frame):
    if spec['kind'].startswith('parent_'):
        parent='article' if spec['kind']=='parent_article' else 'warehouse_name'
        return [parent]+[c for c in frame.select_dtypes(include=[np.number]) if c not in ['target','actual']]
    if spec.get('selected_features'):return spec['selected_features']
    return w.columns(spec)


def encode_frame(frame,columns,categories=None):
    x=frame[columns].copy();categorical=[c for c in CAT if c in columns]
    if categories is None:categories={c:sorted(x[c].astype(str).unique()) for c in categorical}
    for c in categorical:x[c]=pd.Categorical(x[c].astype(str),categories=categories[c])
    return x,categories


def count_labels(y,edges):
    return np.where(y==0,0,np.searchsorted(np.asarray(edges,float),y,side='left')+1).astype('int32')


def decode_count(probability,schema,multiplier=3.):
    representatives=np.asarray(schema['representatives'],float)
    actions=np.unique(np.r_[0.,.5,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.,representatives])
    absolute=np.abs(representatives[:,None]-actions[None,:]);den=representatives[:,None]+actions[None,:]
    smape=np.divide(200*absolute,den,out=np.zeros_like(absolute),where=den>0)
    risk=.7/60*smape+multiplier*.3/50*100*absolute/max(schema['training_mean'],1e-9)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):return actions[np.argmin(probability@risk,axis=1)]


def training_weight(frame,spec):
    weight=np.ones(len(frame))
    if spec.get('half_life'):
        boundary=(pd.to_datetime(frame.week)+pd.Timedelta(days=6)).max()
        age=(boundary-pd.to_datetime(frame.feature_origin)).dt.days.to_numpy(float)
        if (age<0).any():raise ValueError('Negative training age')
        weight*=2.**(-age/spec['half_life'])
    if spec.get('equal_horizons'):
        counts=frame.h.value_counts();weight/=frame.h.map(counts).to_numpy(float)
    if spec.get('dense'):
        count=frame.groupby(ROWKEY,observed=True).h.transform('count').to_numpy(float);weight/=count
    return weight/weight.mean()


def checkpoint(folder,name,ph):
    path=folder/name
    if not path.exists():return None
    x=read(path)
    if x['protocol_sha256']!=ph:raise ValueError('Checkpoint protocol changed')
    for file,digest in x['files'].items():
        if sha(folder/file)!=digest:raise ValueError('Checkpoint changed: '+str(folder/file))
    return x


def native_prediction(folder,frame,inner=False,raw=False):
    import lightgbm as lgb
    prefix='inner' if inner else 'refit'
    schema=read(folder/(prefix+'_schema.json'));spec=schema['spec']
    transformed=transform_target_frame(frame,spec,predict=True)
    x,_=encode_frame(transformed,schema['columns'],schema['categories'])
    model=lgb.Booster(model_file=str(folder/(prefix+'_model.txt')))
    p=model.predict(x,num_threads=CONFIG['threads'])
    if spec['kind']=='count':p=p if raw else decode_count(p,schema)
    elif spec.get('log_target'):p=np.maximum(np.expm1(np.clip(p,-20,20)),0.)
    elif spec['kind']!='binary':p=np.maximum(p,0.)
    if spec['kind']=='total':
        lookup=pd.Series(p,index=pd.MultiIndex.from_frame(transformed[['feature_origin',*KEY]]))
        p=pd.MultiIndex.from_frame(frame[['feature_origin',*KEY]]).map(lookup).to_numpy(float)
    elif spec['kind'].startswith('parent_'):
        parent='article' if spec['kind']=='parent_article' else 'warehouse_name'
        keys=['feature_origin','week','h',parent]
        lookup=pd.Series(p,index=pd.MultiIndex.from_frame(transformed[keys]))
        p=pd.MultiIndex.from_frame(frame[keys]).map(lookup).to_numpy(float)
    return np.asarray(p,float)


def fit_native(w,spec,origin,final=False):
    import lightgbm as lgb
    import psutil
    folder=w.out/'models'/spec['id']/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    done=checkpoint(folder,'complete.json',w.ph)
    if done:return folder
    params={**w.protocol_native,'objective':spec.get('objective','multiclass' if spec['kind']=='count' else 'regression_l1')}
    if spec['kind']=='binary':params['metric']='binary_logloss'
    elif spec['kind']=='count':params.update(metric='multi_logloss',num_class=len(spec['edges'])+2)
    elif spec.get('log_target'):params['metric']='rmse'
    else:params['metric']='l1'
    if spec.get('alpha') is not None:params['alpha']=spec['alpha']
    if spec.get('objective')=='tweedie':params['tweedie_variance_power']=1.5
    def guard(environment):
        if psutil.Process().memory_info().rss>CONFIG['memory_limit_gib']*2**30:
            raise MemoryError('Memory budget exceeded; resume after reviewing workload, do not silently reduce data.')
    inner=checkpoint(folder,'inner.json',w.ph)
    if inner is None:
        train=transform_target_frame(w.dataset(origin,'inner',spec,final=final),spec)
        valid=transform_target_frame(w.dataset(origin,'valid',spec,final=final),spec)
        if not len(train) or not len(valid):raise ValueError('Empty training/validation data for '+spec['id'])
        columns=model_columns(w,spec,train);x,categories=encode_frame(train,columns);xv,_=encode_frame(valid,columns,categories)
        y=train.target.to_numpy(float);yv=valid.target.to_numpy(float)
        schema=dict(spec=spec,columns=columns,categories=categories)
        if spec['kind']=='binary':y=(y>0).astype('int8');yv=(yv>0).astype('int8')
        elif spec['kind']=='count':
            labels=count_labels(y,spec['edges']);reps=np.r_[0.,spec['edges'],spec['edges'][-1]*2].astype(float)
            for k in range(1,len(reps)):
                if (labels==k).any():reps[k]=y[labels==k].mean()
            schema.update(representatives=reps.tolist(),training_mean=float(y.mean()))
            y=labels;yv=count_labels(yv,spec['edges'])
        elif spec.get('log_target'):y=np.log1p(y);yv=np.log1p(yv)
        ds=lgb.Dataset(x,label=y,weight=training_weight(train,spec),categorical_feature=list(categories),free_raw_data=True)
        dv=lgb.Dataset(xv,label=yv,reference=ds,categorical_feature=list(categories),free_raw_data=True)
        log('FIT inner',spec['id'],origin,len(train),len(columns))
        maximum=spec.get('max_iterations',CONFIG['count_iterations'] if spec['kind']=='count' else CONFIG['max_iterations'])
        model=lgb.train(params,ds,num_boost_round=maximum,valid_sets=[dv],
            callbacks=[guard,lgb.early_stopping(spec.get('early_stopping',CONFIG['early_stopping']),verbose=False)])
        iterations=int(model.best_iteration or model.current_iteration())
        model.save_model(str(folder/'inner_model.txt'),num_iteration=iterations);write(folder/'inner_schema.json',schema)
        # Honest predictions for stacking/router training, never in-sample refit outputs.
        p=model.predict(xv,num_iteration=iterations,num_threads=CONFIG['threads'])
        if spec['kind']=='count':p=decode_count(p,schema)
        elif spec.get('log_target'):p=np.maximum(np.expm1(np.clip(p,-20,20)),0.)
        else:p=np.maximum(p,0.)
        oof=valid[[c for c in ['feature_origin',*ROWKEY,'h','target'] if c in valid]].copy();oof['prediction']=p
        oof.to_parquet(folder/'inner_predictions.parquet',index=False)
        importance=dict(zip(columns,model.feature_importance(importance_type='gain').tolist()))
        write(folder/'importance.json',importance)
        inner=dict(protocol_sha256=w.ph,iterations=iterations,files={name:sha(folder/name) for name in
            ['inner_model.txt','inner_schema.json','inner_predictions.parquet','importance.json']})
        write(folder/'inner.json',inner)
        del train,valid,x,xv,y,yv,ds,dv,model,oof;gc.collect()
    train=transform_target_frame(w.dataset(origin,'refit',spec,final=final),spec)
    columns=model_columns(w,spec,train);x,categories=encode_frame(train,columns);y=train.target.to_numpy(float)
    schema=dict(spec=spec,columns=columns,categories=categories)
    if spec['kind']=='binary':y=(y>0).astype('int8')
    elif spec['kind']=='count':
        labels=count_labels(y,spec['edges']);reps=np.r_[0.,spec['edges'],spec['edges'][-1]*2].astype(float)
        for k in range(1,len(reps)):
            if (labels==k).any():reps[k]=y[labels==k].mean()
        schema.update(representatives=reps.tolist(),training_mean=float(y.mean()));y=labels
    elif spec.get('log_target'):y=np.log1p(y)
    ds=lgb.Dataset(x,label=y,weight=training_weight(train,spec),categorical_feature=list(categories),free_raw_data=True)
    log('FIT refit',spec['id'],origin,len(train),inner['iterations'])
    model=lgb.train(params,ds,num_boost_round=inner['iterations'],callbacks=[guard])
    model.save_model(str(folder/'refit_model.txt'));write(folder/'refit_schema.json',schema)
    write(folder/'complete.json',dict(protocol_sha256=w.ph,rows=len(train),iterations=inner['iterations'],files={name:sha(folder/name)
        for name in ['inner.json',*inner['files'],'refit_model.txt','refit_schema.json']}))
    del train,x,y,ds,model;gc.collect();return folder


def evaluate_native(w,spec):
    predictions={};raw_predictions={}
    for origin in ORIGINS:
        model=fit_native(w,spec,origin);f=w.feature_frame(origin,w.schedule(origin)['test'])
        raw_predictions[origin]=native_prediction(model,f)
    if spec['kind'].startswith('parent_') or spec['kind'] in ['total','profile']:return raw_predictions
    for weight in [.25,.5,1.]:
        for keep_zeros in ([False,True] if spec['kind']=='quantity' else [False]):
            for origin in ORIGINS:
                f,values,arrays=w.common.load(w.bundle,origin);base=arrays['baseline'];p=raw_predictions[origin]
                if spec['kind']=='binary':
                    v=dict(values);v['dynamics']={**values['dynamics'],'probability':p}
                    from app.models.research_model import recipe_prediction
                    q=recipe_prediction(v,f,w.recipe);predictions[origin]=(1-weight)*base+weight*q
                elif spec['kind']=='count':
                    scope=(f.mean_28.to_numpy(float)*7>=.5)&(f.mean_28.to_numpy(float)*7<10)
                    q=base.copy();q[scope]=(1-weight)*base[scope]+weight*p[scope];predictions[origin]=q
                else:
                    scope=f.mean_28.to_numpy(float)*7>=2
                    q=base.copy();q[scope]=(1-weight)*base[scope]+weight*p[scope]
                    predictions[origin]=np.where(base>0,q,0.) if keep_zeros else q
            w.evaluate('native_'+spec['id']+'_w'+str(weight)+('_zeros' if keep_zeros else ''),predictions,spec['tasks'],
                dict(kind='native_overlay',spec=spec,weight=weight,keep_base_zeros=keep_zeros))
    return raw_predictions


def available_training_horizons(cutoff,history_end):
    from app.features import next_week_start
    first=next_week_start(pd.Timestamp(cutoff))
    return [h for h in range(1,10) if first+pd.Timedelta(days=7*h-1)<=pd.Timestamp(history_end)]


def dense_dataset(w,origin,split,spec,final=False):
    # Validation stays unchanged; only the training origins get denser.
    if split=='valid':return w.dataset(origin,split,{**spec,'dense':False},final=final)
    if final:
        names=sorted(read(ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json'))
        boundary=pd.Timestamp('2026-05-24' if split=='inner' else '2026-07-26')
    else:
        schedule=w.schedule(origin);names=schedule[split]
        boundary=pd.Timestamp(schedule['inner_origin'] if split=='inner' else origin)
    start=pd.Timestamp(min(names)[:10]);end=pd.Timestamp(max(names)[:10])
    cuts=pd.date_range(start,end,freq=f'{CONFIG["dense_step_days"]}D')
    cuts=cuts[cuts<boundary]
    paths=[]
    for cut in cuts:
        path=w.out/'features'/f'dense_{origin}_{cut.date()}.parquet';paths.append(path)
        if path.exists():continue
        log('PREP dense cutoff',origin,str(cut.date()))
        if w.panel is None or w.panel_origin!=origin:
            from app.csv_io import read_csv_checked
            from app.features import truncate_history
            from app.pipeline import prepare_panel
            sales=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv',parse_dates=['date'])
            stock=read_csv_checked(ROOT/'data/stock_dataset_hakaton_train.csv',parse_dates=['date'])
            sales,stock=truncate_history(sales,stock,pd.Timestamp(origin))
            w.panel=prepare_panel(sales,stock,'research',read(ROOT/'model/config.json')['model_params'])
            w.panel_origin=origin;del sales,stock;gc.collect()
        from app.pipeline import feature_frame
        horizons=available_training_horizons(cut,w.panel.date.max())
        if not horizons:raise ValueError('Dense cutoff has no complete target week: '+str(cut))
        frame=feature_frame(w.panel,cut,'research',training=True,horizons=horizons,
            model_params=read(ROOT/'model/config.json')['model_params'])
        frame['feature_origin']=cut
        if w.history is None:w.history=History()
        frame=w.history.add(frame,cut)
        extra=w.event_module.extra_features(frame)
        for c in extra:
            if c not in frame:frame[c]=extra[c].to_numpy()
        write_frame(path,frame);del frame;gc.collect()
    w.panel=None;gc.collect()
    frames=[]
    for path in paths:
        f=pd.read_parquet(path);f=f.loc[(pd.to_datetime(f.week)+pd.Timedelta(days=6))<=boundary]
        if len(f):frames.append(f)
    if not frames:raise ValueError('Dense schedule has no legal labels')
    return pd.concat(frames,ignore_index=True)


def torch_model(outputs,context_size):
    import torch
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence=torch.nn.Sequential(torch.nn.Conv1d(2,16,3,padding=1),torch.nn.ReLU(),
                torch.nn.Conv1d(16,32,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool1d(1))
            self.head=torch.nn.Sequential(torch.nn.Linear(32+context_size,48),torch.nn.ReLU(),torch.nn.Linear(48,outputs))
        def forward(self,sequence,context):return self.head(torch.cat([self.sequence(sequence).squeeze(-1),context],dim=1))
    return Network()


def tensor_data(frame,joint=False,schema=None):
    lag_columns=[f'long_lag_{i}' if i>=14 else f'weekly_lag_{i}' for i in range(52,0,-1)]
    context=['h','mean_28','std_week_13','zero_week_share_13','stock_last','stock_zero_share_90','change_ratio_2_8','life_age_days']
    if joint:
        code,unique=pd.factorize(pd.MultiIndex.from_frame(frame[['feature_origin',*KEY]]))
        first=np.unique(code,return_index=True)[1];selected=frame.iloc[first]
        y=np.zeros((len(first),9),dtype='float32');mask=np.zeros_like(y)
        if 'target' in frame:
            y[code,frame.h.to_numpy(int)-1]=np.log1p(frame.target.to_numpy(float));mask[code,frame.h.to_numpy(int)-1]=1
    else:
        selected=frame;code=None
        y=(frame.target.to_numpy(float)>0).astype('float32').reshape(-1,1) if 'target' in frame else np.zeros((len(frame),1),dtype='float32')
        mask=np.ones_like(y)
    lag=selected[lag_columns].to_numpy(dtype='float32');known=np.isfinite(lag)
    sequence=np.stack([np.log1p(np.maximum(np.nan_to_num(lag,nan=0),0)),known.astype('float32')],axis=1)
    ctx=selected[context].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(dtype='float32')
    if schema is None:schema=dict(mean=ctx.mean(axis=0).tolist(),std=np.maximum(ctx.std(axis=0),.1).tolist(),context=context,lag_columns=lag_columns)
    ctx=(ctx-np.asarray(schema['mean'],dtype='float32'))/np.asarray(schema['std'],dtype='float32')
    return sequence,ctx,y,mask,schema,code


def fit_torch(w,origin,joint=False,final=False):
    import torch
    import psutil
    name='cnn_joint9' if joint else 'cnn_event';folder=w.out/'models'/name/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    if checkpoint(folder,'complete.json',w.ph):return folder
    torch.set_num_threads(CONFIG['threads']);torch.manual_seed(CONFIG['seed']);torch.use_deterministic_algorithms(True)
    for phase in ['inner','refit']:
        if checkpoint(folder,phase+'.json',w.ph):continue
        spec=dict(kind='quantity',groups=['all'])
        frame=w.dataset(origin,'inner' if phase=='inner' else 'refit',spec,final=final)
        arrays=tensor_data(frame,joint);schema=arrays[4];del frame;gc.collect()
        sequence,ctx,y,mask=arrays[:4]
        model=torch_model(9 if joint else 1,ctx.shape[1]);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
        if phase=='inner':
            v=w.dataset(origin,'valid',spec,final=final);validation=tensor_data(v,joint,schema);del v;gc.collect()
            epochs=CONFIG['cnn_epochs']
        else:
            epochs=read(folder/'inner.json')['epochs'];validation=None
        dataset=torch.utils.data.TensorDataset(*[torch.from_numpy(a) for a in [sequence,ctx,y,mask]])
        generator=torch.Generator().manual_seed(CONFIG['seed'])
        loader=torch.utils.data.DataLoader(dataset,batch_size=CONFIG['cnn_batch_size'],shuffle=True,generator=generator,num_workers=0)
        progress=folder/(phase+'_training.json');begin=0;best=float('inf');bad=0;best_epoch=0
        if progress.exists():
            state=read(progress)
            if state['protocol_sha256']!=w.ph or sha(folder/state['file'])!=state['sha256']:raise ValueError('CNN resume state differs')
            saved=torch.load(folder/state['file'],map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['generator'])
            torch.set_rng_state(saved['rng']);begin=state['epoch'];best=state['best'];bad=state['bad'];best_epoch=state['best_epoch']
        for epoch in range(begin,epochs):
            if phase=='inner' and bad>=2:break
            log('CNN',name,origin,phase,epoch+1,'/',epochs)
            model.train()
            for xx,cc,yy,mm in loader:
                optimizer.zero_grad();out=model(xx,cc)
                loss=(torch.nn.functional.smooth_l1_loss(torch.nn.functional.softplus(out),yy,reduction='none') if joint else
                      torch.nn.functional.binary_cross_entropy_with_logits(out,yy,reduction='none'))
                loss=(loss*mm).sum()/mm.sum().clamp_min(1);loss.backward();optimizer.step()
                if psutil.Process().memory_info().rss>CONFIG['memory_limit_gib']*2**30:raise MemoryError('CNN memory budget exceeded')
            score=0.
            if phase=='inner':
                model.eval();total=den=0.
                with torch.no_grad():
                    for start in range(0,len(validation[0]),2048):
                        a=[torch.from_numpy(v[start:start+2048]) for v in validation[:4]]
                        out=model(a[0],a[1]);loss=(torch.nn.functional.smooth_l1_loss(torch.nn.functional.softplus(out),a[2],reduction='none') if joint else
                            torch.nn.functional.binary_cross_entropy_with_logits(out,a[2],reduction='none'))
                        total+=float((loss*a[3]).sum());den+=float(a[3].sum())
                score=total/max(den,1)
                if score<best-1e-7:best=score;bad=0;best_epoch=epoch+1
                else:bad+=1
            else:best=0.;best_epoch=epoch+1
            statefile=f'{phase}_epoch{epoch+1}.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),generator=generator.get_state(),rng=torch.get_rng_state()),folder/statefile)
            write(progress,dict(protocol_sha256=w.ph,file=statefile,sha256=sha(folder/statefile),epoch=epoch+1,
                best=best,bad=bad,best_epoch=best_epoch))
        state=read(progress);chosen=folder/f'{phase}_epoch{state["best_epoch"]}.pt'
        best_state=torch.load(chosen,map_location='cpu',weights_only=True)
        torch.save(best_state['model'],folder/(phase+'_model.pt'));write(folder/(phase+'_schema.json'),dict(**schema,joint=joint))
        write(folder/(phase+'.json'),dict(protocol_sha256=w.ph,epochs=state['best_epoch'],files={name:sha(folder/name)
            for name in [phase+'_model.pt',phase+'_schema.json']}))
        del arrays,sequence,ctx,y,mask,dataset,loader,model,optimizer,validation;gc.collect()
    write(folder/'complete.json',dict(protocol_sha256=w.ph,files={name:sha(folder/name) for name in
        ['inner.json','inner_model.pt','inner_schema.json','refit.json','refit_model.pt','refit_schema.json']}))
    return folder


def torch_prediction(folder,frame,inner=False):
    import torch
    prefix='inner' if inner else 'refit';schema=read(folder/(prefix+'_schema.json'))
    joint=schema['joint'];data=tensor_data(frame,joint,schema);model=torch_model(9 if joint else 1,len(schema['context']))
    model.load_state_dict(torch.load(folder/(prefix+'_model.pt'),map_location='cpu',weights_only=True));model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(data[0]),2048):
            value=model(torch.from_numpy(data[0][start:start+2048]),torch.from_numpy(data[1][start:start+2048]))
            out.append((torch.expm1(torch.nn.functional.softplus(value).clamp(max=20)) if joint else torch.sigmoid(value)).numpy())
    q=np.concatenate(out)
    return q[data[5],frame.h.to_numpy(int)-1] if joint else q[:,0]


def core_predictions(w,origin,inner=False,final=False,include_cnn=False):
    specs={s['id']:s for s in native_specs() if s['id'].startswith('core_')}
    if inner:frame=w.dataset(origin,'valid',{},final=final)
    else:frame=w.feature_frame(origin,w.schedule(origin)['test'])
    out={}
    for name,spec in specs.items():out[name]=native_prediction(fit_native(w,spec,origin,final),frame,inner=inner)
    values={'dynamics':dict(probability=out['core_event'],magnitude=out['core_positive'],lgbm_raw=out['core_log']),
            'mae':{'raw':out['core_mae']},'metric_count_lambda3':{'raw':out['core_count']},
            'metric_regular_recent_mae':{'raw':out['core_recent']}}
    from app.models.research_model import recipe_prediction
    out['recipe']=recipe_prediction(values,frame,w.recipe)
    names=['recipe','core_mae','core_count','core_recent','core_log']
    if include_cnn:
        # Complete dependencies first; never fit a router without its CNN inputs.
        event_folder=fit_torch(w,origin,False,final);joint_folder=fit_torch(w,origin,True,final)
        p=torch_prediction(event_folder,frame,inner=inner)
        neural_values={**values,'dynamics':{**values['dynamics'],'probability':p}}
        out['cnn_event_recipe']=recipe_prediction(neural_values,frame,w.recipe)
        out['cnn_joint9']=torch_prediction(joint_folder,frame,inner=inner)
        names+=['cnn_event_recipe','cnn_joint9']
    experts=np.column_stack([out[k] for k in names]);out['_expert_names']=names
    return frame,experts,out


def router_features(frame,experts):
    columns=['mean_28','zero_week_share_13','positive_cv2_13','h','stock_last','stock_zero_share_90','change_ratio_2_8']
    x=frame[columns].replace([np.inf,-np.inf],np.nan).fillna(0).reset_index(drop=True).copy()
    for i in range(experts.shape[1]):x[f'expert_{i}']=experts[:,i]
    return x


def fit_router(w,origin,final=False):
    import lightgbm as lgb
    folder=w.out/'models/learned_router'/('FINAL' if final else origin);folder.mkdir(parents=True,exist_ok=True)
    if checkpoint(folder,'complete.json',w.ph):return folder
    frame,experts,components=core_predictions(w,origin,inner=True,final=final,include_cnn=True)
    y=frame.target.to_numpy(float);absolute=np.abs(y[:,None]-experts);den=y[:,None]+experts
    smape=np.divide(200*absolute,den,out=np.zeros_like(absolute),where=den>0)
    loss=.7/60*smape+.3/50*100*absolute/max(y.mean(),1e-9)
    labels=np.argmin(loss,axis=1);x=router_features(frame,experts)
    weeks=sorted(pd.to_datetime(frame.week).unique());cut=weeks[max(0,int(len(weeks)*.7)-1)]
    m=pd.to_datetime(frame.week)<=cut
    if not m.any() or m.all():raise ValueError('Router requires a chronological meta train/validation split')
    params={**w.protocol_native,'objective':'multiclass','num_class':experts.shape[1],'metric':'multi_logloss'}
    ds=lgb.Dataset(x.loc[m],label=labels[m]);dv=lgb.Dataset(x.loc[~m],label=labels[~m],reference=ds)
    log('FIT router',origin,len(x),'honest earlier validation predictions')
    model=lgb.train(params,ds,num_boost_round=150,valid_sets=[dv],callbacks=[lgb.early_stopping(25,verbose=False)])
    iterations=int(model.best_iteration or model.current_iteration())
    model=lgb.train(params,lgb.Dataset(x,label=labels),num_boost_round=iterations)
    model.save_model(str(folder/'model.txt'))
    write(folder/'schema.json',dict(columns=list(x),experts=components['_expert_names'],
        objective='best expert under pointwise SMAPE/absolute-error proxy; not official total score',
        meta_source='inner-model early-stopping validation predictions, not refit in-sample predictions; stopping labels reused for meta training; all labels end before outer evaluation'))
    write(folder/'complete.json',dict(protocol_sha256=w.ph,iterations=iterations,files={n:sha(folder/n) for n in ['model.txt','schema.json']}))
    del frame,experts,x,ds,dv,model;gc.collect();return folder


def evaluate_router(w):
    import lightgbm as lgb
    predictions={}
    for origin in ORIGINS:
        folder=fit_router(w,origin);frame,experts,_=core_predictions(w,origin,include_cnn=True)
        model=lgb.Booster(model_file=str(folder/'model.txt'))
        weight=model.predict(router_features(frame,experts),num_threads=CONFIG['threads'])
        predictions[origin]=np.sum(weight*experts,axis=1)
    w.evaluate('learned_router',predictions,[29],dict(kind='router',rounding=None))
    for threshold in [.5,.65]:
        q={o:(np.rint(p) if threshold==.5 else np.floor(p+1-threshold)) for o,p in predictions.items()}
        w.evaluate('learned_router_round'+str(threshold),q,[29],dict(kind='router',rounding=threshold))


def reconcile(base,frame,article_target,warehouse_target,penalty,keep_zeros=True):
    ac,_=pd.factorize(frame.article.astype(str));wc,_=pd.factorize(frame.warehouse_name.astype(str))
    # Each future week has its own cross-classified reconciliation problem.
    p=base.copy()
    for h in range(1,10):
        m=frame.h.to_numpy(int)==h;original=base[m];q=np.maximum(original,1e-6)
        for _ in range(8):
            for code,target in [(ac[m],article_target[m]),(wc[m],warehouse_target[m])]:
                totals=np.bincount(code,weights=q);count=np.bincount(code)
                desired=np.divide(np.bincount(code,weights=target),count,out=np.zeros(len(count)),where=count>0)
                factor=np.divide(desired,totals,out=np.ones(len(totals)),where=totals>0)
                q*=np.clip(factor[code],.5,2.)**penalty
        p[m]=np.where(original>0,q,0.) if keep_zeros else q
    return p


def cold_fallback_study(w):
    if w.history is None:w.history=History()
    records=[]
    for origin in ORIGINS:
        frame,values,arrays=w.common.load(w.bundle,origin)
        held_path=ROOT/'artifacts/cold-pairs-20261029'/f'held_pairs_{origin}.csv'
        if not held_path.exists():raise RuntimeError('Fixed pseudo-cold pair catalogue missing')
        held=pd.read_csv(held_path,dtype={k:str for k in KEY})
        held_index=pd.MultiIndex.from_frame(held[KEY]);rows_index=pd.MultiIndex.from_frame(frame[KEY].astype(str))
        held_mask=rows_index.isin(held_index);cut=pd.Timestamp(origin)
        dates=pd.date_range(cut-pd.Timedelta(days=27),cut)
        volume=w.history.daily.reindex(columns=dates).fillna(0).sum(axis=1)/4
        # Completely remove held histories before article/warehouse aggregates.
        volume=volume.loc[~volume.index.isin(held_index)]
        observed=frame.drop_duplicates(KEY)[KEY+['category_lvl_1']].astype(str)
        observed=observed.loc[~pd.MultiIndex.from_frame(observed[KEY]).isin(held_index)]
        warm=observed.merge(volume.rename('weekly').reset_index(),on=KEY,how='left').fillna({'weekly':0.})
        article=warm.groupby('article',observed=True).weekly.mean();support=warm.groupby('article',observed=True).size()
        category=warm.groupby('category_lvl_1',observed=True).weekly.mean()
        warehouse=warm.groupby('warehouse_name',observed=True).weekly.mean();global_mean=max(warm.weekly.mean(),1e-9)
        target=frame.loc[held_mask];a=target.article.astype(str).map(article).to_numpy(float)
        c=target.category_lvl_1.astype(str).map(category).fillna(global_mean).to_numpy(float)
        n=target.article.astype(str).map(support).fillna(0).to_numpy(float)
        wh=target.warehouse_name.astype(str).map(warehouse).fillna(global_mean).to_numpy(float)/global_mean
        methods={'article_mean':np.nan_to_num(a,nan=global_mean),
            'category_shrink':(n*np.nan_to_num(a,nan=0)+3*c)/(n+3),
            'warehouse_category_shrink':(n*np.nan_to_num(a,nan=0)+3*c)/(n+3)*np.clip(wh,.5,2.)}
        composition=warm.pivot_table(index='warehouse_name',columns='category_lvl_1',values='weekly',aggfunc='sum').fillna(0)
        composition=composition.div(composition.sum(axis=1).replace(0,1),axis=0)
        donor=warm.pivot_table(index='article',columns='warehouse_name',values='weekly',aggfunc='mean')
        estimates={}
        for warehouse in composition.index:
            distances=((composition-composition.loc[warehouse])**2).sum(axis=1).drop(warehouse)
            nearest=distances.nsmallest(3).index;weights=1/(distances.reindex(nearest).to_numpy(float)+.1)
            values=donor.reindex(columns=nearest).to_numpy(float);known=np.isfinite(values)
            total=(known*weights).sum(axis=1)
            estimates[warehouse]=pd.Series(np.divide((np.nan_to_num(values)*weights).sum(axis=1),total,
                out=np.full(len(values),np.nan),where=total>0),index=donor.index)
        similar=np.array([estimates.get(str(warehouse),pd.Series(dtype=float)).get(str(article),np.nan)
            for warehouse,article in target[KEY].astype(str).itertuples(index=False,name=None)])
        methods['similar_warehouses_shrink']=np.where(np.isfinite(similar),.5*similar+.5*methods['category_shrink'],methods['category_shrink'])
        target_index=pd.MultiIndex.from_frame(target[KEY].astype(str))
        stock=w.history.stock.reindex(index=target_index,columns=dates).to_numpy(float)
        observed=np.isfinite(stock).sum(axis=1)
        available=np.divide((np.isfinite(stock)&(stock>0)).sum(axis=1),observed,
            out=np.ones(len(stock)),where=observed>0)
        methods['similar_warehouses_availability']=methods['similar_warehouses_shrink']*np.minimum(2.,1/np.maximum(available,.25))
        for name,q in methods.items():
            p=np.rint(np.maximum(q,0));m=w.common.metrics(target.actual,p,w.common.codes(target))
            records.append(dict(origin=origin,method=name,rows=len(target),metrics=m))
    write(w.out/'cold_fallback.json',dict(records=records,task=34,
        protocol='fixed pseudo-cold histories removed before aggregates; separate cohort, excluded from warm full-catalog ranking',
        production_changed=False,independent_confirmation=False))
    w.mark(34,'completed','pseudo-cold article/category/similar-warehouse shrinkage and observed stock availability; separate cohort')


def run_architectures(w,specs,raw):
    # Point 19: independent total and normalized weekly profile.
    for weight in [.25,.5,1.]:
        for post in ['raw','round','base_zeros']:
            predictions={}
            for o in ORIGINS:
                frame,_,arrays=w.common.load(w.bundle,o);code=w.common.codes(frame)
                shares=np.maximum(raw['profile9'][o],0);sums=np.bincount(code,weights=shares)
                share=np.divide(shares,sums[code],out=np.full(len(shares),1/9),where=sums[code]>0)
                q=raw['total9'][o]*share;p=(1-weight)*arrays['baseline']+weight*q
                if post=='round':p=np.rint(p)
                if post=='base_zeros':p=np.where(arrays['baseline']>0,p,0.)
                predictions[o]=p
            w.evaluate('total9_profile_w'+str(weight)+'_'+post,predictions,[19],dict(kind='total_profile',weight=weight,post=post))
    w.mark(19,'completed','separate total9 and fraction models, normalized profile; all labels past-only')
    # Point 33: actual parent forecasts, not future facts or fixed past totals.
    for penalty in [.1,.25,.5]:
        predictions={}
        for o in ORIGINS:
            frame,_,arrays=w.common.load(w.bundle,o)
            predictions[o]=reconcile(arrays['baseline'],frame,raw['parent_article'][o],raw['parent_warehouse'][o],penalty)
        w.evaluate('parent_reconcile_'+str(penalty),predictions,[33],dict(kind='reconcile',penalty=penalty))
    w.mark(33,'completed','independent article/warehouse forecasts plus soft cross-classified reconciliation')
    # Point 38: jointly replace all six underlying boosters with refreshed iterations.
    predictions={o:core_predictions(w,o)[2]['recipe'] for o in ORIGINS}
    w.evaluate('refreshed_entire_recipe',predictions,[38],dict(kind='refreshed'))
    w.mark(38,'completed','six underlying components reselected on inner time validation and refit')
    evaluate_router(w);w.mark(29,'completed','router trained on inner early-stopping forecasts, not outer outcomes; shared stopping labels disclosed')
    # Points 30/31: a different use of CNN, and a genuine joint 9-output model.
    for joint,task in [(False,30),(True,31)]:
        predictions={};raw_values={}
        for o in ORIGINS:
            folder=fit_torch(w,o,joint);evaluation=w.feature_frame(o,w.schedule(o)['test'])
            q=torch_prediction(folder,evaluation);raw_values[o]=q
            frame,values,arrays=w.common.load(w.bundle,o)
            if joint:predictions[o]=q
            else:
                from app.models.research_model import recipe_prediction
                v=dict(values);v['dynamics']={**values['dynamics'],'probability':q}
                predictions[o]=recipe_prediction(v,frame,w.recipe)
        w.evaluate('cnn_joint9' if joint else 'cnn_event',predictions,[task],dict(kind='cnn',joint=joint))
        if joint:
            for threshold in [.5,.65]:
                p={o:(np.rint(q) if threshold==.5 else np.floor(q+1-threshold)) for o,q in predictions.items()}
                w.evaluate('cnn_joint9_round'+str(threshold),p,[31],dict(kind='cnn',joint=True,rounding=threshold))
            p={o:np.where(w.common.load(w.bundle,o)[2]['baseline']>0,q,0.) for o,q in predictions.items()}
            w.evaluate('cnn_joint9_base_zeros',p,[31],dict(kind='cnn',joint=True,keep_base_zeros=True))
        if not joint:
            count_predictions={}
            for o in ORIGINS:
                frame,values,arrays=w.common.load(w.bundle,o);p=raw_values[o]
                old=arrays['count_probability'];positive=old[:,1:];mass=positive.sum(axis=1,keepdims=True)
                conditional=np.divide(positive,mass,out=np.full_like(positive,1/positive.shape[1]),where=mass>0)
                probability=np.c_[1-p,conditional*p[:,None]]
                fit=read(w.bundle/'data'/f'{o}.count.json')
                values=dict(values);values['metric_count_lambda3']={'raw':w.common.bayes_quantities(probability,fit['representatives'],fit['training_mean'],3.)}
                from app.models.research_model import recipe_prediction
                count_predictions[o]=recipe_prediction(values,frame,w.recipe)
            w.evaluate('cnn_event_count_zero',count_predictions,[30],dict(kind='cnn',joint=False,route='count_zero'))
        w.mark(task,'completed','CPU CNN with masked 52-week history; temporal validation and epoch checkpoints')


def choose_and_ensemble(w):
    # Collapse duplicate IDs after resume; every score uses the same two full windows.
    w.records=list({r['id']:r for r in w.records}.values())
    baseline={o:w.common.load(w.bundle,o)[2]['baseline'] for o in ORIGINS}
    w.evaluate('baseline',baseline,[40],dict(kind='baseline'))
    # A resumed run must not grow a new generation of blends from old blends.
    standalone=[r for r in w.records if r['details'].get('kind')!='ensemble']
    ordered=sorted(standalone,key=lambda r:(r['weighted_ratio'],r['id']))
    # Small declared blend grid; only already fitted, complete components may enter.
    pool=ordered[:5]
    for i,a in enumerate(pool):
        for b in pool[i+1:]:
            for weight in [.25,.5,.75]:
                name='blend_'+hashlib.sha256((a['id']+'|'+b['id']+'|'+str(weight)).encode()).hexdigest()[:12]
                p={o:(1-weight)*w.saved_prediction(a['id'],o)+weight*w.saved_prediction(b['id'],o) for o in ORIGINS}
                w.evaluate(name,p,[32],dict(kind='ensemble',left=a['id'],right=b['id'],weight=weight))
                rounded={o:np.rint(q) for o,q in p.items()}
                w.evaluate(name+'_round',rounded,[32],dict(kind='ensemble',left=a['id'],right=b['id'],weight=weight,rounding=.5))
    w.mark(32,'completed','small fixed blend grid over complete standalone candidates, no missing dependencies')
    w.records=list({r['id']:r for r in w.records}.values())
    groups={}
    for r in w.records:
        signature=tuple(round(r['metrics'][o][k],8) for o in ORIGINS for k in METRICS)
        groups.setdefault(signature,[]).append(r)
    distinct=list(groups.values());scores=np.zeros(len(distinct))
    for o in ORIGINS:
        for weight,k in zip(WEIGHTS,METRICS):
            rank=pd.Series([g[0]['metrics'][o][k] for g in distinct]).rank(method='average').to_numpy()
            scores+=weight*(rank-1)/max(len(distinct)-1,1)/len(ORIGINS)
    for score,group in zip(scores,distinct):
        for r in group:r['local_rank_score']=float(score)
    eligible=[r for r in w.records if r['worst_window_ratio']<=1.01]
    best=min(eligible,key=lambda r:r['weighted_ratio']);robust=min(eligible,key=lambda r:(r['worst_window_ratio'],r['weighted_ratio']))
    rank=min(eligible,key=lambda r:r['local_rank_score'])
    result=dict(best_mean=best,best_worst_window=robust,best_local_rank=rank,records=sorted(w.records,key=lambda r:r['weighted_ratio']),
        metric_weights=dict(zip(METRICS,WEIGHTS)),independent_confirmation=False,
        warning='Different normalizations may choose different winners; local ranks are not external leaderboard ranks',
        production_changed=False)
    write(w.out/'SELECTION.json',result)
    pd.DataFrame([dict(id=r['id'],weighted_ratio=r['weighted_ratio'],worst_window_ratio=r['worst_window_ratio'],local_rank_score=r['local_rank_score'])
        for r in result['records']]).to_csv(w.out/'ranking.csv',index=False)
    w.mark(40,'completed','mean, worst-window and fixed-pool rank winners saved separately; no hidden-test claim')
    return result


def feature_selection(w,k,final=False,family='quantity_all'):
    control=next(s for s in native_specs() if s['id']==family)
    predictions={}
    for origin in (['2026-07-26'] if final else ORIGINS):
        model=fit_native(w,control,origin,final=final)
        importance=read(model/'importance.json')
        required=[c for c in CAT+['h'] if c in importance]
        columns=list(dict.fromkeys(required+[c for c,v in sorted(importance.items(),key=lambda item:(-item[1],item[0]))[:k]]))
        spec={**control,'id':'selected_'+family+'_'+str(k),'tasks':[39],'selected_features':columns}
        folder=fit_native(w,spec,origin,final=final)
        if final:return folder
        frame=w.feature_frame(origin,w.schedule(origin)['test']);predictions[origin]=native_prediction(folder,frame)
    for weight in [.25,.5]:
        outputs={}
        for origin in ORIGINS:
            frame,values,arrays=w.common.load(w.bundle,origin);p=arrays['baseline'].copy();q=predictions[origin]
            if control['kind']=='binary':
                from app.models.research_model import recipe_prediction
                values={**values,'dynamics':{**values['dynamics'],'probability':q}}
                q=recipe_prediction(values,frame,w.recipe);p=(1-weight)*p+weight*q
            else:
                r=frame.mean_28.to_numpy(float)*7
                m=((r>=.5)&(r<10)) if control['kind']=='count' else r>=2
                p[m]=(1-weight)*p[m]+weight*q[m]
            outputs[origin]=p
        w.evaluate(f'feature_selected_{family}_{k}_w{weight}',outputs,[39],
            dict(kind='feature_selection',family=family,k=k,weight=weight))


def error_diagnostics(w,selection):
    best=selection['best_mean'];reports={}
    for origin in ORIGINS:
        frame,_,arrays=w.common.load(w.bundle,origin);p=w.saved_prediction(best['id'],origin)
        before=w.common.diagnostics(frame,arrays['baseline']);after=w.common.diagnostics(frame,p)
        reports[origin]=dict(before=before,after=after)
        table=frame[ROWKEY+['actual','h','mean_28']].copy()
        table['baseline']=arrays['baseline'];table['candidate']=p
        table['squared_error_increase']=(p-frame.actual.to_numpy(float))**2-(arrays['baseline']-frame.actual.to_numpy(float))**2
        table.sort_values('squared_error_increase',ascending=False).head(150).to_csv(w.out/f'{origin}_largest_regressions.csv',index=False)
    write(w.out/'error_diagnostics.json',dict(origins=reports,task=18,
        caveat='Historical row-level facts only. Closed-test error attribution is unavailable.'))


def final_dependencies(w,record,seen=None):
    if seen is None:seen=set()
    if record['id'] in seen:return
    seen.add(record['id']);d=record['details'];kind=d.get('kind')
    log('FINAL dependencies',record['id'])
    if kind=='native_overlay':fit_native(w,d['spec'],'2026-07-26',final=True)
    elif kind=='feature_selection':feature_selection(w,d['k'],final=True,family=d['family'])
    elif kind=='cnn':fit_torch(w,'2026-07-26',d.get('joint',False),final=True)
    elif kind=='router':fit_router(w,'2026-07-26',final=True)
    elif kind=='refreshed':
        for spec in native_specs():
            if spec['id'].startswith('core_'):fit_native(w,spec,'2026-07-26',final=True)
    elif kind in ['total_profile','reconcile']:
        names=['total9','profile9'] if kind=='total_profile' else ['parent_article','parent_warehouse']
        for spec in native_specs():
            if spec['id'] in names:fit_native(w,spec,'2026-07-26',final=True)
    elif kind=='ensemble':
        lookup={r['id']:r for r in w.records}
        for key in ['left','right']:final_dependencies(w,lookup[d[key]],seen)
    elif 'source_zip' in d:
        variant=next(v for v in w.event_protocol['variants'] if v['id']==d['variant'])
        for head in variant['heads']:
            spec=dict(id='final_returned_'+variant['id']+'_'+head,kind='binary',tasks=[variant['task']],
                selected_features=variant['features'],groups=['event'],objective='binary',head_filter=head,
                max_iterations=400,early_stopping=40)
            # Same feature and mask definitions as the completed round09 experiment.
            fit_native(w,spec,'2026-07-26',final=True)
    elif kind=='baseline' or 'policy' in d:
        pass  # All required baseline members are already fitted through July 26.
    else:raise ValueError('Final deployment dependency is not implemented: '+repr(d))


def package_results(w,selection):
    export=w.out/'export';export.mkdir(exist_ok=True)
    # Export predictions/models/reports, never raw CSV or feature caches.
    for folder in ['models','evaluations','predictions']:
        source=w.out/folder
        if source.exists():
            target=export/folder
            if not target.resolve().is_relative_to(w.out.resolve()):raise ValueError('Export path escaped run directory')
            shutil.copytree(source,target,dirs_exist_ok=True)
    for path in w.out.glob('*'):
        if path.is_file() and path.suffix in ['.json','.csv','.md','.log']:shutil.copy2(path,export/path.name)
    final=export/'baseline_model';final.mkdir(exist_ok=True)
    for path in (ROOT/'model').iterdir():
        if path.is_file():shutil.copy2(path,final/path.name)
    shutil.copy2(Path(__file__),export/'train_all_improvements.py')
    manifest={p.relative_to(export).as_posix():sha(p) for p in export.rglob('*') if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    write(export/'OUTPUT_MANIFEST.json',dict(protocol_sha256=w.ph,files=manifest,production_changed=False))
    archive=w.out/'all40_results.zip';temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in export.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(export))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise ValueError('Corrupt final archive')
    temporary.replace(archive);log('DONE. SEND THIS FILE:',archive)


def run_all(w):
    w.resume_evaluations();w.prior_rounds();w.discover_groups()
    specs=native_specs();raw={}
    for spec in specs:
        log('EXPERIMENT',spec['id'],'tasks',spec['tasks'])
        raw[spec['id']]=evaluate_native(w,spec)
        for task in spec['tasks']:
            w.mark(task,'in_progress','completed native candidate '+spec['id']+'; awaiting related comparisons')
    for task in [14,15,16,17,20,21,22,23,24,25,26,27,35,37]:
        w.mark(task,'completed','declared native experiments on both fixed windows; full predictions and models saved')
    # Denser historical origins are a separate, fully runnable candidate.
    dense=dict(id='dense_origins',tasks=[36],kind='quantity',groups=['all'],objective='regression_l1',dense=True)
    evaluate_native(w,dense);w.mark(36,'completed','14-day origins over the same chronological training span; duplicate-label weighting')
    selection_families=['quantity_all']+[s['id'] for s in specs if s['id'].startswith('core_')]
    for family in selection_families:
        for k in CONFIG['feature_top_k']:feature_selection(w,k,family=family)
    w.mark(39,'completed','individual inner-gain selection for every core expert and all-feature quantity model; new inner/refit models evaluated')
    run_architectures(w,specs,raw);cold_fallback_study(w)
    selection=choose_and_ensemble(w);error_diagnostics(w,selection)
    for task in [13,18,28]:w.mark(task,'completed','cached finite experiments retained; diagnostics updated against selected candidate')
    if CONFIG['final_refit']:
        winners={r['id']:r for r in [selection['best_mean'],selection['best_worst_window'],selection['best_local_rank']]}
        seen=set()
        for record in winners.values():final_dependencies(w,record,seen)
        write(w.out/'FINAL_MODELS.json',dict(winners=list(winners),records=list(winners.values()),
            label_end='2026-07-26',early_stopping_origin='2026-05-24',
            preserved_baseline_sha256=BASE_MODEL_SHA,source_model_unchanged=True,
            native_artifacts='models/*/FINAL',deployment_ready=False,
            note='Native components are fitted. Integrate graph and verify serving before replacing production model.'))
    missing=[int(k) for k,v in w.progress.items() if not v['status'].startswith('completed')]
    if missing:raise RuntimeError('Plan incomplete, refusing success archive: '+repr(missing))
    write(w.out/'RUN_SUMMARY.json',dict(completed_tasks=list(range(1,41)),config=CONFIG,
        model_families='native LightGBM, binary CNN, joint9 CNN, learned router, total/profile, parents, cold fallback',
        independent_confirmation=False,production_changed=False,final_refit=CONFIG['final_refit'],
        no_external_business_data=True,baseline_sha256=BASE_MODEL_SHA))
    package_results(w,selection)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Validate packages, hashes and one feature frame; NEVER train')
    args=parser.parse_args()
    if dependencies(args.check):return
    global np,pd
    import numpy as np
    import pandas as pd
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    w=Workbench(check=args.check)
    if args.check:
        w.discover_groups()
        frame=w.feature_frame(ORIGINS[0],w.schedule(ORIGINS[0])['test'])
        for spec in native_specs():
            if not spec['kind'].startswith('parent_'):
                if not set(w.columns(spec))<=set(frame):raise ValueError('Missing columns for '+spec['id'])
        log('CHECK ONLY PASSED. No model fitting executed.');return
    with run_lock(w.out/'.run.lock'):
        archive=w.out/'all40_results.zip'
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
                if manifest['protocol_sha256']!=w.ph:raise ValueError('Completed archive protocol differs')
                for name,digest in manifest['files'].items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Completed archive changed: '+name)
            log('ALREADY COMPLETED. SEND:',archive);return
        try:run_all(w)
        except BaseException:
            (w.out/'LAST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            log('STOPPED. Completed checkpoints preserved. Send LAST_ERROR.txt from',w.out)
            raise


if __name__=='__main__':main()
