"""Refit an explicitly CV-tuned recipe on permitted history; stage, never promote."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from app.csv_io import read_csv_checked
from app.features import KEY, CAT_FEATURES
from app.event_features import make_event_features
from app.models.research_model import WeeklyResearchModel, recipe_member_names
from app.models.metric_experts import MetricExpertPart
from app.predictor import Predictor
from scripts.confirm_smape_wape import expert_specs
from scripts.research_weekly import frozen_record, freeze_record, read, sha
from train import save_artifact

CV=ROOT/'artifacts/smape-wape-adaptive-cv-20261029'
CACHE=ROOT/'artifacts/improvements-20261028-r1/final-cache'


class WeeklyLagHistory:
    def __init__(self,sales):
        sales=sales[['date',*KEY,'quantity']].copy()
        sales['date']=pd.to_datetime(sales.date)
        for key in KEY:sales[key]=sales[key].astype(str)
        self.start=sales.date.min()
        self.first=sales.groupby(KEY,observed=True).date.min()
        sales['week']=sales.date-pd.to_timedelta(sales.date.dt.weekday,unit='D')
        self.weekly=sales.groupby([*KEY,'week'],observed=True).quantity.sum().unstack('week')

    def add(self,frame,origin):
        origin=pd.Timestamp(origin)
        end=origin-pd.Timedelta(days=(origin.weekday()+1)%7)
        weeks=pd.date_range(end-pd.Timedelta(days=90),end-pd.Timedelta(days=6),freq='7D')
        pairs=pd.MultiIndex.from_frame(frame[KEY].drop_duplicates())
        y=self.weekly.reindex(index=pairs,columns=weeks).fillna(0).to_numpy(dtype='float32')
        first=self.first.reindex(pairs).fillna(origin+pd.Timedelta(days=1)).to_numpy(dtype='datetime64[ns]')
        valid=(weeks.to_numpy()[None,:]>=first[:,None]) & (weeks.to_numpy()[None,:]>=self.start.to_datetime64())
        y[~valid]=np.nan
        extra=pd.DataFrame(index=pairs)
        for lag in range(1,14):extra[f'weekly_lag_{lag}']=y[:,-lag]
        extra['weekly_support_13']=np.isfinite(y).sum(axis=1)
        result=frame.merge(extra.reset_index(),on=KEY,how='left',validate='many_to_one')
        derived=make_event_features(result)
        result=pd.concat([result,derived[[c for c in derived if c not in result]]],axis=1)
        result['feature_origin']=origin
        return result


def initialize(out):
    selection=frozen_record(CV/'selection.json')
    if not selection['accepted']:raise ValueError('No simultaneous-primary CV candidate')
    if (out/'protocol.json').exists():
        protocol=frozen_record(out/'protocol.json')
        for name,digest in protocol['files'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen finalization source changed: '+name)
        return selection,protocol
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty finalization directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(CACHE/'manifest.json')
    paths=[Path(__file__),CV/'selection.json',CV/'protocol.json',CACHE/'manifest.json',
           ROOT/'data/sales_dataset_hakaton_train.csv',*[ROOT/'model'/n for n in ['model.pkl','config.json','preprocessor.pkl']],
           ROOT/'app/models/metric_experts.py',ROOT/'app/models/research_model.py',ROOT/'app/sequence_features.py',ROOT/'train.py']
    for name,digest in manifest.items():
        path=CACHE/name
        if sha(path)!=digest:raise ValueError('Final training cache changed')
        paths.append(path)
    iterations=frozen_record(ROOT/'artifacts/smape-wape-confirmation-20261029/iterations.json')
    extra=frozen_record(CV/'may_recent_balanced_mae.json')
    iterations['metric_regular_recent_balanced_mae']=extra['iterations']
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),recipe=selection['recipe'],
        expert_specs=expert_specs(selection['recipe']),iterations=iterations,history_cutoff='2026-07-26',
        evaluation_kind=selection['evaluation_kind'],independent_confirmation=False,
        strategy='reuse exact full-data dynamics champion; refit chosen metric experts on the same sixteen allowed origins',
        files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},training_cache=list(sorted(manifest)))
    freeze_record(out/'protocol.json',protocol)
    return selection,protocol


def finalize(out,selection,protocol):
    if (out/'final_training.json').exists():return frozen_record(out/'final_training.json')
    model_dir=out/'model'
    if model_dir.exists() and any(model_dir.iterdir()):raise ValueError('Partial final model exists; preserve it and choose a fresh directory')
    sales=read_csv_checked(ROOT/'data/sales_dataset_hakaton_train.csv',usecols=['date',*KEY,'quantity'],
                           parse_dates=['date'],dtype={'quantity':'float32'})
    sales=sales.loc[sales.date<=protocol['history_cutoff']]
    history=WeeklyLagHistory(sales)
    del sales;gc.collect()
    # Prove the dense-free builder equals the original masked cache algorithm.
    import research_events as events
    parity={}
    for origin in ['2026-01-18','2026-03-22','2026-05-24']:
        cached=events.cached(origin,f'{origin}-False.parquet',labels=False)
        columns=[f'weekly_lag_{i}' for i in range(1,14)]+['weekly_support_13']
        added=history.add(cached.drop(columns=columns),origin)
        np.testing.assert_allclose(added[columns],cached[columns],rtol=1e-6,atol=1e-6,equal_nan=True)
        parity[origin]={'rows':len(cached),'lag_values_equal':True}
    freeze_record(out/'lag_builder_parity.json',parity)
    frames=[]
    for name in protocol['training_cache']:
        frame=pd.read_parquet(CACHE/name)
        if (pd.to_datetime(frame.week)+pd.Timedelta(days=6)).max()>pd.Timestamp(protocol['history_cutoff']):
            raise ValueError('Final training target exceeds authorized cutoff')
        frame=history.add(frame,name[:10])
        frames.append(frame)
    del history;gc.collect()
    train=pd.concat(frames,ignore_index=True)
    del frames;gc.collect()
    base=Predictor(ROOT/'model').load().model
    model=WeeklyResearchModel({'recipe':protocol['recipe'],'expert_specs':protocol['expert_specs'],
                               'threads':6,'estimators':600,'seed':42})
    names=recipe_member_names(protocol['recipe'])
    for name in names-set(protocol['expert_specs']):
        if name not in base.members:raise ValueError('Unexpected unfitted legacy member')
        model.members[name]=base.members[name]
    for name,spec in protocol['expert_specs'].items():
        print('FINAL ALLOWED-HISTORY FIT',name,len(train),protocol['iterations'][name],flush=True)
        model.members[name]=MetricExpertPart(spec,6).fit(train,iterations=protocol['iterations'][name])
        gc.collect()
    model.check_schema()
    rows=len(train);del train;gc.collect()
    validation={'scheme':'adaptive_historical_rolling_cv_previously_inspected',
        'independent_confirmation':False,'selection_used_all_reported_origins':True,
        'origins':selection['metrics'],'mean_ratios':selection['mean_ratios'],
        'metrics':selection['metrics']['2026-05-24']}
    config=save_artifact(model,model_dir,'2026-07-26','2026-07-26',rows,len(protocol['training_cache']),
        validation=validation,provenance={'cv_selection_sha256':sha(CV/'selection.json'),
            'finalization_protocol_sha256':sha(out/'protocol.json'),'history_cutoff':'2026-07-26'})
    Predictor(model_dir).load()
    result={'rows':rows,'cutoffs':len(protocol['training_cache']),'iterations':model.selected_iterations,
        'model_sha256':config['model_artifact_sha256'],'preprocessor_sha256':config['preprocessor_sha256'],
        'config_sha256':sha(model_dir/'config.json'),'production_replaced':False,
        'independent_confirmation':False,'evaluation_kind':selection['evaluation_kind']}
    freeze_record(out/'final_training.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/smape-wape-final-20261029')
    args=parser.parse_args(argv);out=args.out.resolve()
    selection,protocol=initialize(out)
    print(finalize(out,selection,protocol),flush=True)
    initialize(out)


if __name__=='__main__':main()
