"""Frozen weekly ablations: development -> one confirmation -> final refit.

No final weights are used for historical scoring. All fit labels end by origin;
early-stopping targets are disjoint from training targets. Previously inspected
history is explicitly acknowledged rather than claimed as a pristine holdout.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

CONTAINER=Path(__file__).resolve().parents[1]
ROOT=CONTAINER.parent
sys.path.insert(0,str(CONTAINER))
from app.features import KEY,CAT_FEATURES,build_panel,make_features,make_training_frame,forecast_end
from app.enhancements import add_money,augment_features,feature_columns,enhanced_spec
from app.models.research_model import fit_member,member_values,recipe_prediction,WeeklyResearchModel
from train import make_cutoffs,save_artifact

MEMBERS=['base','calendar','dynamics','monetary','all','poisson','tweedie','mae','quantile','catboost','grouped']
NAMES=['smape','wape','rmse','bias','rmsle']
WEIGHTS=np.array([.5,.2,.15,.1,.05])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def candidate_grid():
    recipes={}
    def add(recipe):
        key=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:12]
        recipes[key]=recipe
    add({'source':'base','post':{'round_int':True}})
    for source in MEMBERS[:5]:
        for w,g,s,t in itertools.product([0.,.3,.6,1.],[.35,.5,.65],[.9,1.,1.15],[.35,.5,.65]):
            add(dict(source=source,post=dict(weight_hurdle=w,gate=g,scale=s,round_int=True,round_threshold=t)))
        for medium,regular,t in itertools.product([.3,.4,.5,.6],[.9,1.1,1.2],[.5,.65]):
            add(dict(source=source,post=dict(medium_gate=medium,regular_scale=regular,round_int=True,round_threshold=t)))
        for rare,regular,t in itertools.product([.35,.5,.65],[.9,1.1,1.2],[.5,.65]):
            add(dict(source=source,post=dict(rare_gate=rare,weekly_regular_scale=regular,round_int=True,round_threshold=t)))
        for scale in [1.,1.1,1.2]:
            add(dict(source=source,smearing=scale,post=dict(round_int=True)))
        for b in [.1,.2]:add(dict(source=source,post=dict(mean_weight=b,round_int=True)))
        add(dict(source=source,post=dict(expected=True,round_int=True)))
        add(dict(source=source,post=dict(apportion=True)))
        add(dict(source=source,post=dict(round_int=False)))
    for source in MEMBERS[5:]+['tsb','sba']:
        for s,t in itertools.product([.8,1.,1.2],[.35,.5,.65]):
            add(dict(source=source,post=dict(scale=s,round_int=True,round_threshold=t)))
    for source,aux,w,t in itertools.product(['base','all'],MEMBERS[5:]+['tsb','sba'],[.1,.3],[.5,.65]):
        add(dict(source=source,aux=aux,aux_weight=w,post=dict(round_int=True,round_threshold=t)))
    return recipes


def source_paths():
    return sorted(list((CONTAINER/'app').rglob('*.py'))+[CONTAINER/'train.py',Path(__file__).resolve(),
                  CONTAINER/'requirements.lock.txt'])


def verify_protocol(out):
    protocol=read(out/'protocol.json')
    if sha(out/'protocol.json')!=(out/'protocol.sha256').read_text().strip():raise ValueError('Frozen protocol changed')
    if sha(out/'candidate_grid.json')!=protocol['candidate_grid_sha256']:raise ValueError('Frozen grid changed')
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths()}
    if actual!=protocol['source_hashes']:raise ValueError('Frozen ML sources changed; use a new run directory')
    for name,h in protocol['input_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen input changed: '+name)
    return protocol


def initialize(out):
    if (out/'protocol.json').exists():return verify_protocol(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty research directory')
    out.mkdir(parents=True,exist_ok=True)
    write(out/'candidate_grid.json',candidate_grid())
    protocol=dict(research_version=1,created_on='2026-10-28',history_cutoff='2026-07-26',
                  development_origins=['2025-10-19','2026-01-18','2026-03-22'],confirmation_origin='2026-05-24',
                  final_training_end='2026-07-26',granularity='week',forecast_weeks=9,
                  prior_history_inspected=True,confirmation_status='historical_out_of_sample_previously_inspected_period',
                  selection_rule='equal-origin mean weighted ratio to refitted rounded base; mean<=0.995, mean SMAPE ratio<=1.01, >=3/5 mean metrics improve; each-origin SMAPE ratio<=1.05',
                  confirmation_rule='one frozen candidate only; local ratio<=0.995, SMAPE<=base*1.01, >=3/5 metrics improve; otherwise retain original production',
                  calendar_hypothesis='federal only; anonymous dates/regions unconfirmed; publication-asof transfers',
                  official_pairs_available=False,seed=42,threads=6,estimators=600,max_cutoffs=16,early_stopping_rounds=60,
                  members=MEMBERS,enhancements=enhanced_spec(),
                  candidate_grid_sha256=sha(out/'candidate_grid.json'),
                  input_hashes={f'data/{n}_dataset_hakaton_train.csv':sha(ROOT/f'data/{n}_dataset_hakaton_train.csv') for n in ['sales','stock']},
                  source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths()})
    write(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    print('PROTOCOL FROZEN',len(candidate_grid()),'recipes',flush=True)
    return protocol


def load_data():
    path=ROOT/'data/sales_dataset_hakaton_train.csv'
    cols=set(pd.read_csv(path,nrows=0).columns)
    use=['date','quantity',*CAT_FEATURES]+[c for c in ['sales_sum','discount_sum'] if c in cols]
    sales=pd.read_csv(path,usecols=use,parse_dates=['date'],dtype={**{c:'category' for c in CAT_FEATURES},'quantity':'float32'})
    path=ROOT/'data/stock_dataset_hakaton_train.csv';cols=set(pd.read_csv(path,nrows=0).columns)
    item='article' if 'article' in cols else 'char_article'
    stock=pd.read_csv(path,usecols=['date','warehouse_name',item,'stock'],parse_dates=['date'],dtype={'warehouse_name':'category',item:'category','stock':'float32'}).rename(columns={item:'article'})
    return sales.loc[sales.date<='2026-07-26'].copy(),stock.loc[stock.date<='2026-07-26'].copy()


def fast_metrics(y,p,codes):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid forecast')
    count=np.bincount(codes);actual=np.bincount(codes,weights=y);pred=np.bincount(codes,weights=p)
    den=(y+p)/2;rat=np.divide(np.abs(y-p),den,out=np.zeros(len(y)),where=den>0)
    smape=(np.bincount(codes,weights=rat)/count).mean()*100
    err=np.abs(y-p).sum();total=y.sum();positive=actual>0
    return dict(smape=float(smape),wape=float(err/total*100 if total else (100 if err else 0)),
                rmse=float(np.sqrt(np.mean((y-p)**2))),bias=float((np.abs(pred[positive]-actual[positive])/actual[positive]).mean()*100) if positive.any() else 0.,
                rmsle=float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2))),
                volume_bias=float((p.sum()/total-1)*100) if total else 0.,zero_share=float(np.mean(p==0)))


def ratio(m,baseline):return float(sum(w*m[k]/max(baseline[k],1e-12) for w,k in zip(WEIGHTS,NAMES)))


def feature_frame(panel,cutoff,cache,training=True):
    path=cache/f'{pd.Timestamp(cutoff).date()}-{training}.parquet'
    manifest=read(cache/'manifest.json') if (cache/'manifest.json').exists() else {}
    if path.exists():
        if manifest.get(path.name)!=sha(path):raise ValueError('Cached frame changed: '+str(path))
        return pd.read_parquet(path)
    complete=[h for h in range(1,10) if forecast_end(cutoff,h)<=panel.date.max()]
    base=make_training_frame(panel,cutoff,complete) if training else make_features(panel,cutoff)
    frame=augment_features(panel,base,cutoff)
    for c in CAT_FEATURES:frame[c]=frame[c].astype('category')
    cache.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)
    manifest[path.name]=sha(path);write(cache/'manifest.json',manifest)
    return frame


def fold(out,origin,sales,stock,names):
    cut=pd.Timestamp(origin);directory=out/origin;directory.mkdir(exist_ok=True)
    cache=directory/'cache';started=time.perf_counter()
    all_done=(directory/'complete.json').exists()
    if all_done:
        result=read(directory/'complete.json')
        if set(names)<=set(result['members']):
            for p,h in result['files'].items():
                if sha(directory/p)!=h:raise ValueError('Fold output changed: '+p)
            return pd.read_parquet(directory/'evaluation.parquet'),pickle.loads((directory/'values.pkl').read_bytes()),result
    past_sales=sales.loc[sales.date<=cut];past_stock=stock.loc[stock.date<=cut]
    panel=add_money(build_panel(past_sales,past_stock),past_sales)
    selection=cut-pd.Timedelta(weeks=9)
    cutoffs=make_cutoffs(panel)
    eligible=[c for c in cutoffs if forecast_end(c,9)<=selection][-16:]
    if not eligible:raise ValueError('Insufficient purged training history')
    train=pd.concat([feature_frame(panel,c,cache) for c in eligible],ignore_index=True)
    valid=feature_frame(panel,selection,cache)
    assert (train.week+pd.Timedelta(days=6)).max()<=selection
    assert (valid.week+pd.Timedelta(days=6)).max()<=cut
    iteration_file=directory/'iterations.json';iterations=read(iteration_file) if iteration_file.exists() else {}
    if iterations and sha(iteration_file)!=read(directory/'iteration_metadata.json')['sha256']:raise ValueError('Selected iterations changed')
    for name in names:
        if name in iterations:continue
        print('EARLY STOP',origin,name,len(train),flush=True)
        model=fit_member(name,train,valid)
        iterations[name]=model.selected_iterations;write(iteration_file,iterations)
        write(directory/'iteration_metadata.json',dict(sha256=sha(iteration_file)))
        del model;gc.collect()
    del train,valid;gc.collect()
    refit=pd.concat([feature_frame(panel,c,cache) for c in cutoffs[-16:]],ignore_index=True)
    assert (refit.week+pd.Timedelta(days=6)).max()<=cut
    frame=feature_frame(panel,cut,cache,False)
    # Only this stage's allowed truth is assembled, after the historical fit split.
    truth=sales.loc[(sales.date>cut)&(sales.date<=forecast_end(cut,9)),['date',*KEY,'quantity']].copy()
    truth['week']=truth.date-pd.to_timedelta(truth.date.dt.weekday,unit='D')
    truth=truth.groupby(['week',*KEY],observed=True,as_index=False).quantity.sum()
    frame=frame.merge(truth,on=['week',*KEY],how='left',validate='one_to_one').rename(columns={'quantity':'actual'})
    frame['actual']=frame.actual.fillna(0)
    values={};model_files=[]
    progress=read(directory/'member_records.json') if (directory/'member_records.json').exists() else {}
    for name in names:
        model_file=directory/f'{name}.pkl';value_file=directory/f'{name}-values.pkl'
        if model_file.exists() and value_file.exists():
            if progress.get(model_file.name)!=sha(model_file) or progress.get(value_file.name)!=sha(value_file):raise ValueError('Partial member files changed')
            values[name]=pickle.loads(value_file.read_bytes());model_files.extend([model_file,value_file]);continue
        print('PAST ONLY REFIT',origin,name,len(refit),flush=True)
        model=fit_member(name,refit,iterations=iterations[name])
        values[name]=member_values(model,frame)
        model_file.write_bytes(pickle.dumps(model,protocol=4));value_file.write_bytes(pickle.dumps(values[name],protocol=4))
        model_files.extend([model_file,value_file]);del model;gc.collect()
        progress[model_file.name]=sha(model_file);progress[value_file.name]=sha(value_file)
        write(directory/'member_records.json',progress)
    del refit,panel;gc.collect()
    frame.to_parquet(directory/'evaluation.parquet',index=False)
    (directory/'values.pkl').write_bytes(pickle.dumps(values,protocol=4))
    files=model_files+[directory/'evaluation.parquet',directory/'values.pkl',iteration_file,directory/'iteration_metadata.json',directory/'member_records.json',cache/'manifest.json']
    report=dict(origin=origin,inner_selection_origin=str(selection.date()),members=names,iterations=iterations,
                rows=len(frame),pairs=len(frame[KEY].drop_duplicates()),history_end=origin,
                seconds=round(time.perf_counter()-started,3),
                peak_process_gib=psutil.Process().memory_info().peak_wset/2**30,
                files={p.relative_to(directory).as_posix():sha(p) for p in files})
    write(directory/'complete.json',report)
    print('FOLD COMPLETE',origin,report['seconds'],flush=True)
    return frame,values,report


def evaluate_recipes(frame,values,grid):
    y=frame.actual.to_numpy(float);codes,_=pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))
    base_recipe={'source':'base','post':{'round_int':True}}
    base=fast_metrics(y,recipe_prediction(values,frame,base_recipe),codes)
    results={}
    for i,(key,recipe) in enumerate(grid.items()):
        metrics=fast_metrics(y,recipe_prediction(values,frame,recipe),codes)
        results[key]=dict(metrics=metrics,local_ratio=ratio(metrics,base))
    return base,results


def development(out,protocol):
    if (out/'selection.json').exists():
        if sha(out/'selection.json')!=(out/'selection.sha256').read_text().strip():raise ValueError('Selection changed')
        print('Frozen selection already exists',flush=True);return
    grid=read(out/'candidate_grid.json');sales,stock=load_data();folds={}
    for origin in protocol['development_origins']:
        frame,values,report=fold(out,origin,sales,stock,MEMBERS)
        base,results=evaluate_recipes(frame,values,grid)
        write(out/origin/'development_metrics.json',dict(baseline=base,results=results))
        folds[origin]=dict(baseline=base,results=results)
        del frame,values;gc.collect()
    summary={}
    for key,recipe in grid.items():
        ratios=[v['results'][key]['local_ratio'] for v in folds.values()]
        metric_ratios={k:float(np.mean([v['results'][key]['metrics'][k]/max(v['baseline'][k],1e-12) for v in folds.values()])) for k in NAMES}
        improved=sum(v<1 for v in metric_ratios.values())
        max_smape=max(v['results'][key]['metrics']['smape']/v['baseline']['smape'] for v in folds.values())
        eligible=np.mean(ratios)<=.995 and metric_ratios['smape']<=1.01 and improved>=3 and max_smape<=1.05
        summary[key]=dict(recipe=recipe,mean_ratio=float(np.mean(ratios)),per_origin_ratio=ratios,
                          metric_ratios=metric_ratios,improved_metrics=improved,eligible=bool(eligible))
    eligible=[(k,v) for k,v in summary.items() if v['eligible']]
    baseline_key=next(k for k,v in grid.items() if v=={'source':'base','post':{'round_int':True}})
    selected=min(eligible,key=lambda item:(item[1]['mean_ratio'],item[0]))[0] if eligible else baseline_key
    selection=dict(selected_id=selected,selected_recipe=grid[selected],development_origins=protocol['development_origins'],
                   results=summary,protocol_sha256=sha(out/'protocol.json'),grid_sha256=sha(out/'candidate_grid.json'),
                   previous_periods_inspected=True)
    write(out/'selection.json',selection);(out/'selection.sha256').write_text(sha(out/'selection.json')+'\n')
    print('SELECTED',json.dumps(summary[selected]),flush=True)


def confirmation(out,protocol):
    selection=read(out/'selection.json')
    if sha(out/'selection.json')!=(out/'selection.sha256').read_text().strip():raise ValueError('Selection changed')
    if selection['protocol_sha256']!=sha(out/'protocol.json'):raise ValueError('Selection protocol changed')
    if (out/'decision.json').exists():print('One confirmation already completed');return
    recipe=selection['selected_recipe'];names=sorted({'base',recipe['source'],recipe.get('aux')}-{'tsb','sba',None})
    sales,stock=load_data();origin=protocol['confirmation_origin']
    frame,values,report=fold(out,origin,sales,stock,names)
    y=frame.actual.to_numpy(float);codes,_=pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))
    base=fast_metrics(y,recipe_prediction(values,frame,{'source':'base','post':{'round_int':True}}),codes)
    pred=recipe_prediction(values,frame,recipe);candidate=fast_metrics(y,pred,codes)
    score=ratio(candidate,base);improved=sum(candidate[k]<base[k] for k in NAMES)
    promote=score<=.995 and candidate['smape']<=base['smape']*1.01 and improved>=3
    segment_report={}
    from app.ru_calendar import weekly_calendar
    weekcal=weekly_calendar(frame.week.unique(),origin)
    labels={'holiday_week':frame.week.map(weekcal.ru_holidays)>0,'ordinary_week':frame.week.map(weekcal.ru_holidays)==0,
            'rare':frame.zero_week_share_13>=.6,'regular':frame.zero_week_share_13<.3}
    for name,mask in labels.items():
        if not mask.any():continue
        c,_=pd.factorize(pd.MultiIndex.from_frame(frame.loc[mask,KEY]))
        segment_report[name]=dict(rows=int(mask.sum()),candidate=fast_metrics(y[mask],pred[mask],c),
            baseline=fast_metrics(y[mask],recipe_prediction(values,frame,{'source':'base','post':{'round_int':True}})[mask],c))
    baselines={name:fast_metrics(y,raw,codes) for name,raw in [('zero',np.zeros(len(frame))),('last_week',frame.last_week.to_numpy(float)),('mean4',frame.mean_week_4.to_numpy(float)),('mean8',frame.mean_week_8.to_numpy(float)),('mean13',frame.mean_week_13.to_numpy(float))]}
    decision=dict(promote=bool(promote),selected_recipe=recipe,confirmation_origin=origin,
                  baseline=base,candidate=candidate,local_ratio=score,improved_metrics=improved,
                  baselines=baselines,segments=segment_report,selection_sha256=sha(out/'selection.json'),
                  protocol_sha256=sha(out/'protocol.json'),confirmation_previously_inspected=True,
                  final_action='refit_selected_candidate' if promote else 'retain_previous_production')
    write(out/'decision.json',decision);print('CONFIRMATION',json.dumps(decision),flush=True)


def final_refit(out,protocol):
    decision=read(out/'decision.json')
    if decision['selection_sha256']!=sha(out/'selection.json'):raise ValueError('Changed selection')
    if not decision['promote']:
        print('Candidate not promoted; original production preserved',flush=True);return
    target=out/'final'
    if (target/'config.json').exists():print('Final artifact already exists');return
    sales,stock=load_data();end=pd.Timestamp('2026-07-26')
    panel=add_money(build_panel(sales,stock),sales)
    cutoffs=make_cutoffs(panel)[-16:];cache=out/'final-cache'
    train=pd.concat([feature_frame(panel,c,cache) for c in cutoffs],ignore_index=True)
    assert (train.week+pd.Timedelta(days=6)).max()<=end
    origin=protocol['confirmation_origin'];iterations=read(out/origin/'iterations.json')
    recipe=decision['selected_recipe'];required={recipe['source'],recipe.get('aux')}-{'tsb','sba',None}
    model=WeeklyResearchModel(dict(recipe=recipe,threads=6,estimators=600))
    model.fit(train,train.target.to_numpy(float),n_iterations={k:iterations[k] for k in required})
    config=save_artifact(model,target,end,end,len(train),len(cutoffs),
                validation=dict(scheme='frozen_research_historical_confirmation',metrics=decision['candidate'],
                                confirmation_origin=origin,previously_inspected_period=True),
                provenance=dict(research_protocol_sha256=sha(out/'protocol.json'),selection_sha256=sha(out/'selection.json'),
                                decision_sha256=sha(out/'decision.json'),history_cutoff='2026-07-26'))
    write(out/'final_training.json',dict(rows=len(train),cutoffs=len(cutoffs),iterations=model.selected_iterations,
                                       model_sha256=config['model_artifact_sha256'],preprocessor_sha256=config['preprocessor_sha256']))
    print('FINAL REFIT COMPLETE',config['model_artifact_sha256'],flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage',choices=['development','confirm','finalize'],required=True)
    args=ap.parse_args(argv);out=args.out.resolve();protocol=initialize(out)
    if args.stage=='development':development(out,protocol)
    elif args.stage=='confirm':confirmation(out,protocol)
    else:final_refit(out,protocol)
    verify_protocol(out)


if __name__=='__main__':main()
