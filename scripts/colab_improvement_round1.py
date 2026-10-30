"""Sequential weekly research: cached screening locally, calibration fits in Colab only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.models.research_model import recipe_prediction
from app.models.metric_experts import bayes_quantities

ORIGINS = ['2026-01-18', '2026-03-22']
NAMES = ['smape', 'wape', 'rmse', 'bias', 'rmsle']
WEIGHTS = np.array([.5, .2, .15, .1, .05])
KEY = ['week', 'warehouse_name', 'article']
LAMBDAS = [0., .5, 1., 2., 3., 4., 6., 8., 16.]
DEFAULT = ROOT / 'artifacts/colab-sequential-20261029'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def codes(frame):
    return pd.factorize(pd.MultiIndex.from_frame(frame[['warehouse_name', 'article']]))[0]


def metrics(y, p, code):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid forecast')
    n = np.bincount(code); a = np.bincount(code, weights=y); b = np.bincount(code, weights=p)
    point = np.divide(200*np.abs(y-p), y+p, out=np.zeros(len(y)), where=y+p > 0)
    series = np.bincount(code, weights=point)/n
    return dict(smape=float(series.mean()),
        wape=float(100*np.abs(y-p).sum()/y.sum()) if y.sum() else (0. if p.sum()==0 else 100.),
        rmse=float(np.sqrt(np.mean((y-p)**2))),
        bias=float(np.mean(100*np.abs(b[a>0]-a[a>0])/a[a>0])) if (a>0).any() else 0.,
        rmsle=float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2))))


def ratio(a, b):
    if b == 0:
        if a != 0: raise ValueError('Cannot normalize nonzero error by zero baseline')
        return 1.
    return a/b


def diagnostics(frame, prediction):
    result={};weekly=7*frame.mean_28.to_numpy(float)
    masks={f'h{h}':frame.h.to_numpy(int)==h for h in range(1,10)}
    for name,lo,hi in [('rare',0.,.5),('low',.5,2.),('medium',2.,10.),('regular',10.,50.),('bulk',50.,np.inf)]:
        masks[name]=(weekly>=lo)&(weekly<hi)
    for name,mask in masks.items():
        if not mask.any():continue
        part=frame.loc[mask];y=part.actual.to_numpy(float);p=prediction[mask]
        result[name]=dict(rows=int(mask.sum()),metrics=metrics(y,p,codes(part)),
            squared_error_sum=float(np.sum((y-p)**2)),event_mismatch_rate=float(np.mean((y>0)!=(p>0))))
    return result


def candidates(recipe):
    result = []
    def add(task, label, policy): result.append(dict(id=f'{task:02d}_{label}', task=task, policy=policy))
    def rec(task, label, r): add(task, label, dict(kind='recipe', recipe=r))
    rec(0, 'baseline', recipe)
    for a, b in itertools.product([.1, .25, .4, .5], repeat=2):
        r=copy.deepcopy(recipe);r['segments'][3]['blend_weight']=a;r['segments'][4]['blend_weight']=b
        rec(1, f'weights_{a}_{b}', r)
    for schedule in [[.5,.4,.25],[.25,.4,.5],[.5,.5,.25],[.25,.5,.5]]:
        recipes=[]
        for w in schedule:
            r=copy.deepcopy(recipe)
            for i in [3,4]:r['segments'][i]['blend_weight']=w
            recipes.append(r)
        add(2, 'horizon_'+str(len(result)), dict(kind='horizon', recipes=recipes))
    for gate in [.45,.5,.55,.6,.65,.7]:
        r=copy.deepcopy(recipe);r['segments'][2]['recipe']['post'].update(gate=gate, medium_gate=gate)
        rec(3, f'gate_{gate}', r)
    for schedule in [[.5,.6,.7],[.55,.6,.65],[.65,.6,.55],[.7,.6,.5]]:
        recipes=[]
        for g in schedule:
            r=copy.deepcopy(recipe);r['segments'][2]['recipe']['post'].update(gate=g,medium_gate=g);recipes.append(r)
        add(3,'horizon_'+str(len(result)),dict(kind='horizon',recipes=recipes))
    for scale in [.9,1.,1.05,1.1,1.15]:
        r=copy.deepcopy(recipe);r['post']['regular_scale']=scale
        for s in r['segments']:
            if s['recipe']['source']=='dynamics':s['recipe']['post']['regular_scale']=scale
        rec(4, f'scale_{scale}', r)
    for index in [-1,0,1,2,3,4]:
        for threshold in [None,.35,.5,.65]:
            r=copy.deepcopy(recipe);p=r['post'] if index==-1 else r['segments'][index]['recipe']['post']
            p['round_int']=threshold is not None
            if threshold is not None:p['round_threshold']=threshold
            rec(5,f'round_{index}_{threshold}',r)
    for index in [3,4]:
        for mode in ['none','zeros','exact']:
            r=copy.deepcopy(recipe);s=r['segments'][index]
            s.pop('keep_base_mask',None);s.pop('keep_base_zeros',None)
            if mode=='zeros':s['keep_base_zeros']=True
            if mode=='exact':s['keep_base_mask']=True
            rec(6,f'mask_{index}_{mode}',r)
    for value in LAMBDAS:
        r=copy.deepcopy(recipe);r['segments'][1]['recipe']['source']=f'metric_count_lambda{value:g}'
        rec(7,f'lambda_{value:g}',r)
    for schedule in [[1.,3.,8.],[3.,4.,8.],[8.,4.,3.]]:
        recipes=[]
        for value in schedule:
            r=copy.deepcopy(recipe);r['segments'][1]['recipe']['source']=f'metric_count_lambda{value:g}';recipes.append(r)
        add(7,'horizon_'+str(len(result)),dict(kind='horizon',recipes=recipes))
    # Further existing-component screens are explicit, not new training.
    for weight in [.1,.25,.5,.75]:
        for mask in ['none','zeros']:
            add(13,f'medium_{weight}_{mask}',dict(kind='segment_blend',lower=2.,upper=10.,
                source='metric_regular_recent_mae',weight=weight,mask=mask))
    for index in [0,2,3,4]:
        for value in [1.,3.,8.]:
            r=copy.deepcopy(recipe);s=r['segments'][index]
            s['recipe']=dict(source=f'metric_count_lambda{value:g}',post={'round_int':False});s['blend_weight']=1.
            s.pop('keep_base_mask',None);s.pop('keep_base_zeros',None)
            rec(14,f'count_{index}_{value:g}',r)
    for source in ['mae','quantile','poisson','tweedie']:
        for index in [2,3,4]:
            r=copy.deepcopy(recipe);s=r['segments'][index]
            s['recipe']=dict(source=source,post={'round_int':False});s['blend_weight']=.25
            rec(17,f'cached_{source}_{index}',r)
    for multiplier in [3.,5.,8.]:add(18,f'cap_{multiplier}',dict(kind='cap',multiplier=multiplier))
    for boundary in range(4):
        for width in [.1,.2]:add(28,f'smooth_{boundary}_{width}',dict(kind='smooth',boundary=boundary,width=width))
    for event in ['event31','event63']:add(32,event,dict(kind='event',event=event))
    for w in [.1,.25,.5]:add(32,f'balanced_{w}',dict(kind='balanced_blend',weight=w))
    return result


def load(bundle, origin):
    frame=pd.read_parquet(bundle/'data'/f'{origin}.parquet')
    with np.load(bundle/'data'/f'{origin}.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    values={'dynamics':{k:arrays['dynamics__'+k] for k in ['probability','magnitude','lgbm_raw']}}
    for name in ['mae','quantile','poisson','tweedie','metric_regular_recent_mae','metric_regular_recent_balanced_mae']:
        values[name]={'raw':arrays[name]}
    fit=read(bundle/'data'/f'{origin}.count.json')
    for value in LAMBDAS:
        values[f'metric_count_lambda{value:g}']={'raw':bayes_quantities(arrays['count_probability'],
            fit['representatives'],fit['training_mean'],value)}
    return frame,values,arrays


def predict(policy, frame, values, base, recipe, arrays):
    kind=policy['kind']
    if kind=='recipe':return recipe_prediction(values,frame,policy['recipe'])
    if kind=='horizon':
        p=base.copy();hgroup=np.minimum((frame.h.to_numpy(int)-1)//3,2)
        for i,r in enumerate(policy['recipes']):
            q=recipe_prediction(values,frame,r);p[hgroup==i]=q[hgroup==i]
        return p
    if kind=='cap':
        cap=np.maximum.reduce([np.full(len(frame),3.),frame.last_week.to_numpy(float)*policy['multiplier'],
            7*frame.mean_28.to_numpy(float)*policy['multiplier'],
            frame.mean_week_13.to_numpy(float)+policy['multiplier']*frame.std_week_13.to_numpy(float)])
        return np.minimum(base,np.nan_to_num(cap,nan=np.inf))
    if kind=='smooth':
        i=policy['boundary'];left=recipe['segments'][i];right=recipe['segments'][i+1]
        boundary=float(left['max_weekly_mean']);width=boundary*policy['width']
        r=7*frame.mean_28.to_numpy(float);mask=np.abs(r-boundary)<width;p=base.copy()
        alternates=[]
        for s in [left,right]:
            rcp=copy.deepcopy(recipe);s=copy.deepcopy(s);s['min_weekly_mean']=0.;s['max_weekly_mean']=None
            rcp['segments']=[s];alternates.append(recipe_prediction(values,frame,rcp))
        w=np.clip((r-(boundary-width))/(2*width),0,1)
        p[mask]=((1-w)*alternates[0]+w*alternates[1])[mask]
        return p
    if kind=='event':
        v=dict(values);v['dynamics']={**values['dynamics'],'probability':arrays[policy['event']]}
        return recipe_prediction(v,frame,recipe)
    if kind=='balanced_blend':
        p=base.copy();mask=(frame.mean_28.to_numpy(float)*7>=10)&(frame.mean_28.to_numpy(float)*7<50)
        raw=values['metric_regular_recent_balanced_mae']['raw'];w=policy['weight']
        p[mask]=((1-w)*base+w*raw)[mask];return p
    if kind=='segment_blend':
        p=base.copy();weekly=7*frame.mean_28.to_numpy(float)
        mask=(weekly>=policy['lower'])&(weekly<policy['upper']);w=policy['weight']
        q=(1-w)*base+w*values[policy['source']]['raw']
        if policy['mask']=='zeros':q=np.where(base>0,q,0.)
        p[mask]=q[mask];return p
    raise ValueError('Unknown policy: '+kind)


def verify(bundle):
    manifest=read(bundle/'MANIFEST.json')
    for name,digest in manifest['files'].items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path)!=digest:raise ValueError('Bundle changed: '+name)
    return read(bundle/'protocol.json')


def screen(bundle):
    protocol=verify(bundle);out=bundle/'screening';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Screen already exists; preserve frozen results')
    records=[dict(c,metrics={}) for c in protocol['candidates']];baselines={};naive={};best_arrays={}
    for origin in ORIGINS:
        frame,values,arrays=load(bundle,origin);code=codes(frame);y=frame.actual.to_numpy(float)
        base=recipe_prediction(values,frame,protocol['recipe'])
        if not np.allclose(base,arrays['baseline'],rtol=0,atol=1e-10):raise ValueError('Baseline replay differs')
        baselines[origin]=metrics(y,base,code);naive[origin]=metrics(y,frame.last_week.to_numpy(float),code)
        for r in records:
            p=predict(r['policy'],frame,values,base,protocol['recipe'],arrays)
            r['metrics'][origin]=metrics(y,p,code)
        e=(base-y)**2;worst=frame[KEY+['actual','h','mean_28']].copy();worst['prediction']=base;worst['squared_error']=e
        worst.sort_values('squared_error',ascending=False).head(100).to_csv(out/f'{origin}.largest_errors.csv',index=False)
    for r in records:
        per={o:float(sum(w*ratio(r['metrics'][o][k],baselines[o][k]) for w,k in zip(WEIGHTS,NAMES))) for o in ORIGINS}
        r.update(weighted_ratio=float(np.mean(list(per.values()))),per_origin_weighted_ratio=per,
                 worst_origin_ratio=max(per.values()),naive_normalized_score=float(np.mean([
                 sum(w*min(ratio(r['metrics'][o][k],naive[o][k]),1.) for w,k in zip(WEIGHTS,NAMES)) for o in ORIGINS])))
    # Fixed-pool rank; exact copies do not inflate ranks.
    unique={}
    for r in records:
        sig=tuple(round(r['metrics'][o][k],8) for o in ORIGINS for k in NAMES)
        unique.setdefault(sig,[]).append(r)
    groups=list(unique.values());score=np.zeros(len(groups))
    for o in ORIGINS:
        for w,k in zip(WEIGHTS,NAMES):
            rank=pd.Series([g[0]['metrics'][o][k] for g in groups]).round(8).rank(method='average').to_numpy()
            score+=w*(rank-1)/max(len(groups)-1,1)/len(ORIGINS)
    for s,g in zip(score,groups):
        for r in g:r['local_rank_score']=float(s)
    ordered=sorted(records,key=lambda r:(r['weighted_ratio'],r['id']))
    selected={}
    for task in sorted({r['task'] for r in records}):
        best=min([r for r in records if r['task']==task],key=lambda r:r['weighted_ratio']);selected[str(task)]=best
    # Keep a compact set of full forecasts for cross-stage comparisons.
    retained={ordered[0]['id'],protocol['candidates'][0]['id'],*[r['id'] for r in selected.values()]}
    for origin in ORIGINS:
        frame,values,arrays=load(bundle,origin);base=arrays['baseline'];preds={}
        details={}
        for r in records:
            if r['id'] in retained:
                preds[r['id']]=predict(r['policy'],frame,values,base,protocol['recipe'],arrays)
                details[r['id']]=diagnostics(frame,preds[r['id']])
        np.savez_compressed(out/f'{origin}.shortlist.npz',**preds)
        write(out/f'{origin}.diagnostics.json',details)
    result=dict(protocol_sha256=sha(bundle/'protocol.json'),training_executed=False,independent_confirmation=False,
        candidates=len(records),unique_outcomes=len(groups),baseline_metrics=baselines,best_by_task=selected,
        ranking_by_weighted_ratio=ordered,selection_is_retrospective=True)
    write(out/'results.json',result)
    pd.DataFrame([dict(id=r['id'],task=r['task'],weighted_ratio=r['weighted_ratio'],
        worst_origin_ratio=r['worst_origin_ratio'],local_rank_score=r['local_rank_score'],
        naive_normalized_score=r['naive_normalized_score']) for r in ordered]).to_csv(out/'ranking.csv',index=False)
    print(json.dumps(dict(candidates=len(records),best=ordered[0]['id'],weighted_ratio=ordered[0]['weighted_ratio']),indent=2))


def apply_calibrator(p, model):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    if model['kind']=='platt':
        z=np.clip(model['coef']*np.log(p/(1-p))+model['intercept'],-35,35)
        return 1/(1+np.exp(-z))
    if model['kind']=='isotonic':return np.interp(p,model['x'],model['y'])
    raise ValueError('Unknown calibrator')


def fit_calibrator(p, y, kind):
    # This is the FIRST fitting boundary. Only --stage calibrate in Colab calls it.
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8);y=np.asarray(y,int)
    if len(np.unique(y))!=2:raise ValueError('Calibration data requires both classes')
    if kind=='platt':
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(np.log(p/(1-p)).reshape(-1,1),y)
        return dict(kind=kind,coef=float(model.coef_[0,0]),intercept=float(model.intercept_[0]))
    from sklearn.isotonic import IsotonicRegression
    model=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p,y)
    return dict(kind=kind,x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())


def calibrate(bundle):
    started=time.perf_counter()
    protocol=verify(bundle);out=bundle/'calibration_results';out.mkdir(exist_ok=True)
    all_records=[]
    for kind,scope in itertools.product(['platt','isotonic'],['global','horizon_groups']):
        identity=kind+'_'+scope;dest=out/identity;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            previous=read(dest/'complete.json')
            if previous.get('protocol_sha256')!=sha(bundle/'protocol.json'):raise ValueError('Resume protocol differs')
            for name,digest in previous['output_hashes'].items():
                if sha(dest/name)!=digest:raise ValueError('Partial calibration result changed: '+name)
            all_records.append(previous);continue
        record=dict(id=identity,task=8,metrics={},baselines={},calibration_quality={},training_sources={})
        for origin in ORIGINS:
            frame,values,arrays=load(bundle,origin);y=(frame.actual.to_numpy(float)>0).astype(int);hgroup=(frame.h.to_numpy(int)-1)//3
            past=[];sources=[]
            for p in sorted((bundle/'calibration_data').glob('*.parquet')):
                if p.stem >= origin:continue
                f=pd.read_parquet(p)
                if (pd.to_datetime(f.week)+pd.Timedelta(days=6)).max()>pd.Timestamp(origin):raise ValueError('Calibration labels exceed origin')
                past.append(f);sources.append(p.stem)
            if not past:raise ValueError('No earlier calibration outcomes')
            train=pd.concat(past,ignore_index=True);yp=(train.actual.to_numpy(float)>0).astype(int)
            group=(train.h.to_numpy(int)-1)//3;model={'global':fit_calibrator(train.probability,yp,kind)}
            if scope=='horizon_groups':
                for g in range(3):
                    mask=group==g;model[str(g)]=fit_calibrator(train.probability.to_numpy()[mask],yp[mask],kind)
            raw=values['dynamics']['probability'];calibrated=apply_calibrator(raw,model['global'])
            if scope=='horizon_groups':
                for g in range(3):calibrated[hgroup==g]=apply_calibrator(raw[hgroup==g],model[str(g)])
            v=dict(values);v['dynamics']={**values['dynamics'],'probability':calibrated}
            prediction=recipe_prediction(v,frame,protocol['recipe']);code=codes(frame)
            record['metrics'][origin]=metrics(frame.actual,prediction,code)
            record['baselines'][origin]=metrics(frame.actual,arrays['baseline'],code)
            from sklearn.metrics import log_loss
            record['calibration_quality'][origin]=dict(brier_before=float(np.mean((raw-y)**2)),
                brier_after=float(np.mean((calibrated-y)**2)),logloss_before=float(log_loss(y,raw,labels=[0,1])),
                logloss_after=float(log_loss(y,np.clip(calibrated,1e-8,1-1e-8),labels=[0,1])))
            record['training_sources'][origin]=sources
            write(dest/f'{origin}.model.json',model)
            np.save(dest/f'{origin}.prediction.npy',prediction)
            write(dest/f'{origin}.diagnostics.json',diagnostics(frame,prediction))
        record['weighted_ratio']=float(np.mean([sum(w*ratio(record['metrics'][o][k],record['baselines'][o][k])
            for w,k in zip(WEIGHTS,NAMES)) for o in ORIGINS]))
        record['independent_confirmation']=False
        record['protocol_sha256']=sha(bundle/'protocol.json')
        record['output_hashes']={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}
        write(dest/'complete.json',record);all_records.append(record)
        print(identity,record['weighted_ratio'],flush=True)
    write(out/'summary.json',dict(protocol_sha256=sha(bundle/'protocol.json'),task=8,records=all_records,
        modifies='dynamics probability only; count probabilities not calibrated',production_changed=False))
    import importlib.metadata as metadata
    write(out/'run_manifest.json',dict(protocol_sha256=sha(bundle/'protocol.json'),
        runner_sha256=sha(Path(__file__)),input_manifest_sha256=sha(bundle/'MANIFEST.json'),
        python=platform.python_version(),platform=platform.platform(),
        versions={name:metadata.version(name) for name in ['numpy','pandas','scikit-learn','lightgbm','pyarrow']},
        seconds=time.perf_counter()-started,training_executed=True,production_changed=False,
        independent_confirmation=False,seed=42,origins=ORIGINS))
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    write(out/'OUTPUT_MANIFEST.json',dict(protocol_sha256=sha(bundle/'protocol.json'),files=files))
    archive=bundle.parent/'round08_results.zip'
    if archive.exists():raise FileExistsError('Result archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print('RETURN THIS ARCHIVE:',archive,flush=True)


def build(out):
    if out.exists():raise FileExistsError('Preserve prior run; choose a new directory')
    out.mkdir(parents=True);bundle=out/'colab_round08';(bundle/'data').mkdir(parents=True);(bundle/'calibration_data').mkdir()
    source=ROOT/'artifacts/improvements-20261028-r1';recipe=read(ROOT/'model/config.json')['model_params']['recipe']
    expected=read(ROOT/'artifacts/score-review-20261029/SELECTED_CONFIGURATION.json')
    if recipe!=expected['recipe']:raise ValueError('Current recipe differs from the selected baseline')
    provenance={}
    def checked(path,digest=None):
        digest_now=sha(path)
        if digest is not None and digest_now!=digest:raise ValueError('Input changed: '+str(path))
        provenance[path.relative_to(ROOT).as_posix()]=digest_now
        return path
    for origin in ['2025-10-19',*ORIGINS]:
        folder=source/origin;complete=read(folder/'complete.json')
        frame=pd.read_parquet(checked(folder/'evaluation.parquet',complete['files']['evaluation.parquet']))
        dyn=pickle.loads(checked(folder/'dynamics-values.pkl',complete['files']['dynamics-values.pkl']).read_bytes())
        if frame.groupby(['warehouse_name','article'],observed=True).size().ne(9).any():raise ValueError('Incomplete horizon')
        c=frame[KEY+['h','actual']].copy();c['probability']=dyn['probability']
        c.to_parquet(bundle/'calibration_data'/f'{origin}.parquet',index=False)
        if origin not in ORIGINS:continue
        aligned=read(ROOT/'artifacts/regular-quantity-20261029/origin_alignment.json')[origin]
        key_sha=hashlib.sha256(frame[KEY].astype(str).to_csv(index=False,lineterminator='\n').encode()).hexdigest()
        if key_sha!=aligned['ordered_key_sha256']:raise ValueError('Key order differs')
        arrays={f'dynamics__{k}':v for k,v in dyn.items()}
        values={'dynamics':dyn}
        for name in ['mae','quantile','poisson','tweedie']:
            obj=pickle.loads(checked(folder/f'{name}-values.pkl',complete['files'][f'{name}-values.pkl']).read_bytes())
            arrays[name]=obj['raw'];values[name]=obj
        reg=ROOT/'artifacts/regular-quantity-20261029';reg_manifest=read(reg/'output_manifest.json')['files']
        for name in ['recent_mae','recent_balanced_mae']:
            path=reg/origin/f'{name}.raw.npy';raw=np.load(checked(path,reg_manifest[f'{origin}/{path.name}']),allow_pickle=False)
            arrays['metric_regular_'+name]=raw;values['metric_regular_'+name]={'raw':raw}
        count=ROOT/'artifacts/count-distribution-20261029';manifest=read(count/'output_manifest.json')['files']
        fit=read(checked(count/origin/'fit.json',manifest[f'{origin}/fit.json']))
        arrays['count_probability']=np.load(checked(count/origin/'probability.npy',manifest[f'{origin}/probability.npy']),allow_pickle=False)
        for value in LAMBDAS:
            values[f'metric_count_lambda{value:g}']={'raw':bayes_quantities(arrays['count_probability'],fit['representatives'],fit['training_mean'],value)}
        for event in ['event31','event63']:
            path=ROOT/'artifacts/events-20261029'/origin/f'{event}.probability.npy'
            event_manifest=read(ROOT/'artifacts/events-20261029/output_manifest.json')
            arrays[event]=np.load(checked(path,event_manifest[f'{origin}/{path.name}']),allow_pickle=False)
        arrays['baseline']=recipe_prediction(values,frame,recipe)
        measured=metrics(frame.actual,arrays['baseline'],codes(frame))
        for name in NAMES:
            if not np.isclose(measured[name],expected['metrics_by_origin'][origin][name],rtol=1e-10,atol=1e-10):raise ValueError('Baseline metric parity failed')
        frame.to_parquet(bundle/'data'/f'{origin}.parquet',index=False)
        np.savez_compressed(bundle/'data'/f'{origin}.npz',**arrays)
        write(bundle/'data'/f'{origin}.count.json',dict(representatives=fit['representatives'],training_mean=fit['training_mean']))
    shutil.copytree(ROOT/'app',bundle/'app',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),bundle/'run.py')
    protocol=dict(version=1,origins=ORIGINS,baseline_model_sha256=sha(ROOT/'model/model.pkl'),
        recipe=recipe,candidates=candidates(recipe),weights=dict(zip(NAMES,WEIGHTS.tolist())),
        primary='mean weighted error ratios to fixed baseline, weights 50/20/15/10/5; not official score',
        calibration_methods=['platt','isotonic'],calibration_scopes=['global','horizon_groups'],
        calibration_training='only completed earlier origin predictions; no evaluation outcome used for its own calibrator',
        independent_confirmation=False,source_files=provenance,production_changed=False,
        training_boundary='--stage calibrate must be run by user in Colab, not locally')
    write(bundle/'protocol.json',protocol)
    manifest={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    write(bundle/'MANIFEST.json',dict(files=manifest))
    screen(bundle)
    # The training bundle has immutable inputs, plus already executed cached results.
    archive=out/'colab_round08.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in bundle.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print('COLAB INPUT:',archive,'bytes',archive.stat().st_size,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['build','screen','verify','calibrate'],required=True)
    parser.add_argument('--path',type=Path,default=DEFAULT)
    args=parser.parse_args();path=args.path.resolve()
    if args.stage=='build':build(path)
    elif args.stage=='screen':screen(path)
    elif args.stage=='verify':print('VERIFIED',verify(path)['baseline_model_sha256'])
    else:calibrate(path)


if __name__=='__main__':main()
