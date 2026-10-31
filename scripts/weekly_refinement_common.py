"""Frozen no-fit refinement of the deployed all40 weekly ensemble."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
ORIGINS=['2026-01-18','2026-03-22'];KEY=['warehouse_name','article'];WK=['week',*KEY]
METRICS=['smape','wape','rmse','bias','rmsle'];WEIGHTS=[.5,.2,.15,.1,.05]
RUN=ROOT/'artifacts/all40-0a385edab6b1'
MODEL_SHA='201c6b747ee351feff7f2017c2c45820862ad53fa0ab9cd9332425a4945d59fd'
SCREEN_CONFIG=dict(global_weights=[0.,.25,.5,.6,.7,.75,.8,.9,1.],
    thresholds=[None,.35,.5,.65],lambdas=[.5,1.,2.,3.,4.,6.,8.,12.],
    cell_weights=[.5,.65,.85,1.],cell_lambdas=[1.,2.,6.,12.],joint_top_k=3)
PARTS=['count_fine','core_event','core_positive','core_log','core_mae','core_count','core_recent']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def cells(frame):
    r=7*frame.mean_28.to_numpy(float);h=frame.h.to_numpy(int)
    if not np.isfinite(r).all() or (r<0).any() or ((h<1)|(h>9)).any():raise ValueError('Invalid past-demand/horizon cells')
    return np.searchsorted([.5,2.,10.,50.],r,side='right'),(h-1)//3

def baseline_policy():
    return dict(weights=np.full((5,3),.75).tolist(),lambdas=np.full((5,3),3.).tolist(),threshold=.5)

def quantize(p,threshold):
    p=np.maximum(np.asarray(p,float),0.)
    if threshold is None:return p
    return np.rint(p) if threshold==.5 else np.floor(p+1-threshold)

def decode(probability,schema,multiplier=3.):
    p=np.asarray(probability,float);reps=np.asarray(schema['representatives'],float)
    if p.ndim==1:p=p.reshape(-1,len(reps),order='F')
    if p.ndim!=2 or p.shape[1]!=len(reps) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid class probabilities')
    if not np.allclose(p.sum(axis=1),1.,rtol=1e-6,atol=1e-6):raise ValueError('Class probabilities do not sum to one')
    if not np.isfinite(multiplier) or multiplier<0:raise ValueError('Invalid decoder lambda')
    actions=np.unique(np.r_[0.,.5,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.,reps])
    error=np.abs(reps[:,None]-actions[None,:]);den=reps[:,None]+actions[None,:]
    relative=np.divide(200*error,den,out=np.zeros_like(error),where=den>0)
    risk=.7/60*relative+multiplier*.3/50*100*error/max(float(schema['training_mean']),1e-9)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):return actions[np.argmin(p@risk,axis=1)]

def probabilities(folder,frame,inner=False):
    import lightgbm as lgb
    prefix='inner' if inner else 'refit';schema=read(Path(folder)/f'{prefix}_schema.json')
    x=frame[schema['columns']].copy()
    for c,values in schema['categories'].items():x[c]=pd.Categorical(x[c].astype(str),categories=values)
    model=lgb.Booster(model_file=str(Path(folder)/f'{prefix}_model.txt'))
    return model.predict(x,num_threads=6),schema

def component_values(origin,frame,inner=False,final=False):
    from app.models.all40_model import NativePart
    from app.models.research_model import recipe_prediction
    prefix='inner' if inner else 'refit';values={}
    for name in PARTS[1:]:
        folder=RUN/'models'/name/('FINAL' if final else origin)
        part=NativePart((folder/f'{prefix}_model.txt').read_text(encoding='utf-8'),read(folder/f'{prefix}_schema.json'))
        values[name]=part.predict(frame)
    recipe=read(ROOT/'model/config.json')['model_params']['recipe']
    members={'dynamics':dict(probability=values['core_event'],magnitude=values['core_positive'],lgbm_raw=values['core_log']),
        'mae':dict(raw=values['core_mae']),'metric_count_lambda3':dict(raw=values['core_count']),
        'metric_regular_recent_mae':dict(raw=values['core_recent'])}
    return recipe_prediction(members,frame,recipe)

def apply_policy(context,policy,probability=None,schema=None):
    weights=np.asarray(policy['weights'],float);lambdas=np.asarray(policy['lambdas'],float)
    if weights.shape!=(5,3) or lambdas.shape!=(5,3) or ((weights<0)|(weights>1)).any():raise ValueError('Invalid mixing policy')
    r,h=context['cells'];scope=(r==1)|(r==2);left=context['legacy'].copy()
    probability=context['probability'] if probability is None else probability
    schema=context['schema'] if schema is None else schema
    cache=context.get('decoded',{}) if probability is context['probability'] and schema is context['schema'] else {}
    per_row=lambdas[r,h]
    for value in np.unique(per_row[scope]):
        value=float(value)
        if value not in cache:cache[value]=decode(probability,schema,value)
        mask=scope&(per_row==value);left[mask]=cache[value][mask]
    w=weights[r,h]
    return quantize(w*left+(1-w)*context['refreshed'],policy['threshold'])

class Contexts:
    def __init__(self):
        if sha(ROOT/'model/model.pkl')!=MODEL_SHA:raise ValueError('Current weekly baseline changed')
        self.selection=read(RUN/'SELECTION.json');self.champion=self.selection['best_mean']
        self.bundle=ROOT/'artifacts/colab-sequential-20261029-final/colab_round08'
        self.common=load_module('weekly_refine_metrics',self.bundle/'run.py');self.common.verify(self.bundle)
        paths=[Path(__file__),ROOT/'model/model.pkl',ROOT/'model/config.json',ROOT/'model/preprocessor.pkl',
            ROOT/'forecast.csv',ROOT/'forecast.manifest.json',RUN/'SELECTION.json',RUN/'protocol.json',
            self.bundle/'MANIFEST.json',ROOT/'data/sales_dataset_hakaton_train.csv',ROOT/'data/stock_dataset_hakaton_train.csv']
        paths.extend((ROOT/'app').rglob('*.py'))
        from scripts.research_events import inventory
        self.schedules={};fingerprints={}
        for origin in [*ORIGINS,'2026-05-24']:
            schedule,found=inventory(origin);self.schedules[origin]=schedule;fingerprints.update(found)
        # All cached frames are pinned; they were previously checked against as-of source features.
        paths.extend((RUN/'features').glob('*.parquet'))
        for name in PARTS:
            for origin in [*ORIGINS,'FINAL']:
                paths.extend(p for p in (RUN/'models'/name/origin).glob('*') if p.is_file())
        for origin in ORIGINS:
            paths.extend(RUN/'predictions'/f'{name}_{origin}.npy' for name in
                ['native_count_fine_w1.0','refreshed_entire_recipe',self.champion['id']])
        paths.append(ROOT/'artifacts/improvements-20261028-r1/final-cache/manifest.json')
        self.final_manifest=read(paths[-1])
        for name,h in self.final_manifest.items():
            p=ROOT/'artifacts/improvements-20261028-r1/final-cache'/name
            if sha(p)!=h:raise ValueError('Final source cache changed')
            fingerprints[str(p.relative_to(ROOT))]=h
        fingerprints.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
        self.protocol=dict(version=1,files=fingerprints,screen_config=SCREEN_CONFIG,
            reference_model_sha256=MODEL_SHA,reference_id=self.champion['id'],independent_confirmation=False,
            objective='weighted ratios to current weekly champion, weights 50/20/15/10/5')
        identity=hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()[:12]
        self.out=ROOT/'artifacts'/('weekly-refinement-screen-'+identity);self.out.mkdir(parents=True,exist_ok=True)
        if (self.out/'protocol.json').exists():
            if read(self.out/'protocol.json')!=self.protocol:raise ValueError('Frozen screening protocol changed')
        else:write(self.out/'protocol.json',self.protocol)
        self.ph=sha(self.out/'protocol.json');self.contexts={};self.records=[]
        for origin in ORIGINS:self.contexts[origin]=self.load_context(origin)

    def frame(self,origin,name,final=False):
        path=RUN/'features'/f'{"final" if final else origin}_{name}.parquet'
        if not path.exists():raise ValueError('Verified all40 feature cache is missing: '+str(path))
        f=pd.read_parquet(path)
        if f.feature_origin.nunique()!=1 or pd.Timestamp(f.feature_origin.iloc[0])!=pd.Timestamp(name[:10]):
            raise ValueError('Cached as-of feature origin differs')
        return f

    def load_context(self,origin):
        frame=self.frame(origin,self.schedules[origin]['test'])
        old_frame,_,arrays=self.common.load(self.bundle,origin)
        if not frame[WK].astype(str).equals(old_frame[WK].astype(str)):raise ValueError('Benchmark keys differ')
        probability,schema=probabilities(RUN/'models/count_fine'/origin,frame)
        base=np.load(RUN/'predictions'/f'{self.champion["id"]}_{origin}.npy',allow_pickle=False)
        refreshed=np.load(RUN/'predictions'/f'refreshed_entire_recipe_{origin}.npy',allow_pickle=False)
        y=frame.actual.to_numpy(float);code=self.common.codes(frame)
        context=dict(frame=frame,y=y,codes=code,base=base,legacy=arrays['baseline'],refreshed=refreshed,
            probability=probability,schema=schema,cells=cells(frame),decoded={})
        context['base_metrics']=self.common.metrics(y,base,code)
        for k in METRICS:
            if not np.isclose(context['base_metrics'][k],self.champion['metrics'][origin][k],rtol=1e-11):raise ValueError('Current benchmark metrics differ')
        replay=apply_policy(context,baseline_policy())
        if not np.array_equal(replay,base):raise ValueError('Deployed weekly policy replay differs')
        return context

    def measure(self,predictions):
        metrics={};ratios={}
        for origin,p in predictions.items():
            c=self.contexts[origin];metrics[origin]=self.common.metrics(c['y'],p,c['codes'])
            ratios[origin]=float(sum(w*self.common.ratio(metrics[origin][k],c['base_metrics'][k]) for w,k in zip(WEIGHTS,METRICS)))
        return dict(metrics=metrics,per_origin_ratio=ratios,weighted_ratio=float(np.mean(list(ratios.values()))),
            worst_window_ratio=max(ratios.values()))

    def record_policy(self,policy,stage):
        name='policy_'+hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()[:14]
        if any(r['id']==name for r in self.records):return
        p={o:apply_policy(c,policy) for o,c in self.contexts.items()}
        self.records.append(dict(id=name,policy=policy,stage=stage,**self.measure(p),independent_confirmation=False))

    def screen(self):
        result_path=self.out/'SCREEN_RESULTS.json'
        if result_path.exists():
            previous=read(result_path)
            if previous['protocol_sha256']!=self.ph:raise ValueError('Screening result protocol differs')
            self.records=previous['records'];return previous
        base=baseline_policy();self.record_policy(base,'baseline')
        for weight in SCREEN_CONFIG['global_weights']:
            for threshold in SCREEN_CONFIG['thresholds']:
                p=copy.deepcopy(base);p['weights']=np.full((5,3),weight).tolist();p['threshold']=threshold
                self.record_policy(p,'weights')
        # One-coordinate probes, followed by explicit joint testing. No claim that gains add up.
        for kind,groups in [('segment',[(s,None) for s in range(5)]),('horizon',[(None,h) for h in range(3)]),
                ('cell',[(s,h) for s in range(5) for h in range(3)])]:
            for s,h in groups:
                for value in SCREEN_CONFIG['cell_weights']:
                    p=copy.deepcopy(base);a=np.asarray(p['weights'])
                    if s is None:a[:,h]=value
                    elif h is None:a[s,:]=value
                    else:a[s,h]=value
                    p['weights']=a.tolist();self.record_policy(p,'weights_'+kind)
        for value in SCREEN_CONFIG['lambdas']:
            for threshold in SCREEN_CONFIG['thresholds']:
                p=copy.deepcopy(base);p['lambdas']=np.full((5,3),value).tolist();p['threshold']=threshold
                self.record_policy(p,'decoder')
        for s,h in [(1,None),(2,None),(None,0),(None,1),(None,2)]+[(s,h) for s in [1,2] for h in range(3)]:
            for value in SCREEN_CONFIG['cell_lambdas']:
                p=copy.deepcopy(base);a=np.asarray(p['lambdas'])
                if s is None:a[1:3,h]=value
                elif h is None:a[s,:]=value
                else:a[s,h]=value
                p['lambdas']=a.tolist();self.record_policy(p,'decoder_cell')
        def distinct_candidates(prefix):
            chosen=[];seen=set()
            for r in sorted([r for r in self.records if r['stage'].startswith(prefix)],key=lambda r:r['weighted_ratio']):
                signature=tuple(round(r['metrics'][o][k],8) for o in ORIGINS for k in METRICS)
                if signature not in seen:chosen.append(r);seen.add(signature)
                if len(chosen)==SCREEN_CONFIG['joint_top_k']:break
            return chosen
        weight_candidates=distinct_candidates('weights');decoder_candidates=distinct_candidates('decoder')
        for w in weight_candidates:
            for d in decoder_candidates:
                for threshold in SCREEN_CONFIG['thresholds']:
                    p=dict(weights=w['policy']['weights'],lambdas=d['policy']['lambdas'],threshold=threshold)
                    self.record_policy(p,'joint')
        eligible=[r for r in self.records if r['worst_window_ratio']<=1.01]
        best=min(eligible,key=lambda r:(r['weighted_ratio'],r['worst_window_ratio'],r['id']))
        result=dict(protocol_sha256=self.ph,best=best,records=sorted(self.records,key=lambda r:r['weighted_ratio']),
            candidate_count=len(self.records),independent_confirmation=False,training_executed=False,
            note='Adaptive reuse of January/March. Same frozen models, weekly facts and keys. Improvement is not an official score.')
        write(result_path,result)
        keep=result['records'][:10];np.savez_compressed(self.out/'shortlist_predictions.npz',
            **{r['id']+'_'+o:apply_policy(c,r['policy']) for r in keep for o,c in self.contexts.items()})
        return result
