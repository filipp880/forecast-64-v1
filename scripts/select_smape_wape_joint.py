"""One frozen pre-confirmation selection of finite router/count/regular composites.

January/March are the fair common development origins. October is an explicit
available-component guard. May is never loaded by this selector.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from app.features import KEY
from app.models.research_model import recipe_prediction
from scripts.research_smape_wape import (load_origin, sufficient_stats,
    metric_denominators, metrics_from_stats)
from scripts.research_weekly import fast_metrics, freeze_record, frozen_record, read, sha

ORIGINS=['2026-01-18','2026-03-22']
OCTOBER='2025-10-19'
SOURCE=ROOT/'artifacts/improvements-20261028-r1'
ROUTER=ROOT/'artifacts/smape-wape-20261029'
COUNT=ROOT/'artifacts/count-distribution-20261029'
DECODER=ROOT/'artifacts/count-distribution-decoder-20261029'
REGULAR=ROOT/'artifacts/regular-quantity-20261029'
NAMES=['smape','wape','rmse','bias','rmsle']
BINS=[(0.,.5),(.5,2.),(2.,10.),(10.,50.),(50.,None)]
COUNT_SCOPES={'low':(.5,2.),'low_medium':(.5,10.)}
REGULAR_SCOPES={'medium':(2.,10.),'regular':(10.,50.),'all_regular':(2.,None)}
LAMBDAS=[0.,1.,2.,3.,4.,8.,16.]
VARIANTS=['recent_mae','recent_balanced_mae']
BLENDS=[.5,1.]


def count_options():
    return {'none':None, **{f'{scope}_lambda{value:g}':dict(scope=scope,lambda_value=value)
                            for scope in COUNT_SCOPES for value in LAMBDAS}}


def regular_options():
    return {'none':None, **{f'{variant}_{scope}_{blend}':dict(variant=variant,scope=scope,blend=blend)
                           for variant in VARIANTS for scope in REGULAR_SCOPES for blend in BLENDS}}


def in_scope(weekly,lower,upper):
    return (weekly>=lower)&((weekly<upper) if upper is not None else True)


def covers(scope,bin_bounds):
    lower,upper=scope; blo,bhi=bin_bounds
    return blo>=lower and (upper is None or (bhi is not None and bhi<=upper))


def combine_predictions(router,count_prediction,regular_prediction,weekly,count_spec,regular_spec):
    result=np.asarray(router,float).copy()
    if count_spec is not None:
        mask=in_scope(weekly,*COUNT_SCOPES[count_spec['scope']])
        result[mask]=count_prediction[mask]
    if regular_spec is not None:
        mask=in_scope(weekly,*REGULAR_SCOPES[regular_spec['scope']])
        result[mask]=regular_prediction[mask]
    return result


def compile_recipe(champion,router_recipe,count_spec,regular_spec):
    """Compile the overwrite order into disjoint intervals based on original base."""
    recipe={k:copy.deepcopy(v) for k,v in champion.items() if k!='segments'}
    segments=[]
    for lower,upper in BINS:
        selected=None
        for segment in router_recipe.get('segments',[]):
            if covers((segment['min_weekly_mean'],segment['max_weekly_mean']),(lower,upper)):
                selected={k:copy.deepcopy(v) for k,v in segment.items()
                          if k not in ('min_weekly_mean','max_weekly_mean')}
                break
        if count_spec is not None and covers(COUNT_SCOPES[count_spec['scope']],(lower,upper)):
            selected=dict(recipe=dict(source=f"metric_count_lambda{count_spec['lambda_value']:g}",
                                      post={'round_int':False}),
                          blend_weight=1.,keep_base_mask=False)
        if regular_spec is not None and covers(REGULAR_SCOPES[regular_spec['scope']],(lower,upper)):
            selected=dict(recipe=dict(source='metric_regular_'+regular_spec['variant'],
                                      post={'round_int':False}),
                          blend_weight=regular_spec['blend'],keep_base_zeros=True)
        if selected is not None:
            segments.append(dict(min_weekly_mean=lower,max_weekly_mean=upper,**selected))
    # Adjacent intervals with identical actions can be merged without a policy change.
    merged=[]
    for segment in segments:
        action={k:v for k,v in segment.items() if k not in ('min_weekly_mean','max_weekly_mean')}
        previous=merged[-1] if merged else None
        if previous is not None and previous['max_weekly_mean']==segment['min_weekly_mean'] and {
            k:v for k,v in previous.items() if k not in ('min_weekly_mean','max_weekly_mean')}==action:
            previous['max_weekly_mean']=segment['max_weekly_mean']
        else:
            merged.append(segment)
    if merged:
        recipe['segments']=merged
    return recipe


def metric_assessment(metrics,champions):
    ratios={o:{k:float(metrics[o][k]/champions[o][k]) if champions[o][k] else
               (1. if metrics[o][k]==0 else float('inf')) for k in NAMES} for o in ORIGINS}
    means={k:float(np.mean([v[k] for v in ratios.values()])) for k in NAMES}
    primary=means['smape']<1 and means['wape']<1 and all(
        r[k]<=1.03 for r in ratios.values() for k in ['smape','wape'])
    balanced=primary and all(means[k]<=1.03 for k in ['rmse','bias','rmsle']) and all(
        r[k]<=1.08 for r in ratios.values() for k in ['rmse','bias','rmsle'])
    return dict(primary_eligible=bool(primary),balanced=bool(balanced),mean_ratios=means,
                per_origin_ratios=ratios,primary_score=.7*means['smape']+.3*means['wape'])


def recipe_complexity(recipe):
    members=set()
    def visit(node):
        for key in ('source','aux'):
            if node.get(key) not in (None,'tsb','sba'):
                members.add(node[key])
        for segment in node.get('segments',[]):
            if 'recipe' in segment:visit(segment['recipe'])
    visit(recipe)
    return (len(members),len(recipe.get('segments',[])),len(json.dumps(recipe,sort_keys=True)))


def choose_record(records):
    eligible=[r for r in records if r['assessment']['primary_eligible'] and r['october_guard']['passed']]
    if not eligible:return None
    best=min(r['assessment']['primary_score'] for r in eligible)
    tied=[r for r in eligible if r['assessment']['primary_score']<=best+1e-12]
    return min(tied,key=lambda r:(not r['assessment']['balanced'],recipe_complexity(r['recipe']),
                                  r['router_id'],r['count_id'],r['regular_id']))


def check_frozen(path):
    path=Path(path)
    obj=frozen_record(path)
    return obj,{path.relative_to(ROOT).as_posix():sha(path),
                path.with_suffix('.sha256').relative_to(ROOT).as_posix():sha(path.with_suffix('.sha256'))}


def manifest_inventory(directory):
    manifest_path=directory/'output_manifest.json'
    manifest=read(manifest_path)
    result={manifest_path.relative_to(ROOT).as_posix():sha(manifest_path)}
    for name,digest in manifest['files'].items():
        path=directory/name
        if sha(path)!=digest:
            raise ValueError('Component output changed: '+str(path))
        result[path.relative_to(ROOT).as_posix()]=digest
    return result


def component_provenance(directory):
    """Check the immutable artifacts and their recorded fitting source snapshots."""
    protocol,files=check_frozen(directory/'protocol.json')
    for name,digest in protocol.get('files',protocol.get('source_hashes',{})).items():
        path=ROOT/name
        snapshot=directory/'source_snapshot'/name
        if snapshot.exists() and path.suffix=='.py':
            path=snapshot
        if sha(path)!=digest:
            raise ValueError('Component provenance changed: '+str(path))
        files[path.relative_to(ROOT).as_posix()]=digest
    return protocol,files


def code_paths():
    return [Path(__file__),ROOT/'scripts/research_smape_wape.py',ROOT/'scripts/research_count_distribution.py',
            ROOT/'app/models/research_model.py',ROOT/'app/calibration.py',ROOT/'app/features.py',
            ROOT/'scripts/research_weekly.py',ROOT/'model/config.json']


def initialize(out):
    if (out/'protocol.json').exists():return verify(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose empty selection directory')
    out.mkdir(parents=True,exist_ok=True)
    hashes={}
    router_protocol,inventory=check_frozen(ROUTER/'protocol.json');hashes.update(inventory)
    _,inventory=check_frozen(ROUTER/'primary_diagnostic_protocol.json');hashes.update(inventory)
    routers,inventory=check_frozen(ROUTER/'primary_pareto.json');hashes.update(inventory)
    if len(routers['candidates'])!=5:raise ValueError('Expected five fixed MILP routers')
    for directory in (COUNT,DECODER,REGULAR):
        _,inventory=component_provenance(directory);hashes.update(inventory)
        _,inventory=check_frozen(directory/'development.json');hashes.update(inventory)
        hashes.update(manifest_inventory(directory))
    _,inventory=check_frozen(REGULAR/'origin_alignment.json');hashes.update(inventory)
    for origin in [OCTOBER,*ORIGINS]:
        directory=SOURCE/origin
        complete=read(directory/'complete.json')
        hashes[(directory/'complete.json').relative_to(ROOT).as_posix()]=sha(directory/'complete.json')
        for name in ['evaluation.parquet',*[f'{m}-values.pkl' for m in ['dynamics','mae','quantile','poisson','tweedie']]]:
            path=directory/name
            if sha(path)!=complete['files'][name]:raise ValueError('OOF cache changed: '+str(path))
            hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in code_paths():
        hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        dest=out/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    protocol=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),origins=ORIGINS,
        champion_recipe=router_protocol['champion_recipe'],router_count=6,
        count_options=count_options(),regular_options=regular_options(),combinations=6*15*13,
        overwrite_order=['router','count_scope','regular_scope'],base_for_regular_blend='original champion',
        recipe_segments='five disjoint past mean_28*7 bins; adjacent equal actions may merge',
        primary='both mean SMAPE/WAPE ratios<1; each fair-origin primary ratio<=1.03',
        balanced_tag='primary plus mean auxiliary<=1.03 and each auxiliary<=1.08',
        score='.7*mean_smape_ratio+.3*mean_wape_ratio',
        tie='within1e-12 prefer balanced, fewer unique trained members, fewer routes, shorter recipe, lexical component IDs',
        october_guard='full composite when no count; otherwise separately available router and regular components, primary<=1.03; no full October count claim',
        source_hashes=hashes,confirmation_origin='2026-05-24',confirmation_read=False,
        previous_history_inspected=True,development_adaptive=True,model_fit_executed=False,
        final_selection_before_confirmation=True)
    freeze_record(out/'protocol.json',protocol)
    print('JOINT PROTOCOL FROZEN',sha(out/'protocol.json'),flush=True)
    return protocol


def verify(out):
    protocol=frozen_record(out/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen joint input changed: '+name)
    return protocol


def load_checked(path,length):
    value=np.load(path,allow_pickle=False)
    if value.shape!=(length,) or not np.isfinite(value).all() or (value<0).any():
        raise ValueError('Invalid component predictions: '+str(path))
    return value


def canonical_key_sha(frame):
    content=frame[['week',*KEY]].astype(str).to_csv(index=False,lineterminator='\n').encode()
    return hashlib.sha256(content).hexdigest()


def validate_regular_array(base,raw,array,weekly,spec):
    mask=in_scope(weekly,*REGULAR_SCOPES[spec['scope']])
    expected=base.copy();mixed=(1-spec['blend'])*base+spec['blend']*np.maximum(raw,0.)
    expected[mask]=np.where(base[mask]>0,mixed[mask],0.)
    if not np.array_equal(expected,array):raise ValueError('Regular route differs from original-champion semantics')


def load_components(origin,protocol,router_recipes):
    frame,values,base,y,codes,den,champion=load_origin(origin,protocol['champion_recipe'])
    weekly=7*frame.mean_28.to_numpy(float)
    alignment=frozen_record(REGULAR/'origin_alignment.json')[origin]
    if canonical_key_sha(frame)!=alignment['ordered_key_sha256']:
        raise ValueError('Regular/champion ordered keys differ')
    if hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest()!=alignment['truth_float64_le_sha256']:
        raise ValueError('Regular/champion facts differ')
    routers={name:recipe_prediction(values,frame,recipe) for name,recipe in router_recipes.items()}
    regular={}
    raw={v:load_checked(REGULAR/origin/f'{v}.raw.npy',len(frame)) for v in VARIANTS}
    for name,spec in regular_options().items():
        if spec is None:continue
        q=load_checked(REGULAR/origin/f'{name}.prediction.npy',len(frame))
        validate_regular_array(base,raw[spec['variant']],q,weekly,spec)
        regular[name]=q
    count={}
    if origin in ORIGINS:
        from scripts.research_count_distribution import decode
        from threadpoolctl import threadpool_limits
        fit=frozen_record(COUNT/origin/'fit.json')
        probability=np.load(COUNT/origin/'probability.npy',allow_pickle=False)
        if probability.shape!=(len(frame),8) or not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1),1):
            raise ValueError('Invalid count distribution shape/order')
        for value in LAMBDAS:
            with threadpool_limits(limits=1,user_api='blas'):
                decoded=decode(probability,np.asarray(fit['representatives'],float),fit['training_mean'],value)
            directory=COUNT if value<=2 else DECODER
            for scope,bounds in COUNT_SCOPES.items():
                name=f'{scope}_lambda{value:g}'
                q=load_checked(directory/origin/f'{name}.prediction.npy',len(frame))
                expected=np.where(in_scope(weekly,*bounds),decoded,base)
                if not np.array_equal(q,expected):raise ValueError('Count decoding or source order differs: '+name)
                count[name]=q
    masks=[in_scope(weekly,*bounds) for bounds in BINS]
    if not np.all(np.sum(masks,axis=0)==1):raise ValueError('Past-volume bins are not a partition')
    return dict(frame=frame,values=values,base=base,y=y,codes=codes,den=den,champion=champion,
                weekly=weekly,routers=routers,count=count,regular=regular,masks=masks,raw_regular=raw,
                key_sha256=canonical_key_sha(frame))


def october_guard(data,router_id,count_spec,regular_id,regular_spec,champion):
    if count_spec is None:
        q=combine_predictions(data['routers'][router_id],None,
            data['regular'].get(regular_id),data['weekly'],None,regular_spec)
        measured=fast_metrics(data['y'],q,data['codes'])
        ratios={k:measured[k]/champion[k] for k in NAMES}
        return dict(passed=all(ratios[k]<=1.03 for k in ['smape','wape']),
                    full_composite_available=True,metrics=measured,ratios=ratios)
    components={'router':data['routers'][router_id]}
    if regular_spec is not None:components['regular']=data['regular'][regular_id]
    records={name:fast_metrics(data['y'],q,data['codes']) for name,q in components.items()}
    ratios={name:{k:m[k]/champion[k] for k in NAMES} for name,m in records.items()}
    return dict(passed=all(r[k]<=1.03 for r in ratios.values() for k in ['smape','wape']),
                full_composite_available=False,count_component_unavailable=True,
                available_components=records,available_component_ratios=ratios)


def selection_stage(out,protocol):
    if (out/'selection.json').exists():return frozen_record(out/'selection.json')
    started=time.perf_counter()
    pareto=frozen_record(ROUTER/'primary_pareto.json')
    router_recipes={'champion':protocol['champion_recipe'],**{
        f'router{i}':r['recipe'] for i,r in enumerate(pareto['candidates'])}}
    data={o:load_components(o,protocol,router_recipes) for o in [OCTOBER,*ORIGINS]}
    champions={o:data[o]['champion'] for o in ORIGINS}
    # Every output belongs to one full pair-series bin, so metric sufficient
    # statistics can be added exactly. Precompute once for each component/bin.
    stats={}
    for o in ORIGINS:
        d=data[o];stats[o]={}
        for kind in ['routers','count','regular']:
            stats[o][kind]={name:[sufficient_stats(d['y'][mask],q[mask],d['codes'][mask])
                                  for mask in d['masks']] for name,q in d[kind].items()}
    records=[]
    for router_id,count_id,regular_id in itertools.product(router_recipes,count_options(),regular_options()):
        count_spec=count_options()[count_id];regular_spec=regular_options()[regular_id]
        metrics={}
        for o in ORIGINS:
            total=np.zeros(6)
            for index,bounds in enumerate(BINS):
                kind,name='routers',router_id
                if count_spec is not None and covers(COUNT_SCOPES[count_spec['scope']],bounds):kind,name='count',count_id
                if regular_spec is not None and covers(REGULAR_SCOPES[regular_spec['scope']],bounds):kind,name='regular',regular_id
                total+=stats[o][kind][name][index]
            metrics[o]=metrics_from_stats(total,data[o]['den'])
        assessed=metric_assessment(metrics,champions)
        guard=october_guard(data[OCTOBER],router_id,count_spec,regular_id,regular_spec,data[OCTOBER]['champion'])
        recipe=compile_recipe(protocol['champion_recipe'],router_recipes[router_id],count_spec,regular_spec)
        records.append(dict(router_id=router_id,count_id=count_id,regular_id=regular_id,
            recipe=recipe,metrics=metrics,assessment=assessed,october_guard=guard))
    if len(records)!=protocol['combinations']:raise ValueError('Declared search budget differs')
    chosen=choose_record(records)
    ranked=sorted(records,key=lambda r:r['assessment']['primary_score'])
    summary=dict(protocol_sha256=sha(out/'protocol.json'),combinations=len(records),champion_metrics=champions,
        primary_eligible_count=sum(r['assessment']['primary_eligible'] and r['october_guard']['passed'] for r in records),
        balanced_count=sum(r['assessment']['balanced'] and r['october_guard']['passed'] for r in records),
        top20=ranked[:20],selected=chosen,seconds=time.perf_counter()-started,confirmation_read=False)
    freeze_record(out/'development.json',summary)
    if chosen is not None:
        # Independent full replay guards the statistic combiner and writes the
        # exact pre-confirmation arrays for the parent's runtime parity check.
        for o in ORIGINS:
            d=data[o]
            q=combine_predictions(d['routers'][chosen['router_id']],d['count'].get(chosen['count_id']),
                d['regular'].get(chosen['regular_id']),d['weekly'],count_options()[chosen['count_id']],
                regular_options()[chosen['regular_id']])
            replay=fast_metrics(d['y'],q,d['codes'])
            for key in NAMES:
                if not np.isclose(replay[key],chosen['metrics'][o][key],rtol=1e-10,atol=1e-10):
                    raise ValueError('Selected forecast differs from sufficient-statistic search')
            directory=out/o;directory.mkdir(exist_ok=True)
            np.save(directory/'selected.prediction.npy',q)
        selection=dict(status='frozen_one_recipe_before_confirmation',recipe=chosen['recipe'],components={
            k:chosen[k] for k in ['router_id','count_id','regular_id']},
            assessment=chosen['assessment'],metrics=chosen['metrics'],october_guard=chosen['october_guard'],
            protocol_sha256=sha(out/'protocol.json'),development_sha256=sha(out/'development.json'),
            ordered_key_sha256={o:data[o]['key_sha256'] for o in ORIGINS},
            prediction_sha256={o:sha(out/o/'selected.prediction.npy') for o in ORIGINS},
            confirmation_origin='2026-05-24',confirmation_read=False,promotion_executed=False,
            previous_history_inspected=True)
    else:
        selection=dict(status='no_simultaneous_primary_improvement_candidate',recipe=None,
            protocol_sha256=sha(out/'protocol.json'),development_sha256=sha(out/'development.json'),
            confirmation_read=False,promotion_executed=False)
    freeze_record(out/'selection.json',selection)
    return selection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/smape-wape-joint-20261029')
    parser.add_argument('--stage',choices=['freeze','select'],required=True)
    args=parser.parse_args(argv);out=args.out.resolve();protocol=initialize(out)
    if args.stage=='select':print(json.dumps(selection_stage(out,protocol),indent=2),flush=True)
    verify(out)


if __name__=='__main__':main()
