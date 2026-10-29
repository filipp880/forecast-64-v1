"""Finite segmented search prioritizing simultaneous SMAPE/WAPE improvement.

Uses only three verified historical development OOF caches. May confirmation is
never opened by this script. Segment selection is based on past demand, not ID
or future actuals; every candidate is compared on exactly the champion keys.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.features import KEY
from app.models.research_model import recipe_prediction
from scripts.research_weekly import fast_metrics, freeze_record, frozen_record, read, sha

ORIGINS = ['2025-10-19', '2026-01-18', '2026-03-22']
SOURCE = ROOT/'artifacts/improvements-20261028-r1'
MEMBERS = ['dynamics', 'mae', 'quantile', 'poisson', 'tweedie']
SEGMENTS = [('rare', 0., .5), ('low', .5, 2.), ('medium', 2., 10.),
            ('regular', 10., 50.), ('bulk', 50., None)]
SEARCH_ORDER = ['low', 'medium', 'regular', 'bulk', 'rare']
METRICS = ['smape', 'wape', 'rmse', 'bias', 'rmsle']
GUARDRAILS = dict(mean_smape_ratio_max=1., mean_wape_ratio_max=1.,
                  per_origin_primary_ratio_max=1.01, mean_auxiliary_ratio_max=1.03,
                  per_origin_auxiliary_ratio_max=1.08, preferred_primary_score_max=.995)
BEAM_WIDTH = 64


def policy_id(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]


def segment_mask(frame, lower, upper):
    weekly = 7*frame.mean_28.to_numpy(float)
    return (weekly >= lower) & ((weekly < upper) if upper is not None else True)


def policy_grid(champion):
    """Finite grid declared before inspecting any candidate metrics."""
    grids = {}
    for name, lower, upper in SEGMENTS:
        policies = {'champion': {'recipe': champion, 'blend_weight':1., 'keep_base_mask':False}}

        def add(recipe, weight=1., mask=False):
            policy = dict(recipe=recipe, blend_weight=weight, keep_base_mask=mask)
            policies[policy_id(policy)] = policy

        if name in ('rare', 'low'):
            # source=constant is intentionally avoided; weight_hurdle=0 with
            # scale tiny is not used as a disguised rule. Explicit zero below.
            policies['zero'] = {'zero':True}
            for gate in [.35, .45, .5, .55, .6, .65, .75]:
                for weight in [.3, .6, 1.]:
                    for threshold in [.5, .65]:
                        post = {**champion['post'], 'gate':gate, 'medium_gate':gate,
                                'weight_hurdle':weight, 'round_threshold':threshold}
                        add(dict(source='dynamics', post=post))
            for source in ['mae', 'quantile', 'tsb', 'sba']:
                add(dict(source=source, post={'round_int':True}))
        else:
            for source in ['mae', 'quantile', 'poisson', 'tweedie']:
                for rounded in [False, True]:
                    for weight in [.25, .5, 1.]:
                        add(dict(source=source,post={'round_int':rounded}),weight,True)
                    add(dict(source=source,post={'round_int':rounded}),1.,False)
                for scale in [.85, 1.15]:
                    add(dict(source=source,post={'scale':scale,'round_int':True}),1.,True)
            for scale in [.85, 1., 1.15]:
                for rounded in [False, True]:
                    for weight in [.5, 1.]:
                        add(dict(source='dynamics',post={'weight_hurdle':1.,'gate':0.,
                            'medium_gate':0.,'scale':scale,'round_int':rounded}),weight,True)
            for scale in [.9, 1.1, 1.2]:
                add(dict(source='dynamics',post={**champion['post'],'scale':scale}),1.,True)
            for source in ['tsb','sba']:
                for rounded in [False,True]:
                    for weight in [.5,1.]:
                        add(dict(source=source,post={'round_int':rounded}),weight,True)
            if name == 'medium':
                for gate in [.5,.6,.7]:
                    for threshold in [.5,.65]:
                        add(dict(source='dynamics',post={**champion['post'],'gate':gate,
                            'medium_gate':gate,'round_threshold':threshold}))
        if len(policies) > 100:
            raise AssertionError('Per-segment grid exceeded the declared budget')
        grids[name] = dict(lower=lower, upper=upper, policies=policies)
    return grids


def policy_prediction(values, frame, base, policy):
    if policy.get('zero'):
        return np.zeros(len(frame))
    nested = recipe_prediction(values, frame, policy['recipe'])
    weight = policy.get('blend_weight',1.)
    prediction = (1-weight)*base+weight*nested
    if policy.get('keep_base_mask'):
        prediction = np.where(base>0,np.maximum(prediction,1e-9),0.)
    if not np.isfinite(prediction).all() or (prediction<0).any():
        raise ValueError('Invalid candidate output')
    return prediction


def sufficient_stats(y, p, codes):
    y,p,codes = np.asarray(y,float),np.asarray(p,float),np.asarray(codes,int)
    if not len(y):
        return np.zeros(6)
    count = np.bincount(codes)
    actual = np.bincount(codes,weights=y)
    predicted = np.bincount(codes,weights=p)
    point = np.divide(200*np.abs(y-p),y+p,out=np.zeros(len(y)),where=(y+p)>0)
    means = np.divide(np.bincount(codes,weights=point),count,out=np.zeros(len(count)),where=count>0)
    bias = np.divide(100*np.abs(predicted-actual),actual,out=np.zeros(len(actual)),where=actual>0)
    return np.array([means.sum(),np.abs(y-p).sum(),np.square(y-p).sum(),
                     bias.sum(),np.square(np.log1p(y)-np.log1p(p)).sum(),p.sum()])


def metric_denominators(y,codes):
    count=np.bincount(codes)
    actual=np.bincount(codes,weights=y)
    return dict(rows=len(y),pairs=int((count>0).sum()),positive_pairs=int((actual>0).sum()),
                actual_volume=float(np.sum(y)))


def metrics_from_stats(stats, denominators):
    s,den = np.asarray(stats,float),denominators
    return dict(smape=float(s[0]/den['pairs']),
        wape=float(100*s[1]/den['actual_volume']) if den['actual_volume'] else (0. if s[1]==0 else 100.),
        rmse=float(np.sqrt(s[2]/den['rows'])),
        bias=float(s[3]/den['positive_pairs']) if den['positive_pairs'] else 0.,
        rmsle=float(np.sqrt(s[4]/den['rows'])),
        volume_bias=float(100*(s[5]/den['actual_volume']-1)) if den['actual_volume'] else None)


def assessment(metrics, champions):
    ratios={o:{k:float(metrics[o][k]/champions[o][k]) if champions[o][k] else
                   (1. if metrics[o][k]==0 else float('inf')) for k in METRICS} for o in ORIGINS}
    means={k:float(np.mean([ratios[o][k] for o in ORIGINS])) for k in METRICS}
    score=.7*means['smape']+.3*means['wape']
    primary_ok=means['smape']<1 and means['wape']<1 and all(
        ratios[o][k]<=GUARDRAILS['per_origin_primary_ratio_max'] for o in ORIGINS for k in ['smape','wape'])
    auxiliary_ok=all(means[k]<=GUARDRAILS['mean_auxiliary_ratio_max'] for k in ['rmse','bias','rmsle']) and all(
        ratios[o][k]<=GUARDRAILS['per_origin_auxiliary_ratio_max'] for o in ORIGINS for k in ['rmse','bias','rmsle'])
    excess=sum(max(0,means[k]-GUARDRAILS['mean_auxiliary_ratio_max'])**2 for k in ['rmse','bias','rmsle'])
    excess+=sum(max(0,ratios[o][k]-GUARDRAILS['per_origin_auxiliary_ratio_max'])**2
                for o in ORIGINS for k in ['rmse','bias','rmsle'])
    excess+=sum(max(0,ratios[o][k]-GUARDRAILS['per_origin_primary_ratio_max'])**2
                for o in ORIGINS for k in ['smape','wape'])
    return dict(eligible=bool(primary_ok and auxiliary_ok),preferred=bool(primary_ok and auxiliary_ok and score<=.995),
                primary_score=score,metric_ratios=means,per_origin_ratios=ratios,
                search_rank=score+10*excess)


def code_paths():
    return [Path(__file__),ROOT/'app/models/research_model.py',ROOT/'app/calibration.py',
            ROOT/'app/features.py',ROOT/'scripts/research_weekly.py',ROOT/'model/config.json']


def initialize(out):
    if (out/'protocol.json').exists():
        return verify(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    config=read(ROOT/'model/config.json')
    champion=config['model_params']['recipe']
    if champion.get('segments') or champion['source']!='dynamics':
        raise ValueError('This study expects the original current dynamics champion')
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in code_paths()}
    for origin in ORIGINS:
        directory=SOURCE/origin
        complete=read(directory/'complete.json')
        hashes[(directory/'complete.json').relative_to(ROOT).as_posix()]=sha(directory/'complete.json')
        for name in ['evaluation.parquet',*[f'{m}-values.pkl' for m in MEMBERS]]:
            path=directory/name
            if sha(path)!=complete['files'][name]:
                raise ValueError('Frozen OOF cache mismatch: '+str(path))
            hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    grids=policy_grid(champion)
    protocol=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),origins=ORIGINS,
        champion_recipe=champion,segments=grids,search_order=SEARCH_ORDER,beam_width=BEAM_WIDTH,
        local_keep_by_search_rank=12,local_extra_by_primary_metric=2,local_extra_by_auxiliary=2,
        equivalent_policy_pruning='same exact predictions on all three development segments; prefer champion then fewer extra members',
        guardrails=GUARDRAILS,primary_score='.7*mean_smape_ratio+.3*mean_wape_ratio',
        primary_acceptance='both mean ratios strictly below1; preferred primary_score<=.995',
        source_hashes=hashes,confirmation_origin='2026-05-24',confirmation_read=False,
        limitations=['Previously inspected historical origins; no new pristine holdout',
                     'Adaptive development selection; exactly one frozen candidate may later use confirmation',
                     'No per-ID future lookup; fixed segments only use past weekly mean'])
    for name in code_paths():
        dest=out/'source_snapshot'/name.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,dest)
    freeze_record(out/'protocol.json',protocol)
    print('SMAPE/WAPE PROTOCOL FROZEN',sha(out/'protocol.json'),flush=True)
    return protocol


def verify(out):
    protocol=frozen_record(out/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen study source changed: '+name)
    return protocol


def load_origin(origin,champion):
    directory=SOURCE/origin
    frame=pd.read_parquet(directory/'evaluation.parquet')
    if frame.duplicated(['week',*KEY]).any() or not frame.groupby(KEY,observed=True).size().eq(9).all():
        raise ValueError('Expected exact nine-week key universe')
    if not frame.groupby(KEY,observed=True).mean_28.nunique().le(1).all():
        raise ValueError('Demand segment changed within the forecast horizon')
    values={m:pickle.loads((directory/f'{m}-values.pkl').read_bytes()) for m in MEMBERS}
    base=recipe_prediction(values,frame,champion)
    y=frame.actual.to_numpy(float)
    codes,_=pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))
    den=metric_denominators(y,codes)
    original=fast_metrics(y,base,codes)
    aggregate=metrics_from_stats(sufficient_stats(y,base,codes),den)
    for key in METRICS:
        if not np.isclose(original[key],aggregate[key],rtol=1e-11,atol=1e-11):
            raise AssertionError('Independent sufficient statistics differ: '+key)
    return frame,values,base,y,codes,den,original


def state_metrics(stats,denominators):
    return {o:metrics_from_stats(stats[o],denominators[o]) for o in ORIGINS}


def keep_local_options(records):
    canonical={}
    for name,record in records.items():
        fingerprint=record['prediction_sha256']
        current=canonical.get(fingerprint)
        if current is None or (name!='champion',record['complexity'],name)<(
                current!='champion',records[current]['complexity'],current):
            canonical[fingerprint]=name
    records={name:records[name] for name in canonical.values()}
    ordered=sorted(records,key=lambda n:(records[n]['assessment']['search_rank'],n))
    selected=set(ordered[:12])|{'champion'}
    for key in ['smape','wape']:
        selected.update(sorted(records,key=lambda n:(records[n]['assessment']['metric_ratios'][key],n))[:2])
    selected.update(sorted(records,key=lambda n:(max(records[n]['assessment']['metric_ratios'][k]
                         for k in ['rmse','bias','rmsle']),n))[:2])
    return sorted(selected)


def build_recipe(champion,selection,grids):
    recipe=copy.deepcopy(champion)
    recipe['segments']=[]
    for name,lower,upper in SEGMENTS:
        chosen=selection[name]
        if chosen=='champion':
            continue
        policy=copy.deepcopy(grids[name]['policies'][chosen])
        if policy.pop('zero',False):
            # An explicit supported zero action, represented without a fake
            # scale coefficient. Parent runtime implements segment action.
            segment=dict(min_weekly_mean=lower,max_weekly_mean=upper,action='zero')
        else:
            segment=dict(min_weekly_mean=lower,max_weekly_mean=upper,**policy)
        recipe['segments'].append(segment)
    return recipe


def development(out,protocol):
    if (out/'development.json').exists():
        return frozen_record(out/'development.json')
    started=time.perf_counter()
    grids=protocol['segments']
    universes={o:load_origin(o,protocol['champion_recipe']) for o in ORIGINS}
    champions={o:v[-1] for o,v in universes.items()}
    denominators={o:v[-2] for o,v in universes.items()}
    base_stats={o:sufficient_stats(v[3],v[2],v[4]) for o,v in universes.items()}
    segment_stats={name:{} for name,_,_ in SEGMENTS}
    local={name:{} for name,_,_ in SEGMENTS}
    for name,lower,upper in SEGMENTS:
        fingerprints={}
        for identity,policy in grids[name]['policies'].items():
            stats={}
            fingerprint=hashlib.sha256()
            for origin,(frame,values,base,y,codes,den,original) in universes.items():
                mask=segment_mask(frame,lower,upper)
                prediction=base if identity=='champion' else policy_prediction(values,frame,base,policy)
                stats[origin]=sufficient_stats(y[mask],prediction[mask],codes[mask])
                fingerprint.update(np.asarray(prediction[mask],dtype='<f8').tobytes())
            segment_stats[name][identity]=stats
            fingerprints[identity]=fingerprint.hexdigest()
        champion_segment=segment_stats[name]['champion']
        for identity,stats in segment_stats[name].items():
            replaced={o:base_stats[o]-champion_segment[o]+stats[o] for o in ORIGINS}
            metrics=state_metrics(replaced,denominators)
            policy=grids[name]['policies'][identity]
            complexity=int(policy.get('recipe',{}).get('source') not in (None,'dynamics','tsb','sba'))
            local[name][identity]=dict(assessment=assessment(metrics,champions),metrics=metrics,
                                      prediction_sha256=fingerprints[identity],complexity=complexity)
        print('SEGMENT SEARCHED',name,len(local[name]),flush=True)
    freeze_record(out/'segment_results.json',local)
    options={name:keep_local_options(records) for name,records in local.items()}
    initial=dict(selection={name:'champion' for name,_,_ in SEGMENTS},stats=base_stats,
                 assessment=assessment(champions,champions))
    beam=[initial]
    for name in SEARCH_ORDER:
        expanded=[]
        for state in beam:
            for identity in options[name]:
                chosen={**state['selection'],name:identity}
                stats={o:state['stats'][o]-segment_stats[name][state['selection'][name]][o]
                         +segment_stats[name][identity][o] for o in ORIGINS}
                metrics=state_metrics(stats,denominators)
                expanded.append(dict(selection=chosen,stats=stats,assessment=assessment(metrics,champions)))
        expanded.sort(key=lambda s:(s['assessment']['search_rank'],json.dumps(s['selection'],sort_keys=True)))
        # Preserve two-primary extremes in addition to the main ranked beam.
        selected=expanded[:BEAM_WIDTH-8]
        for key in ['smape','wape']:
            selected.extend(sorted(expanded,key=lambda s:s['assessment']['metric_ratios'][key])[:4])
        unique={tuple(local[n][s['selection'][n]]['prediction_sha256'] for n,_,_ in SEGMENTS):s for s in reversed(selected)}
        beam=list(unique.values())
        print('BEAM',name,len(beam),'best',expanded[0]['assessment']['primary_score'],flush=True)
    eligible=sorted([s for s in beam if s['assessment']['eligible']],key=lambda s:(s['assessment']['primary_score'],json.dumps(s['selection'],sort_keys=True)))
    candidates=eligible[:10] if eligible else sorted(beam,key=lambda s:s['assessment']['search_rank'])[:10]
    records=[]
    for state in candidates:
        recipe=build_recipe(protocol['champion_recipe'],state['selection'],grids)
        metrics=state_metrics(state['stats'],denominators)
        # Exact replay with full forecasts, independent of statistic replacement.
        for origin,(frame,values,base,y,codes,den,original) in universes.items():
            prediction=base.copy()
            for name,lower,upper in SEGMENTS:
                identity=state['selection'][name]
                mask=segment_mask(frame,lower,upper)
                if identity!='champion':
                    candidate=policy_prediction(values,frame,base,grids[name]['policies'][identity])
                    prediction[mask]=candidate[mask]
            replay=fast_metrics(y,prediction,codes)
            for key in METRICS:
                if not np.isclose(replay[key],metrics[origin][key],rtol=1e-10,atol=1e-10):
                    raise AssertionError('Full replay differs from sufficient-statistic search')
        records.append(dict(recipe=recipe,selection=state['selection'],assessment=state['assessment'],metrics=metrics))
    report=dict(protocol_sha256=sha(out/'protocol.json'),champion_metrics=champions,
                local_options=options,candidates=records,selected_candidate=records[0] if eligible else None,
                eligible_count_in_final_beam=len(eligible),seconds=time.perf_counter()-started,
                confirmation_executed=False,production_replaced=False,previously_inspected_history=True)
    freeze_record(out/'development.json',report)
    if report['selected_candidate']:
        freeze_record(out/'selected_recipe.json',report['selected_candidate'])
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/smape-wape-20261029')
    parser.add_argument('--stage',choices=['freeze','development'],required=True)
    args=parser.parse_args(argv)
    out=args.out.resolve()
    protocol=initialize(out)
    if args.stage=='development':
        report=development(out,protocol)
        print(json.dumps({k:report[k] for k in ['eligible_count_in_final_beam','seconds','selected_candidate']},indent=2),flush=True)
    verify(out)


if __name__=='__main__':
    main()
