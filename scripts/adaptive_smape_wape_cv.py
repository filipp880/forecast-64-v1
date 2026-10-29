"""Explicit adaptive historical CV after the reserved May test was spent.

All component models remain past-only. Recipe selection uses January, March and
May outcomes; these scores MUST NOT be described as independent confirmation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import itertools
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
import research_events as events
from app.models.metric_experts import MetricExpertPart, bayes_quantities
from app.models.research_model import recipe_prediction, recipe_member_names
from scripts.confirm_smape_wape import evaluation_frame, restore_expert, add_cached_features
from scripts.research_smape_wape import (policy_grid,policy_id,policy_prediction,segment_mask,
    sufficient_stats,metrics_from_stats,metric_denominators,build_recipe,MEMBERS,SEGMENTS)
from scripts.research_weekly import freeze_record,frozen_record,read,sha,fast_metrics

ORIGINS = ['2026-01-18','2026-03-22','2026-05-24']
LAMBDAS = [0.,1.,2.,3.,4.,8.,16.]
CONFIRM = ROOT/'artifacts/smape-wape-confirmation-20261029'


def initialize(out):
    if (out/'protocol.json').exists():
        protocol=frozen_record(out/'protocol.json')
        for name,digest in protocol['files'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen adaptive CV source changed: '+name)
        return protocol
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose empty adaptive-CV directory')
    out.mkdir(parents=True,exist_ok=True)
    champion=read(ROOT/'model/config.json')['model_params']['recipe']
    paths=[Path(__file__),ROOT/'app/models/metric_experts.py',ROOT/'app/models/research_model.py',
        ROOT/'app/calibration.py',ROOT/'app/event_features.py',ROOT/'scripts/research_smape_wape.py',
        CONFIRM/'decision.json',CONFIRM/'metric_count_lambda3.pkl',
        CONFIRM/'metric_regular_recent_mae.pkl',ROOT/'model/config.json']
    inventory={}
    schedules={}
    for origin in ORIGINS:
        schedule,files=events.inventory(origin);inventory.update(files);schedules[origin]=schedule
        directory=events.CHAMPION_ROOT/origin
        complete=read(directory/'complete.json')
        for member in ['dynamics']:
            path=directory/(member+'-values.pkl')
            if sha(path)!=complete['files'][path.name]:raise ValueError('Legacy OOF member changed')
            paths.append(path)
    common_grids=policy_grid(champion)
    for grid in common_grids.values():
        grid['policies']={name:policy for name,policy in grid['policies'].items()
            if policy.get('zero') or (policy['recipe']['source'] in ['dynamics','tsb','sba']
                and policy['recipe'].get('aux') in [None,'tsb','sba'])}
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),origins=ORIGINS,
        may_was_previously_reserved_and_is_now_explicitly_used_for_selection=True,
        evaluation_kind='adaptive_historical_cross_validation_NOT_independent_confirmation',
        champion_recipe=champion,base_grids=common_grids,count_lambdas=LAMBDAS,
        legacy_profiles='only dynamics and past tsb/sba statistics; other May members were never fitted and are not invented',
        new_expert_policies='count only volume .5..10; recent regular only volume>=2, blends .25/.5/1, one-sided or exact zero masks',
        solver='one fixed alpha .7*meanSMAPEratio+.3*meanWAPEratio; one policy per five nonoverlapping past-demand bins',
        constraints='each origin both primary ratios<=1; each mean primary<=.9999; each auxiliary ratio<=1.15',
        files={**inventory,**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}},schedules=schedules,
        independent_confirmation_available=False,closed_test_used=False,production_changed=False)
    freeze_record(out/'protocol.json',protocol)
    for path in paths:
        if path.suffix=='.py':
            target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
    return protocol


def may_balanced_expert(out,protocol):
    path=out/'may_recent_balanced_mae.pkl'
    if path.exists():return pickle.loads(path.read_bytes())
    from scripts.research_regular_quantity import load_regular
    schedule=protocol['schedules']['2026-05-24']
    train=add_cached_features(load_regular('2026-05-24',schedule['inner'],schedule['inner_origin']))
    valid=add_cached_features(load_regular('2026-05-24',[schedule['valid']],'2026-05-24'))
    part=MetricExpertPart({'kind':'recent_balanced_mae'},6).fit(train,valid)
    iterations=part.selected_iterations
    del train,valid,part;gc.collect()
    refit=add_cached_features(load_regular('2026-05-24',schedule['refit'],'2026-05-24'))
    part=MetricExpertPart({'kind':'recent_balanced_mae'},6).fit(refit,iterations=iterations)
    path.write_bytes(pickle.dumps(part,protocol=4))
    freeze_record(out/'may_recent_balanced_mae.json',{'sha256':sha(path),'iterations':iterations,
        'past_only':True,'labels_end_by':'2026-05-24','selection_history_includes_may':True})
    del refit;gc.collect()
    return part


def candidate_values(origin,out,protocol):
    frame=evaluation_frame(origin)
    directory=events.CHAMPION_ROOT/origin
    values={'dynamics':pickle.loads((directory/'dynamics-values.pkl').read_bytes())}
    if origin=='2026-05-24':
        count=pickle.loads((CONFIRM/'metric_count_lambda3.pkl').read_bytes())
    else:count=restore_expert('metric_count_lambda3',{'kind':'count','multiplier':3.},origin)
    probabilities=count.booster.predict(count.encode(frame),num_threads=6)
    for multiplier in LAMBDAS:
        values[f'metric_count_lambda{multiplier:g}']={'raw':bayes_quantities(probabilities,count.representatives,count.training_mean,multiplier)}
    for kind in ['recent_mae','recent_balanced_mae']:
        if origin=='2026-05-24':
            part=pickle.loads((CONFIRM/'metric_regular_recent_mae.pkl').read_bytes()) if kind=='recent_mae' else may_balanced_expert(out,protocol)
        else:part=restore_expert('metric_regular_'+kind,{'kind':kind},origin)
        values['metric_regular_'+kind]={'raw':part.predict(frame)}
    return frame,values


def grids(protocol):
    result=protocol['base_grids']
    for name,lower,upper in SEGMENTS:
        policies=result[name]['policies']
        sources=[]
        if name in ['low','medium']:sources += [f'metric_count_lambda{x:g}' for x in LAMBDAS]
        if lower>=2:sources += ['metric_regular_recent_mae','metric_regular_recent_balanced_mae']
        for source,weight,mask in itertools.product(sources,[.25,.5,1.],['none','one_sided','exact']):
            policy={'recipe':{'source':source,'post':{'round_int':False}},'blend_weight':weight}
            if mask=='one_sided':policy['keep_base_zeros']=True
            elif mask=='exact':policy['keep_base_mask']=True
            policies[policy_id(policy)]=policy
    return result


def predict_policy(values,frame,base,policy):
    p=policy_prediction(values,frame,base,policy)
    if policy.get('keep_base_zeros'):p=np.where(base>0,p,0.)
    return p


def development(out,protocol):
    if (out/'selection.json').exists():return frozen_record(out/'selection.json')
    policies=grids(protocol)
    variables=[(name,identity) for name,_,_ in SEGMENTS for identity in policies[name]['policies']]
    n=len(variables)
    groups=np.zeros((5,n));primary=np.zeros((6,n));aux=np.zeros((9,n));stats={};data={};champions={}
    for oi,origin in enumerate(ORIGINS):
        frame,values=candidate_values(origin,out,protocol)
        base=recipe_prediction(values,frame,protocol['champion_recipe'])
        y=frame.actual.to_numpy(float);codes,_=pd.factorize(pd.MultiIndex.from_frame(frame[['warehouse_name','article']]))
        den=metric_denominators(y,codes);baseline=fast_metrics(y,base,codes);champions[origin]=baseline
        stats[origin]={}
        for j,(name,identity) in enumerate(variables):
            grid=policies[name];mask=segment_mask(frame,grid['lower'],grid['upper'])
            p=predict_policy(values,frame,base,grid['policies'][identity])
            s=sufficient_stats(y[mask],p[mask],codes[mask]);stats[origin][j]=s
            m=metrics_from_stats(s,den)
            primary[2*oi,j]=m['smape']/baseline['smape'];primary[2*oi+1,j]=m['wape']/baseline['wape']
            aux[3*oi,j]=s[2]/den['rows']/baseline['rmse']**2
            aux[3*oi+1,j]=m['bias']/baseline['bias']
            aux[3*oi+2,j]=s[4]/den['rows']/baseline['rmsle']**2
            groups[[v[0] for v in SEGMENTS].index(name),j]=1
        data[origin]=(frame,values,base,y,codes,den)
        print('ADAPTIVE CV INPUTS',origin,len(variables),flush=True)
    mean_smape=primary[::2].mean(axis=0);mean_wape=primary[1::2].mean(axis=0)
    matrix=np.vstack([groups,primary,mean_smape,mean_wape,aux])
    lower=np.r_[np.ones(5),[-np.inf]*17]
    upper=np.r_[np.ones(5),np.ones(6),[.9999,.9999],np.tile([1.15**2,1.15,1.15**2],3)]
    if len(lower)!=len(upper):raise AssertionError('Constraint dimensions differ')
    complexity=np.array([int(identity!='champion') for _,identity in variables])
    objective=.7*mean_smape+.3*mean_wape+1e-9*complexity
    solved=milp(objective,integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lower,upper),
                options={'time_limit':60.,'mip_rel_gap':1e-9})
    result={'protocol_sha256':sha(out/'protocol.json'),'evaluation_kind':protocol['evaluation_kind'],
        'solver_status':int(solved.status),'solver_message':solved.message,'independent_confirmation':False,
        'production_changed':False,'candidate_policy_count':n,'champion':champions}
    if solved.x is None:
        result['accepted']=False
    else:
        chosen={variables[j][0]:variables[j][1] for j in np.flatnonzero(solved.x>.5)}
        recipe=build_recipe(protocol['champion_recipe'],chosen,policies)
        metrics={};ratios={}
        for origin,(frame,values,base,y,codes,den) in data.items():
            p=recipe_prediction(values,frame,recipe)
            metrics[origin]=fast_metrics(y,p,codes)
            ratios[origin]={k:metrics[origin][k]/champions[origin][k] for k in ['smape','wape','rmse','bias','rmsle']}
            np.save(out/(origin+'.prediction.npy'),p)
        means={k:float(np.mean([r[k] for r in ratios.values()])) for k in ratios[ORIGINS[0]]}
        accepted=all(r[k]<=1+1e-9 for r in ratios.values() for k in ['smape','wape']) and means['smape']<1 and means['wape']<1
        result.update(accepted=bool(accepted),recipe=recipe,policies=chosen,metrics=metrics,ratios=ratios,
            mean_ratios=means,primary_score=.7*means['smape']+.3*means['wape'])
    freeze_record(out/'selection.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/smape-wape-adaptive-cv-20261029')
    args=parser.parse_args(argv);out=args.out.resolve();protocol=initialize(out)
    result=development(out,protocol)
    initialize(out)
    print(result,flush=True)


if __name__=='__main__':main()
