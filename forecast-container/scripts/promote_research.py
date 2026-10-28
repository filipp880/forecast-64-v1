"""Verify a confirmed recipe, preserve current production and install its refit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import uuid
from pathlib import Path
import pandas as pd

CONTAINER=Path(__file__).resolve().parents[1]
ROOT=CONTAINER.parent
sys.path[:0]=[str(CONTAINER),str(CONTAINER/'scripts')]
from research_weekly import verify_protocol,fast_metrics,ratio,NAMES
from app.models.research_model import recipe_prediction
from app.predictor import Predictor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--model-dir',type=Path,default=CONTAINER/'model')
    args=ap.parse_args(argv);run=args.run.resolve();current=args.model_dir.resolve()
    if ROOT not in current.parents:raise ValueError('Production model must stay inside the project')
    protocol=verify_protocol(run);selection=read(run/'selection.json');decision=read(run/'decision.json')
    if sha(run/'selection.json')!=(run/'selection.sha256').read_text().strip() or decision['selection_sha256']!=sha(run/'selection.json'):
        raise ValueError('Frozen selection changed')
    if selection['selected_recipe']!=decision['selected_recipe']:raise ValueError('Decision recipe differs from frozen selection')
    fold=run/protocol['confirmation_origin'];complete=read(fold/'complete.json')
    for name,h in complete['files'].items():
        if sha(fold/name)!=h:raise ValueError('Confirmation artifact changed: '+name)
    frame=pd.read_parquet(fold/'evaluation.parquet');values=pickle.loads((fold/'values.pkl').read_bytes())
    codes,_=pd.factorize(pd.MultiIndex.from_frame(frame[['warehouse_name','article']]))
    for name,recipe in [('baseline',{'source':'base','post':{'round_int':True}}),('candidate',selection['selected_recipe'])]:
        actual=fast_metrics(frame.actual.to_numpy(float),recipe_prediction(values,frame,recipe),codes)
        if any(abs(v-decision[name][k])>1e-10 for k,v in actual.items()):raise ValueError('Recorded confirmation metrics changed')
    base,candidate=decision['baseline'],decision['candidate']
    eligible=ratio(candidate,base)<=.995 and candidate['smape']<=base['smape']*1.01 and sum(candidate[k]<base[k] for k in NAMES)>=3
    if bool(decision['promote'])!=bool(eligible):raise ValueError('Decision violates the frozen promotion rule')
    backup=run/'production_before_promotion';backup.mkdir(exist_ok=True)
    if not (backup/'manifest.json').exists():
        before={}
        for name in ['model.pkl','preprocessor.pkl','config.json']:
            if (backup/name).exists():raise FileExistsError('Incomplete backup already exists')
            shutil.copyfile(current/name,backup/name);before[name]=sha(backup/name)
        (backup/'manifest.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    before=read(backup/'manifest.json')
    for name,h in before.items():
        if sha(backup/name)!=h:raise ValueError('Production backup changed')
    if decision['promote']:
        final=run/'final';bundle=Predictor(final).load()
        if bundle.config['provenance']['decision_sha256']!=sha(run/'decision.json'):raise ValueError('Final model decision mismatch')
        if sha(current/'model.pkl') not in [before['model.pkl'],bundle.config['model_artifact_sha256']]:raise ValueError('Unexpected concurrent production change')
        for name in ['model.pkl','preprocessor.pkl','config.json']:
            destination=current/name;temp=destination.with_name(name+'.'+uuid.uuid4().hex+'.tmp')
            shutil.copyfile(final/name,temp);os.replace(temp,destination)
        print('Promoted selected research bundle')
    else:
        for name,h in before.items():
            if sha(current/name)!=h:raise ValueError('Production changed after rejected candidate')
        (run/'final_training.json').write_text(json.dumps(dict(status='not_refit_candidate_not_promoted',
             retained_model_sha256=sha(current/'model.pkl'),reason='Frozen confirmation rule rejected candidate'),indent=2),encoding='utf-8')
        print('Retained previous production')
    loaded=Predictor(current).load();importance={};tree_counts={}
    def collect(name,model):
        if hasattr(model,'members'):
            for k,m in model.members.items():collect(name+'.'+k,m)
        elif hasattr(model,'models'):
            for k,m in model.models.items():collect(name+'.'+k,m)
        elif hasattr(model,'hurdle'):
            collect(name+'.hurdle',model.hurdle);collect(name+'.lgbm',model.lgbm)
        elif hasattr(model,'columns'):
            tree_counts[name]=int(model.model.tree_count_)
            importance[name]=dict(kind='PredictionValuesChange',features=dict(zip(model.columns,map(float,model.model.get_feature_importance()))))
        else:
            for attr in ['booster','classifier','regressor']:
                booster=getattr(model,attr,None)
                if booster is not None:
                    tree_counts[name+'.'+attr]=int(booster.num_trees())
                    importance[name+'.'+attr]=dict(kind='split',features=dict(zip(booster.feature_name(),map(int,booster.feature_importance('split')))))
    collect('production',loaded.model)
    if decision['promote']:
        training=read(run/'final_training.json')
        if 'iterations' in training:
            training['estimator_best_iteration_after_fixed_refit']=training.pop('iterations')
        training['refit_iterations_requested']={k:read(fold/'iterations.json')[k] for k in loaded.model.members}
        training['actual_tree_counts']=tree_counts
        training['iteration_note']='Fixed refit has no early stopping; SDK best_iteration=0/None means use all fitted trees. Actual counts verified from the saved boosters.'
        (run/'final_training.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
    (run/'feature_importance.json').write_text(json.dumps(dict(model_sha256=sha(current/'model.pkl'),
         note='Usage evidence only; not a causal ablation',members=importance),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Production load/schema checks passed:',loaded.config['model_artifact_sha256'])


if __name__=='__main__':main()
