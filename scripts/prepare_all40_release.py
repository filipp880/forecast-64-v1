"""Package already trained all40 models for serving. This script never trains."""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.models.research_model import WeeklyResearchModel
from app.models.all40_model import All40WeeklyModel,NativePart
from app.advanced_features import ADVANCED_COLUMNS,advanced_spec
from app.predictor import Predictor
import hashlib

RUN=ROOT/'artifacts/all40-0a385edab6b1'
OUT=ROOT/'artifacts/all40-serving-20261030'
PARTS=['count_fine','core_event','core_positive','core_log','core_mae','core_count','core_recent']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def parts_at(origin):
    result={}
    for name in PARTS:
        folder=RUN/'models'/name/origin
        complete=read(folder/'complete.json')
        for f,h in complete['files'].items():
            if sha(folder/f)!=h:raise ValueError('Component hash differs: '+str(folder/f))
        result[name]=NativePart((folder/'refit_model.txt').read_text(encoding='utf-8'),read(folder/'refit_schema.json'))
    return result

def main():
    import lightgbm as lgb
    def forbidden(*args,**kwargs):raise RuntimeError('Training prohibited in serving preparation')
    lgb.train=forbidden
    verification=read(ROOT/'artifacts/all40-review-20261030/verification.json')
    if not verification['verified']:raise ValueError('Results were not verified')
    if sha(ROOT/'model/model.pkl')!='278347022a5d2b2dcf92145155f4a9f728a1e5996a7b89517527bfaddd2d318e':
        raise ValueError('Current baseline changed')
    baseline=WeeklyResearchModel.load(ROOT/'model/model.pkl')
    model=All40WeeklyModel(baseline,parts_at('FINAL'));model.check_schema()
    model_dir=OUT/'model';model_dir.mkdir(parents=True,exist_ok=True)
    model.save(model_dir/'model.pkl')
    config=read(ROOT/'model/config.json');config.update(preprocessing_version=8,feature_columns=ADVANCED_COLUMNS,
        advanced_features=advanced_spec(),model_params=model.params,trained_at=datetime.now(timezone.utc).isoformat())
    selected=read(RUN/'SELECTION.json')['best_mean']
    config['validation']=dict(scheme='retrospective_local_selection_on_january_march_NOT_independent_confirmation',
        independent_confirmation=False,origins=selected['metrics'],weighted_ratio=selected['weighted_ratio'],
        local_rank_score=selected['local_rank_score'],official_score_known=False,
        average_metrics={k:float(np.mean([selected['metrics'][o][k] for o in selected['metrics']])) for k in selected['metrics']['2026-01-18']})
    config['provenance']=dict(history_cutoff='2026-07-26',source_archive_sha256=verification['archive_sha256'],
        selection_sha256=sha(RUN/'SELECTION.json'),winner=selected['id'],packaged_without_training=True,
        baseline_model_sha256=sha(ROOT/'model/model.pkl'),component_iterations=model.selected_iterations)
    preprocessor=pickle.loads((ROOT/'model/preprocessor.pkl').read_bytes())
    preprocessor.update(schema_version=8,feature_columns=ADVANCED_COLUMNS,advanced_features=advanced_spec())
    (model_dir/'preprocessor.pkl').write_bytes(pickle.dumps(preprocessor,protocol=4))
    config.update(model_artifact_sha256=sha(model_dir/'model.pkl'),preprocessor_sha256=sha(model_dir/'preprocessor.pkl'))
    write(model_dir/'config.json',config)
    Predictor(model_dir).load()
    # Replay the complete serving graph with the historically frozen baseline
    # forecast, never with July-trained baseline weights on past evaluation facts.
    bundle=ROOT/'artifacts/colab-sequential-20261029-final/colab_round08'
    replay={}
    for origin in ['2026-01-18','2026-03-22']:
        frame=pd.read_parquet(RUN/'features'/f'{origin}_{origin}-False.parquet.parquet')
        with np.load(bundle/'data'/f'{origin}.npz',allow_pickle=False) as arrays:historical=arrays['baseline'].copy()
        reference=copy.copy(baseline)
        reference.predict=lambda features,p=historical:p.copy()
        evaluation=All40WeeklyModel(reference,parts_at(origin))
        p=evaluation.predict(frame)
        saved=np.load(RUN/'predictions'/f'{selected["id"]}_{origin}.npy',allow_pickle=False)
        maximum=float(np.max(np.abs(p-saved)))
        if maximum!=0:raise ValueError('Serving graph replay differs: '+str(maximum))
        replay[origin]=dict(rows=len(p),max_difference=maximum,baseline_source='historically frozen forecast, no future-fitted baseline used')
    write(OUT/'preparation.json',dict(prepared=True,training_executed=False,production_changed=False,
        model_sha256=config['model_artifact_sha256'],historical_graph_replay=replay,
        source_archive_sha256=verification['archive_sha256']))
    print(json.dumps(read(OUT/'preparation.json'),indent=2))

if __name__=='__main__':main()
