"""One parent-selected SMAPE/WAPE confirmation, never a confirmation search."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import lightgbm as lgb
import numpy as np
import pandas as pd
import research_events as events
from app.event_features import make_event_features
from app.features import KEY
from app.models.metric_experts import MetricExpertPart
from app.models.research_model import recipe_prediction, recipe_member_names
from scripts.research_weekly import freeze_record, frozen_record, read, sha, fast_metrics

JOINT = ROOT/'artifacts/smape-wape-joint-20261029'


def expert_specs(recipe):
    result = {}
    for name in recipe_member_names(recipe):
        if name.startswith('metric_count_lambda'):
            result[name] = {'kind':'count', 'multiplier':float(name.removeprefix('metric_count_lambda'))}
        elif name.startswith('metric_regular_'):
            result[name] = {'kind':name.removeprefix('metric_regular_')}
    return result


def add_cached_features(frame):
    x = make_event_features(frame)
    return pd.concat([frame, x[[c for c in x if c not in frame]]], axis=1)


def verified_member(directory, origin, path):
    manifest = read(directory/'output_manifest.json')
    relative = f'{origin}/{path}'
    target = directory/relative
    if sha(target) != manifest['files'][relative]:
        raise ValueError('Metric member file changed: '+relative)
    return target


def restore_expert(name, spec, origin):
    part = MetricExpertPart(spec, 6)
    if spec['kind'] == 'count':
        directory = ROOT/'artifacts/count-distribution-20261029'
        metadata = read(verified_member(directory,origin,'fit.json'))
        part.booster = lgb.Booster(model_file=str(verified_member(directory,origin,'count_model.txt')))
        part.categories = metadata['categories']
        part.representatives = np.array(metadata['representatives'])
        part.training_mean = metadata['training_mean']
    else:
        directory = ROOT/'artifacts/regular-quantity-20261029'
        kind = spec['kind']
        part.booster = lgb.Booster(model_file=str(verified_member(directory,origin,kind+'.txt')))
        part.categories = read(verified_member(directory,origin,kind+'.categories.json'))
    return part


def evaluation_frame(origin):
    old = pd.read_parquet(events.CHAMPION_ROOT/origin/'evaluation.parquet')
    raw = events.cached(origin,f'{origin}-False.parquet',labels=False)
    if not old[['week',*KEY]].astype(str).equals(raw[['week',*KEY]].astype(str)):
        raise ValueError('Metric expert evaluation keys differ')
    if not np.array_equal(old.actual,raw.actual):
        raise ValueError('Metric expert evaluation facts differ')
    extended = add_cached_features(raw)
    frame = pd.concat([old,extended[[c for c in extended if c not in old]]],axis=1)
    return frame


def runtime_parity(out, selection):
    if (out/'runtime_parity.json').exists():
        return frozen_record(out/'runtime_parity.json')
    reports = {}
    specs = expert_specs(selection['recipe'])
    for origin in ['2026-01-18','2026-03-22']:
        events.inventory(origin)
        frame = evaluation_frame(origin)
        values = {'dynamics':pickle.loads((events.CHAMPION_ROOT/origin/'dynamics-values.pkl').read_bytes())}
        for name,spec in specs.items():
            part = restore_expert(name,spec,origin)
            values[name] = {'raw':part.predict(frame)}
        actual = recipe_prediction(values,frame,selection['recipe'])
        path = JOINT/origin/'selected.prediction.npy'
        if sha(path) != selection['prediction_sha256'][origin]:
            raise ValueError('Selected development prediction changed')
        expected = np.load(path)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        reports[origin] = {'rows':len(actual),'max_abs_difference':float(np.abs(actual-expected).max())}
    report = {'selection_sha256':sha(JOINT/'selection.json'),'origins':reports,'valid':True}
    freeze_record(out/'runtime_parity.json',report)
    return report


def initialize(out):
    selection = frozen_record(JOINT/'selection.json')
    if selection.get('status') != 'frozen_one_recipe_before_confirmation':
        raise ValueError('No selected recipe')
    if (out/'protocol.json').exists():
        protocol = frozen_record(out/'protocol.json')
        for name,digest in protocol['files'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Frozen confirmation source/input changed: '+name)
        return selection,protocol
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose empty confirmation directory')
    out.mkdir(parents=True,exist_ok=True)
    paths = [Path(__file__),JOINT/'selection.json',JOINT/'selection.sha256',
             ROOT/'app/models/metric_experts.py',ROOT/'app/models/research_model.py',
             ROOT/'app/sequence_features.py',ROOT/'app/event_features.py',ROOT/'app/features.py',
             ROOT/'app/calibration.py',ROOT/'scripts/research_events.py',ROOT/'scripts/research_weekly.py',
             *[ROOT/'model'/n for n in ['config.json','model.pkl','preprocessor.pkl']]]
    protocol = {'created_at':datetime.now(timezone.utc).isoformat(),'selection_sha256':sha(JOINT/'selection.json'),
                'recipe':selection['recipe'],'expert_specs':expert_specs(selection['recipe']),
                'confirmation_origin':'2026-05-24','one_frozen_candidate':True,
                'acceptance':'SMAPE AND WAPE strictly lower than refitted champion; all auxiliary metrics reported',
                'balanced_tag':'all auxiliary ratios<=1.03', 'previous_history_inspected':True,
                'files':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    freeze_record(out/'protocol.json',protocol)
    snapshot = out/'source_snapshot'
    for path in paths:
        if path.suffix == '.py':
            target = snapshot/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
    return selection,protocol


def confirmation(out, selection, protocol):
    runtime_parity(out,selection)
    if (out/'decision.json').exists():
        return frozen_record(out/'decision.json')
    origin = protocol['confirmation_origin']
    # Candidate and protocol are immutable BEFORE this first May access.
    schedule, inventory = events.inventory(origin)
    freeze_record(out/'confirmation_input_hashes.json',inventory)
    monitor = events.MemoryMonitor(5.)
    monitor.thread.start()
    started = time.perf_counter()
    try:
        train = add_cached_features(events.load_frames(origin,schedule,schedule['inner'],schedule['inner_origin']))
        valid = add_cached_features(events.load_frames(origin,schedule,[schedule['valid']],origin))
        iterations = {}
        for name,spec in protocol['expert_specs'].items():
            print('CONFIRM INNER',name,len(train),flush=True)
            part = MetricExpertPart(spec,6).fit(train,valid)
            iterations[name] = part.selected_iterations
            del part; gc.collect(); monitor.check()
        del train,valid;gc.collect()
        freeze_record(out/'iterations.json',iterations)
        refit = add_cached_features(events.load_frames(origin,schedule,schedule['refit'],origin))
        frame = evaluation_frame(origin)
        values = {'dynamics':pickle.loads((events.CHAMPION_ROOT/origin/'dynamics-values.pkl').read_bytes())}
        for name,spec in protocol['expert_specs'].items():
            print('CONFIRM PAST-ONLY REFIT',name,len(refit),iterations[name],flush=True)
            part = MetricExpertPart(spec,6).fit(refit,iterations=iterations[name])
            values[name] = {'raw':part.predict(frame)}
            (out/(name+'.pkl')).write_bytes(pickle.dumps(part,protocol=4))
            np.save(out/(name+'.prediction.npy'),values[name]['raw'])
            del part;gc.collect();monitor.check()
        del refit;gc.collect()
        candidate = recipe_prediction(values,frame,protocol['recipe'])
        champion_recipe = read(ROOT/'model/config.json')['model_params']['recipe']
        champion = recipe_prediction(values,frame,champion_recipe)
        y = frame.actual.to_numpy(float)
        codes,_ = pd.factorize(pd.MultiIndex.from_frame(frame[KEY]))
        cm,bm = fast_metrics(y,candidate,codes),fast_metrics(y,champion,codes)
        ratios = {k:cm[k]/bm[k] for k in ['smape','wape','rmse','bias','rmsle']}
        accepted = cm['smape'] < bm['smape']-1e-8 and cm['wape'] < bm['wape']-1e-8
        result = {'selection_sha256':sha(JOINT/'selection.json'),'protocol_sha256':sha(out/'protocol.json'),
                  'origin':origin,'champion':bm,'candidate':cm,'ratios':ratios,'accepted':bool(accepted),
                  'balanced':all(ratios[k]<=1.03 for k in ['rmse','bias','rmsle']),
                  'seconds':time.perf_counter()-started,'peak_sampled_rss_gib':monitor.peak/2**30,
                  'confirmation_candidate_count':1,'production_replaced':False,'previous_history_inspected':True}
        np.save(out/'candidate.prediction.npy',candidate)
        freeze_record(out/'decision.json',result)
        return result
    finally:
        monitor.stop.set();monitor.thread.join(timeout=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/smape-wape-confirmation-20261029')
    parser.add_argument('--stage',choices=['parity','confirm'],required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    selection,protocol = initialize(out)
    result = runtime_parity(out,selection) if args.stage=='parity' else confirmation(out,selection,protocol)
    print(json.dumps(result,indent=2),flush=True)
    initialize(out)


if __name__ == '__main__':
    main()
