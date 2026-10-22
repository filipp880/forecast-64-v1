"""Выбор на первом окне, единственная проверка выбранного варианта на втором."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.calibration import combine
from app.metrics import METRIC_WEIGHTS
from verified_backtest import metrics

def candidate_grid():
    candidates = {'incumbent': {}}
    for w in [0.0, 0.15, 0.45, 0.6, 1.0]:
        candidates[f'weight_{w}'] = {'weight_hurdle': w}
    for g in [0.4, 0.45, 0.55, 0.6]:
        candidates[f'gate_{g}'] = {'gate': g}
    for scale in [1.05, 1.1, 1.2]:
        candidates[f'scale_{scale}'] = {'scale': scale}
    for w in [0.15, 0.3]:
        candidates[f'expected_{w}'] = {'expected': True, 'weight_hurdle': w}
    for w in [0.1, 0.25, 0.5]:
        candidates[f'mean28_blend_{w}'] = {'mean_weight': w}
    for scale in [1.1, 1.2]:
        candidates[f'regular_scale_{scale}'] = {'regular_scale': scale}
    for g in [0.4, 0.45]:
        candidates[f'medium_gate_{g}'] = {'medium_gate': g}
    for tau in [0.4, 0.6]:
        candidates[f'round_threshold_{tau}'] = {'round_threshold': tau}
    candidates['unrounded'] = {'round_int': False}
    return candidates

def initialize_grid(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'candidate_grid.json'
    value = {'candidates': candidate_grid(), 'promotion_extra': 'at least three of five metrics improve on both origins; confirmation is never used to choose another candidate'}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Сетка уже зафиксирована с другими параметрами')
    else:
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def score(value, baseline):
    return sum((METRIC_WEIGHTS[k] * value[k] / max(baseline[k], 1e-12) for k in METRIC_WEIGHTS))

def predict(df, spec):
    return combine(df.probability.to_numpy(float), df.magnitude.to_numpy(float), df.lgbm_raw.to_numpy(float), df.mean_28.to_numpy(float), df.n_sale_days_90.to_numpy(float), spec)

def summary(df, p):
    return metrics(df.actual.to_numpy(float), p, pd.MultiIndex.from_frame(df[['warehouse_name', 'article']]).to_numpy())

def segment_summary(df, p):
    df = df.copy()
    df['prediction'] = p
    df['frequency'] = pd.cut(df.n_sale_days_90, [-1, 0, 9, 44, 90], labels=['0/90', '1-9/90', '10-44/90', '45-90/90'])
    df['horizon'] = pd.cut(df.h, [0, 7, 28, 64], labels=['1-7', '8-28', '29-64'])
    out = {}
    for col in ['warehouse_name', 'frequency', 'category_lvl_1', 'horizon']:
        out[col] = {str(k): summary(g, g.prediction.to_numpy(float)) for (k, g) in df.groupby(col, observed=True)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('../artifacts/verified'))
    ap.add_argument('--confirm', action='store_true')
    a = ap.parse_args()
    out = a.out
    if not a.confirm:
        if (out / 'selection.json').exists():
            raise FileExistsError('Выбор уже зафиксирован')
        df = pd.read_csv(out / '2026-03-25/predictions.csv.gz')
        base = summary(df, predict(df, {}))
        grid = json.loads((out / 'candidate_grid.json').read_text())['candidates']
        results = {}
        for (name, spec) in grid.items():
            value = summary(df, predict(df, spec))
            idx = score(value, base)
            better = sum((value[k] < base[k] for k in METRIC_WEIGHTS))
            eligible = idx <= 0.995 and value['smape'] <= base['smape'] * 1.01 and (better >= 3)
            results[name] = {'spec': spec, 'metrics': value, 'local_index': idx, 'improved_metrics': better, 'eligible': eligible}
            print(name, round(idx, 5), 'eligible', eligible, flush=True)
        names = [n for (n, r) in results.items() if r['eligible']]
        selected = min(names, key=lambda n: results[n]['local_index']) if names else 'incumbent'
        record = {'selected_name': selected, 'selected_spec': results[selected]['spec'], 'selection_origin': '2026-03-25', 'results': results, 'baseline': base, 'baselines': {'mean28_round': summary(df, np.rint(df.mean_28.to_numpy(float))), 'mean90_round': summary(df, np.rint(df.mean_90.to_numpy(float))), 'zero': summary(df, np.zeros(len(df)))}}
        record['segments_before'] = segment_summary(df, predict(df, {}))
        record['segments_selected'] = segment_summary(df, predict(df, record['selected_spec']))
        (out / 'selection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('ВЫБОР ЗАФИКСИРОВАН:', selected, flush=True)
    else:
        if (out / 'decision.json').exists():
            raise FileExistsError('Подтверждение уже завершено')
        record = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
        df = pd.read_csv(out / '2026-05-28/predictions.csv.gz')
        base = summary(df, predict(df, {}))
        chosen = summary(df, predict(df, record['selected_spec']))
        idx = score(chosen, base)
        better = sum((chosen[k] < base[k] for k in METRIC_WEIGHTS))
        promote = record['selected_name'] != 'incumbent' and idx <= 0.995 and (chosen['smape'] <= base['smape'] * 1.01) and (better >= 3)
        decision = {'selection_name': record['selected_name'], 'promote': promote, 'final_spec': record['selected_spec'] if promote else {}, 'final_name': record['selected_name'] if promote else 'incumbent', 'confirmation_before': base, 'confirmation_candidate': chosen, 'confirmation_local_index': idx, 'improved_metrics': better, 'baselines': {'mean28_round': summary(df, np.rint(df.mean_28.to_numpy(float))), 'mean90_round': summary(df, np.rint(df.mean_90.to_numpy(float))), 'zero': summary(df, np.zeros(len(df)))}, 'segments_before': segment_summary(df, predict(df, {})), 'segments_candidate': segment_summary(df, predict(df, record['selected_spec'])), 'reason': 'Кандидат подтверждён' if promote else 'Консервативно сохранён исходный ансамбль; второго выбора по контрольному окну нет'}
        (out / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
