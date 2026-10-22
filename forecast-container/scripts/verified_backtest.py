"""Два независимых 64-дневных окна: выбор на первом, контроль на втором."""
from __future__ import annotations
import argparse, json, sys, time, gc, hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd, psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import build_panel, make_features, make_training_frame, KEY, FEATURE_COLUMNS, preprocessing_spec
from app.models import get_model_class
from app.metrics import smape, wape, rmse, bias_abs, rmsle
from train import make_cutoffs, build_dataset

def metrics(y, p, ids):
    return dict(smape=smape(y, p, ids), wape=wape(y, p), rmse=rmse(y, p), bias=bias_abs(y, p, ids), rmsle=rmsle(y, p), zero_share=float((p == 0).mean()), volume_bias=float((p.sum() / y.sum() - 1) * 100) if y.sum() > 0 else None)

def peak_memory_gib():
    if sys.platform == 'win32':
        return psutil.Process().memory_info().peak_wset / 2 ** 30
    import resource
    scale = 2 ** 30 if sys.platform == 'darwin' else 2 ** 20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sales', default='../data/sales_dataset_hakaton_train.csv')
    ap.add_argument('--stock', default='../data/stock_dataset_hakaton_train.csv')
    ap.add_argument('--out', default='../artifacts/verified')
    ap.add_argument('--max-cutoffs', type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from select_candidate import initialize_grid
    initialize_grid(out)
    protocol = {'origins': ['2026-03-25', '2026-05-28'], 'horizon': 64, 'selection': 'first_origin_only', 'confirmation': 'second_origin_only', 'surrogate': 'sum of official weights * metric / incumbent_metric; LOCAL, NOT OFFICIAL', 'promotion': 'selection index <=0.995, confirmation index <=0.995; neither origin SMAPE > incumbent*1.01; never tune on confirmation', 'seed': 42, 'threads': 8, 'max_cutoffs': args.max_cutoffs}
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text(encoding='utf-8')) != protocol:
        raise ValueError('Этот каталог содержит другой протокол; создайте новый out')
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Протокол зафиксирован:', protocol, flush=True)
    sales = pd.read_csv(args.sales, parse_dates=['date'])
    stock = pd.read_csv(args.stock, parse_dates=['date'])
    daily = sales.groupby(['date', *KEY], as_index=False).quantity.sum()
    for cut in pd.to_datetime(protocol['origins']):
        fold = out / cut.strftime('%Y-%m-%d')
        fold.mkdir(exist_ok=True)
        if (fold / 'complete.json').exists():
            print('Готовое окно сохранено:', fold, flush=True)
            continue
        started = time.perf_counter()
        panel = build_panel(sales[sales.date <= cut], stock[stock.date <= cut])
        gc.collect()
        select = cut - pd.Timedelta(days=64)
        cutoffs = make_cutoffs(panel, 180, 64, 28)
        eligible = [c for c in cutoffs if c + pd.Timedelta(days=64) <= select]
        if args.max_cutoffs:
            eligible = eligible[-args.max_cutoffs:]
        frames = build_dataset(panel, eligible, 64, 180, stock_policy='legacy')
        train = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()
        assert train.date.max() <= select
        valid = make_training_frame(panel, select, range(1, 65), stock_policy='legacy')
        assert valid.date.max() <= cut and valid.date.min() > train.date.max()
        m = get_model_class('blend')({'weight_hurdle': 0.3, 'seed': 42, 'threads': 8})
        print('Отбор итераций', cut.date(), len(train), flush=True)
        m.fit(train, train.target.to_numpy(float), valid, valid.target.to_numpy(float), 100)
        iterations = m.selected_iterations
        del train, valid
        gc.collect()
        refits = cutoffs[-args.max_cutoffs:] if args.max_cutoffs else cutoffs
        frames = build_dataset(panel, refits, 64, 180, stock_policy='legacy')
        refit = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()
        assert refit.date.max() <= cut
        print('Refit только прошлых фактов', cut.date(), len(refit), iterations, flush=True)
        m.fit(refit, refit.target.to_numpy(float), n_iterations=iterations)
        del refit
        gc.collect()
        m.save(fold / 'model.pkl')
        config = {'model_type': 'blend', 'model_file': 'model.pkl', 'preprocessor_file': 'preprocessor.pkl', 'horizon': 64, 'lookback_days': 180, 'stock_policy': 'legacy', 'target_transform': 'none', 'postprocess': {'zero_threshold': 0, 'round_int': True}, 'training_data_end': str(cut.date()), 'random_seed': 42, 'thread_count': 8, 'feature_columns': FEATURE_COLUMNS, 'model_params': m.params, 'validation_protocol': str(out / 'protocol.json')}
        (fold / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (fold / 'preprocessor.pkl').write_bytes(pickle.dumps(preprocessing_spec(stock_policy='legacy')))
        f = make_features(panel, cut, range(1, 65), stock_policy='legacy')
        encoded = m.hurdle._encode(f, False)
        p = m.hurdle.classifier.predict(encoded)
        mag = m.hurdle.regressor.predict(encoded)
        lg = np.clip(np.expm1(m.lgbm.predict(f)), 0, None)
        del encoded
        gc.collect()
        result = f[['date', *KEY, 'h', 'mean_28', 'mean_90', 'n_sale_days_90', 'stock_last', 'category_lvl_1']].copy()
        result['origin'] = cut.strftime('%Y-%m-%d')
        result['probability'] = p
        result['magnitude'] = mag
        result['lgbm_raw'] = lg
        result = result.merge(daily, on=['date', *KEY], how='left', validate='one_to_one')
        result['actual'] = result.pop('quantity').fillna(0.0)
        result.to_csv(fold / 'predictions.csv.gz', index=False, compression='gzip')
        base = np.rint(0.3 * np.where(p >= 0.5, mag, 0) + 0.7 * lg)
        ids = pd.MultiIndex.from_frame(result[KEY]).to_numpy()
        summary = {'origin': str(cut.date()), 'evaluation_kind': 'out_of_sample', 'rows': len(result), 'pairs': len(result[KEY].drop_duplicates()), 'iterations': iterations, 'seconds': time.perf_counter() - started, 'peak_gib': peak_memory_gib(), 'incumbent': metrics(result.actual.to_numpy(float), base, ids), 'sha256_model': hashlib.sha256((fold / 'model.pkl').read_bytes()).hexdigest(), 'data_cutoff_assertions': True}
        (fold / 'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('ОКНО ГОТОВО', json.dumps(summary), flush=True)
        del panel, f, result, m
        gc.collect()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
