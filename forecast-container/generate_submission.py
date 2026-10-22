"""Генерация и строгая проверка сдаваемого CSV; старые файлы не перезаписываются."""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
import pandas as pd
from app.predictor import Predictor
from app.validators import validate_sales_csv, validate_stock_csv
from app.submission import validate_forecast

def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument('--sales', type=Path, default=root / 'data/sales_dataset_hakaton_train.csv')
    ap.add_argument('--stock', type=Path, default=root / 'data/stock_dataset_hakaton_train.csv')
    ap.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent / 'model')
    ap.add_argument('--out', type=Path, default=root / 'submission/forecast.csv')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if a.out.exists() and (not a.overwrite):
        raise FileExistsError(f'Файл уже существует: {a.out}')
    start = time.perf_counter()
    sales = pd.read_csv(a.sales, dtype={'warehouse_name': 'string', 'article': 'string'})
    stock = pd.read_csv(a.stock, dtype={'warehouse_name': 'string', 'article': 'string', 'char_article': 'string'})
    errors = validate_sales_csv(sales) + validate_stock_csv(stock)
    if errors:
        raise ValueError('; '.join(errors))
    predictor = Predictor(a.model_dir).load()
    predict_start = time.perf_counter()
    f = predictor.predict(sales, stock)
    predict_seconds = time.perf_counter() - predict_start
    summary = validate_forecast(f, sales)
    repeat_start = time.perf_counter()
    if not f.equals(predictor.predict(sales, stock)):
        raise ValueError('Нарушен детерминизм')
    repeat_seconds = time.perf_counter() - repeat_start
    a.out.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(a.out, index=False)
    summary['predict_seconds'] = predict_seconds
    summary['repeat_seconds'] = repeat_seconds
    try:
        import psutil
        mem = psutil.Process().memory_info()
        summary['peak_process_gib'] = getattr(mem, 'peak_wset', mem.rss) / 2**30
        summary['memory_kind'] = 'peak_working_set' if hasattr(mem, 'peak_wset') else 'current_rss'
    except ImportError:
        summary['peak_process_gib'] = None
    summary['seconds_two_predictions'] = time.perf_counter() - start
    paths = [a.sales, a.stock, a.out, *a.model_dir.glob('*')]
    summary['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    summary['model_config'] = predictor.config
    summary['coverage_note'] = 'Соблюдены все наблюдаемые пары sales. Число полной декартовой сетки из ТЗ требует отдельного уточнения; пары не выдумываются.'
    a.out.with_suffix('.manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
