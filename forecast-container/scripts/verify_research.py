"""Production-bundle smoke checks on shifted sample dates, without retraining."""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

CONTAINER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CONTAINER))
from app.predictor import Predictor
from app.features import KEY,next_week_start
from app.submission import validate_forecast
from app.validators import validate_sales_csv,validate_stock_csv


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',type=Path,default=CONTAINER/'model')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv)
    sales=pd.read_csv(CONTAINER/'data/sample_sales_6mo.csv',parse_dates=['date'])
    stock=pd.read_csv(CONTAINER/'data/sample_stock_6mo.csv',parse_dates=['date'])
    calls=[]
    def denied(*a,**kw):calls.append('network');raise AssertionError('Network during inference')
    scenarios=[]
    with patch.object(socket,'create_connection',denied),patch.object(socket.socket,'connect',denied),\
         patch.object(socket.socket,'connect_ex',denied),patch.object(socket,'getaddrinfo',denied):
        predictor=Predictor(args.model_dir).load()
        for last in pd.date_range('2026-07-27',periods=7):
            shift=last-sales.date.max()
            s=sales.assign(date=sales.date+shift);t=stock.assign(date=stock.date+shift)
            assert not validate_sales_csv(s) and not validate_stock_csv(t)
            f=predictor.predict(s,t)
            validate_forecast(f,s[KEY].drop_duplicates(),next_week_start(last))
            pd.testing.assert_frame_equal(f,predictor.predict(s,t))
            future=t.iloc[-1:].assign(date=last+pd.Timedelta(days=1),stock=123456.)
            pd.testing.assert_frame_equal(f,predictor.predict(s,pd.concat([t,future],ignore_index=True)))
            scenarios.append(dict(last_date=str(last.date()),first_week=f.week.min(),rows=len(f),deterministic=True,future_stock_ignored=True))
        required=sales[KEY].drop_duplicates().astype(str)
        unknown=pd.DataFrame([dict(warehouse_name='unknown_warehouse',article='unknown_article')])
        required=pd.concat([required,unknown],ignore_index=True)
        f=predictor.predict(sales,stock,required_pairs=required)
        validate_forecast(f,required,next_week_start(sales.date.max()))
        missing=f.loc[(f.warehouse_name=='unknown_warehouse')&(f.article=='unknown_article')]
        assert len(missing)==9 and np.isfinite(missing.quantity).all() and (missing.quantity>=0).all()
    record=dict(scope='Synthetic shifts of sample history; alignment/contract checks, not historical quality or independent folds',
                scenarios=scenarios,python_network_calls=len(calls),network_guard_passed=True,
                unknown_pair_fallback_passed=True,production_model_sha256=hashlib.sha256((args.model_dir/predictor.config['model_file']).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
