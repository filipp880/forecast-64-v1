"""Regression checks for request-only features and imported, already fitted weights."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app.advanced_features import AdvancedHistory,augment_advanced_features,ADVANCED_COLUMNS
from app.pipeline import prepare_panel,feature_frame
from app.predictor import Predictor

ROOT=Path(__file__).resolve().parents[1]
STAGED=ROOT/'artifacts/all40-serving-20261030/model'

def inputs():
    dates=pd.date_range('2026-01-01','2026-07-26');rows=[]
    for i in range(3):
        f=pd.DataFrame({'date':dates,'quantity':np.where(np.arange(len(dates))%(i+2)==0,float(i+1),0.)})
        for c in ['warehouse_name','article','category_lvl_1','category_lvl_2','category_lvl_3','characteristic']:
            f[c]=f'{c}_{i%2}' if c in ['warehouse_name','article'] else 'category'
        f['sales_sum']=1.;f['discount_sum']=np.nan
        rows.append(f)
    sales=pd.concat(rows,ignore_index=True)
    stock=sales[['date','warehouse_name','article']].assign(stock=10.)
    return sales,stock

def initial(sales,stock):
    import json
    params=json.loads((ROOT/'model/config.json').read_text(encoding='utf-8'))['model_params']
    panel=prepare_panel(sales,stock,'research',params)
    return feature_frame(panel,pd.Timestamp('2026-07-26'),'research',model_params=params)

def test_advanced_features_ignore_future_input_rows():
    sales,stock=inputs();frame=initial(sales,stock)
    expected=augment_advanced_features(sales,stock,frame,'2026-07-26')
    future=sales.iloc[:1].assign(date=pd.Timestamp('2026-08-01'),quantity=1e6,article='future_only')
    inventory=stock.iloc[:1].assign(date=pd.Timestamp('2026-08-01'),stock=-1e6)
    actual=augment_advanced_features(pd.concat([sales,future],ignore_index=True),
        pd.concat([stock,inventory],ignore_index=True),frame,'2026-07-26')
    pd.testing.assert_frame_equal(expected,actual)
    assert set(ADVANCED_COLUMNS)<=set(actual)

def test_advanced_unknown_history_differs_from_zero():
    sales,stock=inputs();frame=initial(sales,stock)
    result=augment_advanced_features(sales,stock,frame,'2026-07-26')
    assert result.long_lag_52.isna().all()
    assert result.long_lag_14.notna().all()
    assert result.long_support_52.min()<52

@pytest.mark.skipif(not (STAGED/'model.pkl').exists(),reason='requires the locally prepared trained release')
def test_staged_prediction_repeat_subset_and_future_invariance():
    sales,stock=inputs();p=Predictor(STAGED).load()
    expected=p.predict(sales,stock,history_cutoff='2026-07-26')
    pd.testing.assert_frame_equal(expected,p.predict(sales,stock,history_cutoff='2026-07-26'))
    future=sales.iloc[:1].assign(date=pd.Timestamp('2026-08-01'),quantity=1e6)
    pd.testing.assert_frame_equal(expected,p.predict(pd.concat([sales,future],ignore_index=True),stock,
        history_cutoff='2026-07-26'))
    pairs=sales[['warehouse_name','article']].drop_duplicates().iloc[:1]
    subset=p.predict(sales,stock,history_cutoff='2026-07-26',required_pairs=pairs)
    pd.testing.assert_frame_equal(subset,expected.merge(pairs,on=['warehouse_name','article']).reset_index(drop=True))
    assert len(expected)==36 and np.isfinite(expected.quantity).all()

@pytest.mark.skipif(not (STAGED/'model.pkl').exists(),reason='requires the locally prepared trained release')
def test_staged_class_never_silently_trains():
    p=Predictor(STAGED).load()
    with pytest.raises(RuntimeError,match='verified user-trained'):p.model.fit(None,None)

@pytest.mark.skipif(not (STAGED/'model.pkl').exists(),reason='requires the locally prepared trained release')
def test_staged_http_post_and_cached_get_are_identical(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setattr(main,'predictor',Predictor(STAGED).load())
    files={name:(name+'.csv',(ROOT/'data'/filename).read_bytes()) for name,filename in
        [('sales','sample_sales_6mo.csv'),('stock','sample_stock_6mo.csv')]}
    with TestClient(main.app) as client:
        first=client.post('/predict',files=files)
        assert first.status_code==200,first.text
        assert client.get('/predict/all').content==first.content
        repeat=client.post('/predict',files=files)
        assert repeat.status_code==200 and repeat.content==first.content
