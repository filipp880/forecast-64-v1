import json
import numpy as np
import pandas as pd
import pytest
from app.ru_calendar import calendar_days,weekly_calendar
from app.enhancements import add_money,augment_features,intermittent,feature_columns
from app.features import build_panel,make_features,KEY,CAT_FEATURES
from app.models.research_model import quantize,WeeklyResearchModel
from train import save_artifact
from app.predictor import Predictor


def inputs():
    dates=pd.date_range('2024-10-01','2025-09-28')
    sales=pd.DataFrame({'date':dates,'quantity':np.where(np.arange(len(dates))%3==0,2.,0.),
                        'sales_sum':np.log(2.),'discount_sum':np.nan})
    for c in CAT_FEATURES:sales[c]='x'
    stock=sales[['date',*KEY]].assign(stock=2.)
    return sales,stock


def test_official_transfers_and_asof_unknown_year():
    a=calendar_days(['2025-11-01','2025-11-03','2026-01-09','2026-03-09','2026-05-11'],'2025-10-19')
    assert a.working.tolist()==[1,0,0,0,0]
    assert calendar_days(['2026-01-09'],'2025-08-01').working.iloc[0]==1
    assert calendar_days(['2026-01-09'],'2025-08-01').transfer_known.iloc[0]==0
    unknown=calendar_days(['2030-01-09'],'2029-10-01')
    assert unknown.transfer_known.iloc[0]==0
    assert weekly_calendar(['2025-10-27'],'2025-10-19').ru_working_saturdays.iloc[0]==1


def test_intermit_units_and_apportion_uses_fractional_remainder():
    tsb,sba=intermittent(np.array([[2]*13,[0]*13],float))
    np.testing.assert_allclose(tsb,[2,0]);np.testing.assert_allclose(sba,[1.9,0])
    f=pd.DataFrame({'warehouse_name':['w']*9,'article':['a']*9})
    values=np.array([2.1,.9]+[0]*7)
    result=quantize(values,f,{'apportion':True})
    np.testing.assert_array_equal(result,[2,1]+[0]*7)


def test_money_exp_before_sum_and_origin_invariance():
    sales,stock=inputs();past=sales.loc[sales.date<='2025-06-29']
    panel=add_money(build_panel(sales,stock),sales)
    assert panel.discount_sum.sum()==0
    assert panel.sales_sum.sum()==pytest.approx(2*len(sales))
    cut=pd.Timestamp('2025-06-29')
    a=augment_features(panel,make_features(panel,cut),cut)
    changed=sales.copy();changed.loc[changed.date>cut,['quantity','sales_sum','discount_sum']]=[999,9,8]
    p2=add_money(build_panel(changed,stock),changed)
    b=augment_features(p2,make_features(p2,cut),cut)
    pd.testing.assert_frame_equal(a,b)
    assert a.log_revenue_28.iloc[0]==pytest.approx(np.log1p(56))


def test_research_roundtrip_matches_serving_and_rejects_metadata(tmp_path):
    sales,stock=inputs();panel=add_money(build_panel(sales,stock),sales)
    x=augment_features(panel,make_features(panel,'2025-09-28'),'2025-09-28')
    recipe={'source':'tsb','post':{'round_int':True}}
    model=WeeklyResearchModel({'recipe':recipe,'threads':1})
    expected=model.predict(x)
    save_artifact(model,tmp_path,'2025-09-28','2025-09-28',0,0)
    predictor=Predictor(tmp_path).load()
    result=predictor.predict(sales,stock)
    np.testing.assert_array_equal(result.quantity,expected)
    cfg=json.loads((tmp_path/'config.json').read_text())
    cfg['enhancements']['calendar_version']='wrong'
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(Exception,match='enhancements'):Predictor(tmp_path).load()


def test_trained_calendar_blend_roundtrip_and_feature_schema(tmp_path):
    sales,stock=inputs();panel=add_money(build_panel(sales,stock),sales)
    x=augment_features(panel,make_features(panel,'2025-09-28'),'2025-09-28')
    model=WeeklyResearchModel({'recipe':{'source':'calendar','post':{'round_int':True}},'threads':1,'estimators':3})
    model.fit(x,np.arange(len(x),dtype=float),n_iterations={'calendar':{'hurdle':{'classifier':2,'regressor':2},'lgbm':2}})
    expected=model.predict(x)
    save_artifact(model,tmp_path,'2025-09-28','2025-09-28',len(x),1)
    predictor=Predictor(tmp_path).load()
    np.testing.assert_array_equal(predictor.predict(sales,stock).quantity,expected)
