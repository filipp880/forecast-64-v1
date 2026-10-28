import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import research_weekly as research
from app.metrics import smape,wape,rmse,bias_abs,rmsle


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    code=tmp_path/'features.py';code.write_text('# frozen\n')
    data=tmp_path/'data.csv';data.write_text('quantity\n1\n')
    out=tmp_path/'run';out.mkdir()
    research.write(out/'candidate_grid.json',{'a':{'source':'base'}})
    monkeypatch.setattr(research,'ROOT',tmp_path)
    monkeypatch.setattr(research,'source_paths',lambda:sorted(tmp_path.glob('*.py')))
    protocol={'candidate_grid_sha256':research.sha(out/'candidate_grid.json'),
              'input_hashes':{'data.csv':research.sha(data)},'source_hashes':{'features.py':research.sha(code)}}
    research.write(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(research.sha(out/'protocol.json')+'\n')
    assert research.verify_protocol(out)==protocol
    return tmp_path,out


@pytest.mark.parametrize('mutation',['code','added_code','data','grid','protocol'])
def test_changed_frozen_research_rejects_resume_before_fit(frozen,mutation):
    root,out=frozen
    paths={'code':root/'features.py','added_code':root/'new.py','data':root/'data.csv',
           'grid':out/'candidate_grid.json','protocol':out/'protocol.json'}
    if mutation=='protocol':
        obj=research.read(paths[mutation]);obj['unexpected']=True;research.write(paths[mutation],obj)
    else:paths[mutation].write_text('changed')
    with pytest.raises(ValueError,match='changed'):research.verify_protocol(out)


def test_fast_search_metrics_match_public_metrics_for_unequal_series():
    actual=np.array([0,2,0,2,0.]);pred=np.array([0,1,2,3,0.]);codes=np.array([0,0,1,1,1])
    result=research.fast_metrics(actual,pred,codes)
    reference={'smape':smape(actual,pred,codes),'wape':wape(actual,pred),'rmse':rmse(actual,pred),
               'bias':bias_abs(actual,pred,codes),'rmsle':rmsle(actual,pred)}
    for k,v in reference.items():assert result[k]==pytest.approx(v)
