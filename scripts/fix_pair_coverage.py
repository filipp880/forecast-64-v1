"""Complete the saved submission, without training or changing existing rows."""
import hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.coverage import complete_cartesian_forecast

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    backup=ROOT/'artifacts/coverage-fix-20261028';backup.mkdir(parents=True,exist_ok=True)
    path=ROOT/'forecast.csv'
    before_hash=sha(path)
    if (backup/'forecast.before.csv').exists():raise FileExistsError('Coverage repair already started')
    shutil.copyfile(path,backup/'forecast.before.csv')
    shutil.copyfile(ROOT/'forecast.manifest.json',backup/'forecast.before.manifest.json')
    old=pd.read_csv(path,dtype={'warehouse_name':'string','article':'string'})
    sales=pd.read_csv(ROOT/'data/sales_dataset_hakaton_train.csv',usecols=['date','warehouse_name','article','quantity'],
                      parse_dates=['date'],dtype={'warehouse_name':'string','article':'string','quantity':'float64'})
    result,pairs,missing=complete_cartesian_forecast(old,sales,'2026-07-26','2026-07-27')
    if len(old)!=78993 or len(result)!=79101 or len(missing)!=12:raise ValueError('Unexpected correction dimensions')
    merged=old.merge(result,on=['week','warehouse_name','article'],validate='one_to_one',suffixes=('_old','_new'))
    if len(merged)!=len(old) or not np.array_equal(merged.quantity_old,merged.quantity_new):raise ValueError('Existing predictions changed')
    result.to_csv(path,index=False);pairs.to_csv(ROOT/'required_pairs.csv',index=False)
    missing.to_csv(backup/'added_pairs.csv',index=False)
    summary=dict(rows=len(result),pairs=len(pairs),warehouses=pairs.warehouse_name.nunique(),articles=pairs.article.nunique(),
       added_rows=len(result)-len(old),added_pairs=len(missing),existing_predictions_unchanged=True,
       model_files={n:sha(ROOT/'model'/n) for n in ('model.pkl','preprocessor.pkl','config.json')},
       forecast_sha256=sha(path),previous_forecast_sha256=before_hash,required_pairs_sha256=sha(ROOT/'required_pairs.csv'),
       forecast_weeks=9,history_cutoff='2026-07-26',first_week='2026-07-27',last_week='2026-09-21',
       coverage='full_warehouse_article_product',official_pair_coverage_verified=False,
       coverage_basis='Scorer reported 79101 expected rows, 108 missing and zero extra for prior observed-pair submission; exact external key equality awaits recheck',
       fallback='Existing no-history rule: weekly article mean across observed warehouse pairs over last 28 days, rounded',
       training_executed=False,tests_executed=False)
    for p in (ROOT/'forecast.manifest.json',backup/'repair.json'):p.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
