"""Compare the four submitted files and transcribed, rounded external results. No training."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_joint_release import OUT, read, write, sha
from app.submission import validate_forecast


def main():
    # External values are manually transcribed from the user's screenshot, not a scorer export.
    records = [
        dict(submission='daily_candidate_forecast', file='artifacts/daily-weekloss-7c8fdc7c3173/daily_candidate_forecast.csv',
             score=.409677, raw=[61.555036,49.294426,4.163777,88.571413,.546504], ranks_minus_one=[12,9,21,11,13]),
        dict(submission='joint_candidate_forecast', file='artifacts/daily-weekloss-review-20261030/joint_candidate_forecast.csv',
             score=.370968, raw=[61.556648,49.142695,4.157557,87.498300,.544662], ranks_minus_one=[13,5,19,7,9]),
        dict(submission='weekly_forecast_v2', file='artifacts/forecasts-to-check-20261030/weekly_forecast_v2.csv',
             score=.506452, raw=[61.632862,49.608340,4.200819,87.802409,.546065], ranks_minus_one=[15,15,24,10,12]),
        dict(submission='joint_forecast_v2', file='artifacts/forecasts-to-check-20261030/joint_forecast_v2.csv',
             score=.403226, raw=[61.557095,49.211246,4.154562,87.672584,.544563], ranks_minus_one=[14,8,18,8,8]),
    ]
    weights=np.array([.5,.2,.15,.1,.05]);names=['smape','wape','rmse','bias','rmsle']
    keys=['week','warehouse_name','article'];reference=None
    for record in records:
        path=ROOT/record['file'];frame=pd.read_csv(path)
        summary=validate_forecast(frame,pd.read_csv(ROOT/'required_pairs.csv'),'2026-07-27')
        record['sha256']=sha(path);record['csv_validation']=summary
        record['score_recomputed_from_screenshot_ranks']=float(weights@np.array(record['ranks_minus_one'])/31)
        assert abs(record['score_recomputed_from_screenshot_ranks']-record['score'])<.00000051
        frame=frame.sort_values(keys).reset_index(drop=True)
        if reference is None:reference=frame[keys]
        pd.testing.assert_frame_equal(reference,frame[keys])
        record['raw_metrics']=dict(zip(names,record.pop('raw')))
    best=records[1];new=records[3]
    delta=np.array(new['ranks_minus_one'])-np.array(best['ranks_minus_one'])
    contributions=weights*delta/31
    a=pd.read_csv(ROOT/best['file']).sort_values(keys).reset_index(drop=True)
    b=pd.read_csv(ROOT/new['file']).sort_values(keys).reset_index(drop=True)
    difference=b.quantity.to_numpy(float)-a.quantity.to_numpy(float)
    old_local=read(ROOT/'artifacts/daily-weekloss-review-20261030/joint_candidate_manifest.json')
    new_local=read(ROOT/'artifacts/weekly-refinement-review-20261030/verification.json')['best_joint']
    old_means={k:float(np.mean([m[k] for m in old_local['historical_metrics'].values()])) for k in names}
    new_means={k:float(np.mean([m[k] for m in new_local['metrics'].values()])) for k in names}
    result=dict(source='user screenshot, 32 submissions, rounded metrics; not an official machine-readable export',
        hidden_truth_available=False, external_file_hash_confirmed_by_judges=False,
        metric_order=names, metric_weights=weights.tolist(), submitted_files=records,
        distinct_file_contents=len({r['sha256'] for r in records})==4,
        local_new_minus_old={k:new_means[k]-old_means[k] for k in names},
        external_new_minus_old={k:new['raw_metrics'][k]-best['raw_metrics'][k] for k in names},
        rank_score_delta_contributions=dict(zip(names,contributions.tolist())),
        rank_score_new_minus_old=float(sum(contributions)),
        final_prediction_difference=dict(changed_rows=int(np.count_nonzero(difference)),
            max_absolute_difference=float(np.abs(difference).max()),
            mean_absolute_difference=float(np.abs(difference).mean()),total_difference=float(difference.sum())))
    write(OUT/'submission_gap.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='submitted_files'},indent=2))


if __name__=='__main__':main()
