"""Проверка инженерной совместимости window против legacy; не подбор по score."""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import build_panel, make_features, KEY
from app.predictor import Predictor
from app.calibration import predict_calibrated
from verified_backtest import metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'artifacts/verified')
    ap.add_argument('--origin', required=True)
    a = ap.parse_args()
    cut = pd.Timestamp(a.origin)
    root = Path(__file__).resolve().parents[2]
    s = pd.read_csv(root / 'data/sales_dataset_hakaton_train.csv', parse_dates=['date'])
    t = pd.read_csv(root / 'data/stock_dataset_hakaton_train.csv', parse_dates=['date'])
    daily = s.groupby(['date', *KEY], as_index=False).quantity.sum()
    panel = build_panel(s[s.date <= cut], t[t.date <= cut])
    m = Predictor(a.out / a.origin).load().model
    old = make_features(panel, cut, range(1, 65), stock_policy='legacy')
    new = make_features(panel, cut, range(1, 65), stock_policy='window')
    assert old[['date', *KEY]].equals(new[['date', *KEY]])
    p1 = np.rint(m.predict(old))
    p2 = np.rint(m.predict(new))
    truth = new[['date', *KEY]].merge(daily, on=['date', *KEY], how='left', validate='one_to_one').quantity.fillna(0.0).to_numpy(float)
    ids = pd.MultiIndex.from_frame(new[KEY]).to_numpy()
    r = {'origin': a.origin, 'equal_predictions': bool(np.array_equal(p1, p2)), 'changed_rows': int((p1 != p2).sum()), 'legacy': metrics(truth, p1, ids), 'window': metrics(truth, p2, ids), 'changed_features': {col: int((~(old[col].eq(new[col]) | old[col].isna() & new[col].isna())).sum()) for col in ['stock_last', 'stock_mean_28', 'stock_zero_share_90', 'days_of_cover']}}
    (a.out / a.origin / 'inventory_contract.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
    print(json.dumps(r, indent=2))
if __name__ == '__main__':
    main()
