import numpy as np
import pandas as pd

from scripts.finalize_smape_wape_cv import WeeklyLagHistory
from app.pipeline import prepare_panel,feature_frame
from app.sequence_features import add_sequence_features
from app.features import CAT_FEATURES,KEY


def test_dense_free_training_lags_equal_serving_and_ignore_future():
    dates=pd.date_range('2025-01-01','2025-12-31')
    sales=pd.DataFrame({'date':dates,'quantity':np.arange(len(dates))%4})
    for c in CAT_FEATURES:sales[c]='x'
    newer=sales.loc[sales.date>='2025-09-01'].copy();newer['article']='new'
    sales=pd.concat([sales,newer],ignore_index=True)
    stock=sales[['date',*KEY]].assign(stock=10.)
    panel=prepare_panel(sales,stock,'research',{'recipe':{'source':'dynamics'}})
    origin='2025-10-12';base=feature_frame(panel,origin,'research')
    actual=WeeklyLagHistory(sales).add(base,origin)
    expected=add_sequence_features(panel,base,origin)
    columns=[f'weekly_lag_{i}' for i in range(1,14)]+['weekly_support_13']
    np.testing.assert_allclose(actual[columns],expected[columns],equal_nan=True)
    changed=sales.copy();changed.loc[changed.date>origin,'quantity']=999
    after=WeeklyLagHistory(changed).add(base,origin)
    np.testing.assert_allclose(actual[columns],after[columns],equal_nan=True)
