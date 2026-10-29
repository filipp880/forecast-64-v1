import numpy as np
import pandas as pd
import pytest
from scripts.research_regular_quantity import training_weights,route,scalar_score,ORIGINS


def test_weights_are_recency_and_past_feature_only():
    f=pd.DataFrame({'feature_origin':pd.to_datetime(['2025-01-01','2025-06-30']),
                    'mean_28':[1.,1.],'target':[2.,2.]})
    w=training_weights(f,'recent_mae','2025-06-30')
    assert w[1]>w[0]
    np.testing.assert_allclose(w.mean(),1.)
    with pytest.raises(ValueError):training_weights(f,'recent_mae','2025-01-01')


def test_scope_and_one_sided_zero_support():
    base=np.array([0.,1.,2.,3.]);raw=np.array([9.,0.,4.,10.]);weekly=np.array([2.,3.,5.,10.])
    np.testing.assert_array_equal(route(base,raw,weekly,2.,10.,1.),[0.,0.,4.,3.])
    np.testing.assert_array_equal(route(base,raw,weekly,2.,10.,.5),[0.,.5,3.,3.])


def test_metric_and_no_confirmation_origin():
    assert scalar_score([0,1,2],[0,1,2])==0
    assert '2026-05-24' not in ORIGINS
