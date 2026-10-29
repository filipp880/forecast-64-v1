"""Artificial cold-pair selection and aggregate censoring invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_cold_pairs import (fallback_predictions, select_holdout,
                                         source_and_holdout)


def history_grid():
    return pd.DataFrame([
        {"warehouse_name": f"w{warehouse}", "article": f"a{article}", "weekly_mean": 1. + warehouse}
        for warehouse in range(11) for article in range(30)
    ])


def test_holdout_selection_is_id_only_and_invariant_to_row_order():
    history = history_grid()
    held = select_holdout(history)
    shuffled = history.sample(frac=1, random_state=2).assign(weekly_mean=np.arange(len(history)))
    pd.testing.assert_frame_equal(held, select_holdout(shuffled))
    assert 10 < len(held) < 65
    source, validated = source_and_holdout(history, held)
    assert validated.warehouse_name.isin(source.warehouse_name).all()
    assert validated.article.isin(source.article).all()
    assert not pd.MultiIndex.from_frame(source[["warehouse_name", "article"]]).isin(pd.MultiIndex.from_frame(held)).any()


def test_held_history_is_removed_from_every_policy_aggregate():
    history = pd.DataFrame({"warehouse_name": ["w0", "w1", "w0", "w1"],
                            "article": ["a0", "a0", "a1", "a1"],
                            "weekly_mean": [10000., 2., 3., 4.]})
    held = history.iloc[:1][["warehouse_name", "article"]]
    first, support = fallback_predictions(history, held)
    changed = history.copy()
    changed.loc[0, "weekly_mean"] = 1e9
    second, _ = fallback_predictions(changed, held)
    assert first["current_article_mean"].quantity.tolist() == [2.]
    assert support["article_support_min"] == 1
    assert support["source_pairs"] == 3
    for policy in first:
        pd.testing.assert_frame_equal(first[policy], second[policy])


def test_unknown_or_unsupported_id_is_not_a_valid_holdout():
    history = pd.DataFrame({"warehouse_name": ["only", "w1", "w1"],
                            "article": ["single", "a0", "a1"], "weekly_mean": [1., 2., 3.]})
    unsupported = history.iloc[:1][["warehouse_name", "article"]]
    with pytest.raises(ValueError, match="remain known"):
        source_and_holdout(history, unsupported)
    unknown = pd.DataFrame({"warehouse_name": ["missing"], "article": ["a0"]})
    with pytest.raises(ValueError, match="origin-known"):
        source_and_holdout(history, unknown)


def test_zero_source_volume_has_finite_nonnegative_outputs():
    history = history_grid().assign(weekly_mean=0.)
    held = select_holdout(history)
    predictions, support = fallback_predictions(history, held)
    assert support["source_weekly_mean"] == 0
    for prediction in predictions.values():
        assert np.isfinite(prediction.quantity).all()
        assert prediction.quantity.eq(0).all()


def test_nonfinite_history_is_rejected_instead_of_silently_filled():
    history = history_grid()
    held = select_holdout(history)
    history.loc[0, "weekly_mean"] = float("nan")
    with pytest.raises(ValueError, match="finite nonnegative"):
        fallback_predictions(history, held)
