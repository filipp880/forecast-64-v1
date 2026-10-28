"""Модель на CatBoost — альтернативный тип (``--model-type catboost``).

CatBoost принимает категориальные признаки строками нативно, поэтому
кодирование не нужно: признаки передаются как есть из ``app/features.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..features import CAT_FEATURES, FEATURE_COLUMNS
from .base import BaseForecastModel

try:
    from catboost import CatBoostRegressor
except ImportError:  # модель не будет доступна, пока не установлен catboost
    CatBoostRegressor = None  # type: ignore[assignment,misc]


class CatBoostForecastModel(BaseForecastModel):
    """Регрессор на ``log1p(quantity)``; early stopping по окну отбора, refit — по ``n_iterations``."""

    name = "catboost"
    file_suffix = ".cbm"
    install_hint = "pip install catboost"
    supports_early_stopping = True

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self.model = None

    @classmethod
    def default_params(cls) -> dict:
        return {
            "iterations": 2000,
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 8.0,
            "verbose": 200,
        }

    @classmethod
    def is_available(cls) -> bool:
        return CatBoostRegressor is not None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        n_iterations: Optional[int] = None,
    ) -> "CatBoostForecastModel":
        if CatBoostRegressor is None:
            raise RuntimeError(self.install_hint)
        iterations = (
            int(n_iterations) if n_iterations is not None else int(self.params["iterations"])
        )
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=int(self.params["depth"]),
            learning_rate=float(self.params["learning_rate"]),
            l2_leaf_reg=float(self.params["l2_leaf_reg"]),
            random_seed=int(self.params.get("seed", 42)),
            thread_count=int(self.params.get("threads", 8)),
            cat_features=CAT_FEATURES,
            allow_writing_files=False,
            verbose=int(self.params.get("verbose", 200)),
        )
        x = x_train[FEATURE_COLUMNS]
        if n_iterations is not None:
            self.model.fit(x, y_train)
        elif x_valid is not None and y_valid is not None:
            self.model.fit(
                x,
                y_train,
                eval_set=(x_valid[FEATURE_COLUMNS], y_valid),
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True,
            )
        else:
            self.model.fit(x, y_train)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена — сначала вызовите fit()")
        return np.asarray(self.model.predict(features[FEATURE_COLUMNS]), dtype="float64")

    @property
    def selected_iterations(self) -> Optional[int]:
        """Число деревьев модели после early stopping (для refit'а)."""
        if self.model is None:
            return None
        return int(self.model.tree_count_)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Нечего сохранять — модель не обучена")
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "CatBoostForecastModel":
        if CatBoostRegressor is None:
            raise RuntimeError(cls.install_hint)
        model = cls()
        model.model = CatBoostRegressor()
        model.model.load_model(str(path))
        return model
