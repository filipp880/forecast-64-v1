"""Модель на LightGBM — тип по умолчанию (``--model-type lgbm``).

LightGBM обучается на CPU заметно быстрее CatBoost. Категориальные признаки
передаются нативно: строки кодируются через pandas ``category``, причём
соответствие «строка ↔ код» сохраняется в самом файле модели
(``pandas_categorical``) и восстанавливается при загрузке — прогноз после
``save``/``load`` совпадает с прогнозом сразу после обучения.

Детерминизм: фиксированный ``seed`` и ``threads`` плюс ``deterministic`` и
``force_col_wise`` — повторный запуск на тех же данных даёт тот же прогноз.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..features import CAT_FEATURES, FEATURE_COLUMNS
from .base import TARGET_TRANSFORMS, BaseForecastModel, format_duration

try:
    import lightgbm as lgb
except ImportError:  # модель не будет доступна, пока не установлен lightgbm
    lgb = None  # type: ignore[assignment]

EXTRA_LGBM_PARAMS = ("objective", "tweedie_variance_power", "alpha")


def iteration_progress_callback(period: int, total: int, label: str = ""):
    """Callback LightGBM: каждые ``period`` итераций печатает прогресс и оценку остатка.

    Даёт «ноутбучный» прогресс в терминале при длинном обучении. ``total`` —
    запланированное число итераций; при срабатывании early stopping реальное
    число меньше, поэтому оценка остатка — верхняя граница.
    """
    if not period or period <= 0:
        return None
    started = time.time()

    def callback(env) -> None:
        done = int(env.iteration) + 1
        if done % period:
            return
        elapsed = time.time() - started
        eta = elapsed / done * max(total - done, 0)
        prefix = f"{label}: " if label else ""
        print(
            f"    {prefix}итерация {done}/{total} (прошло {format_duration(elapsed)}, "
            f"осталось ~{format_duration(eta)})",
            flush=True,
        )

    return callback


class LightGBMForecastModel(BaseForecastModel):
    """Регрессор на LightGBM; early stopping по окну отбора, refit — по ``n_iterations``.

    По умолчанию таргет — ``log1p(quantity)``, но ``target_transform`` можно
    переопределить в ``params`` (``"none"`` + ``objective=poisson/tweedie`` —
    модель на сырых счётчиках); ``objective``, ``tweedie_variance_power`` и
    ``alpha`` из ``params`` пробрасываются в ``LGBMRegressor``.
    """

    name = "lgbm"
    file_suffix = ".txt"
    install_hint = "pip install lightgbm"
    supports_early_stopping = True
    JOURNAL_PARAM_KEYS = {
        "iterations": "n_estimators",
        "depth": "num_leaves",
        "learning_rate": "learning_rate",
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        transform = str(self.params.get("target_transform", "log1p"))
        if transform not in TARGET_TRANSFORMS:
            raise ValueError(
                f"Неизвестное преобразование таргета: {transform!r} (есть: {TARGET_TRANSFORMS})"
            )
        self.target_transform = transform
        self.booster = None
        self._categories: dict[str, list[str]] = {}
        self._best_iteration: Optional[int] = None

    @classmethod
    def default_params(cls) -> dict:
        return {
            "n_estimators": 3000,
            "num_leaves": 127,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 8.0,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "log_period": 200,
        }

    @classmethod
    def is_available(cls) -> bool:
        return lgb is not None

    def _encode(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Категориальные — в pandas ``category`` с кодировкой, зафиксированной на обучении."""
        x = frame[self.params.get("feature_columns", FEATURE_COLUMNS)].copy()
        for col in CAT_FEATURES:
            values = x[col].astype(str)
            if fit:
                self._categories[col] = sorted(values.unique())
            categories = self._categories.get(col)
            if categories is None:
                raise RuntimeError(
                    f"Нет кодировки категорий для {col!r} — модель загружена некорректно"
                )
            x[col] = pd.Categorical(values, categories=categories)
        return x

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.DataFrame] = None,
        early_stopping_rounds: Optional[int] = None,
        n_iterations: Optional[int] = None,
    ) -> "LightGBMForecastModel":
        if lgb is None:
            raise RuntimeError(self.install_hint)
        iterations = (
            int(n_iterations) if n_iterations is not None else int(self.params["n_estimators"])
        )
        x = self._encode(x_train, fit=True)
        extra = {key: self.params[key] for key in EXTRA_LGBM_PARAMS if key in self.params}
        model = lgb.LGBMRegressor(
            n_estimators=iterations,
            num_leaves=int(self.params["num_leaves"]),
            learning_rate=float(self.params["learning_rate"]),
            min_child_samples=int(self.params["min_child_samples"]),
            subsample=float(self.params["subsample"]),
            subsample_freq=int(self.params["subsample_freq"]),
            colsample_bytree=float(self.params["colsample_bytree"]),
            reg_lambda=float(self.params["reg_lambda"]),
            random_state=int(self.params.get("seed", 42)),
            n_jobs=int(self.params.get("threads", 8)),
            deterministic=bool(self.params.get("deterministic", True)),
            force_col_wise=bool(self.params.get("force_col_wise", True)),
            verbosity=int(self.params.get("verbosity", -1)),
            **extra,
        )
        log_period = int(self.params.get("log_period", 200))
        progress = iteration_progress_callback(log_period, iterations)
        if n_iterations is not None or x_valid is None or y_valid is None:
            model.fit(
                x,
                y_train,
                categorical_feature=CAT_FEATURES,
                callbacks=[progress] if progress else None,
            )
        else:
            callbacks = []
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
            if log_period:
                callbacks.append(lgb.log_evaluation(period=log_period))
            if progress:
                callbacks.append(progress)
            model.fit(
                x,
                y_train,
                eval_X=self._encode(x_valid, fit=False),
                eval_y=y_valid,
                eval_metric=str(self.params.get("eval_metric", "rmse")),
                categorical_feature=CAT_FEATURES,
                callbacks=callbacks,
            )
        self.booster = model.booster_
        best = getattr(model, "best_iteration_", None)
        best = int(best) if best is not None else -1
        self._best_iteration = best if best >= 0 else None
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Модель не обучена — сначала вызовите fit()")
        x = self._encode(features, fit=False)
        return np.asarray(
            self.booster.predict(x, num_iteration=self._best_iteration, num_threads=int(self.params.get("threads", 8))), dtype="float64"
        )

    @property
    def selected_iterations(self) -> Optional[int]:
        """Число деревьев лучшей модели после early stopping (для refit'а)."""
        return self._best_iteration

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("Нечего сохранять — модель не обучена")
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "LightGBMForecastModel":
        if lgb is None:
            raise RuntimeError(cls.install_hint)
        booster = lgb.Booster(model_file=str(path))
        categories = booster.pandas_categorical
        if not categories or len(categories) != len(CAT_FEATURES):
            raise RuntimeError(
                "В файле модели нет метаданных категориальных признаков (pandas_categorical)"
            )
        model = cls()
        model.booster = booster
        model._categories = {
            col: [str(value) for value in values]
            for col, values in zip(CAT_FEATURES, categories)
        }
        return model
