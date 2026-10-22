"""Hurdle-модель: «будет ли продажа» × «сколько продали» (``--model-type hurdle``).

Стандартная архитектура для intermittent demand (Croston, two-part/hurdle):
отдельно моделируется вероятность ненулевой продажи и её величина, вместо
того чтобы один регрессор усреднял нули и продажи в одно число.

1. классификатор ``LGBMClassifier`` — P(quantity > 0) по всем ячейкам;
2. регрессор ``LGBMRegressor(objective="poisson")`` — величина продажи,
   обучен только на ячейках с продажами (нулевой части не «размывает» шкалу);
3. прогноз: величина, если P ≥ ``gate`` (по умолчанию 0.5 — медианное
   правило; при доле нулей > 50% условная медиана ячейки равна 0), иначе ровно 0.

Таргет — сырые счётчики (``target_transform = "none"``): классификатору
нужен бинарный факт продажи, регрессору — целочисленная величина.
Постобработка (``postprocess`` в ``config.json``: порог обнуления, округление)
применяется вызывающей стороной как обычно.

Параметры (``--model-params``; вложенные словари полностью переопределяют
соответствующий набор значений по умолчанию):

- ``gate`` — порог вероятности, ниже которого прогноз обнуляется;
- ``classifier`` / ``regressor`` — гиперпараметры соответствующей LightGBM-модели,
  например ``{"gate": 0.4, "regressor": {"n_estimators": 4000}}``.

Артефакт — один pickle-файл: оба бустера (в текстовом формате LightGBM),
кодировка категорий, число итераций после early stopping и полный ``params``
(включая ``gate``), поэтому поведение после ``save``/``load`` не меняется.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..features import CAT_FEATURES, FEATURE_COLUMNS
from .base import BaseForecastModel
from .lightgbm_model import iteration_progress_callback

try:
    import lightgbm as lgb
except ImportError:  # модель не будет доступна, пока не установлен lightgbm
    lgb = None  # type: ignore[assignment]


class HurdleForecastModel(BaseForecastModel):
    """P(продажа) x величина: классификатор по всем ячейкам + Poisson-регрессор на продажах."""

    name = "hurdle"
    file_suffix = ".pkl"
    install_hint = "pip install lightgbm"
    supports_early_stopping = True
    target_transform = "none"

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        defaults = self.default_params()
        for key in ("classifier", "regressor"):
            self.params[key] = {**defaults[key], **(self.params.get(key) or {})}
        self.classifier = None
        self.regressor = None
        self._categories: dict[str, list[str]] = {}
        self._best_iterations: dict[str, Optional[int]] = {
            "classifier": None,
            "regressor": None,
        }

    @classmethod
    def default_params(cls) -> dict:
        base = {
            "n_estimators": 2000,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 50,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 8.0,
        }
        return {
            "gate": 0.5,
            "log_period": 200,
            "classifier": dict(base),
            "regressor": {**base, "objective": "poisson"},
        }

    @classmethod
    def is_available(cls) -> bool:
        return lgb is not None

    def journal_params(self) -> dict[str, object]:
        """Колонки журнала: ``iterations``/``depth`` — «классификатор/регрессор»."""
        classifier = self.params.get("classifier", {})
        regressor = self.params.get("regressor", {})
        return {
            "iterations": f"{classifier.get('n_estimators', '')}/{regressor.get('n_estimators', '')}",
            "depth": f"{classifier.get('num_leaves', '')}/{regressor.get('num_leaves', '')}",
            "learning_rate": classifier.get("learning_rate", ""),
        }

    def _encode(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Категориальные — в pandas ``category`` с кодировкой, зафиксированной на обучении."""
        x = frame[FEATURE_COLUMNS].copy()
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

    def _common_kwargs(self, sub: dict, n_estimators: int) -> dict:
        return {
            "n_estimators": int(n_estimators),
            "num_leaves": int(sub.get("num_leaves", 63)),
            "learning_rate": float(sub.get("learning_rate", 0.05)),
            "min_child_samples": int(sub.get("min_child_samples", 50)),
            "subsample": float(sub.get("subsample", 0.9)),
            "subsample_freq": int(sub.get("subsample_freq", 1)),
            "colsample_bytree": float(sub.get("colsample_bytree", 0.9)),
            "reg_lambda": float(sub.get("reg_lambda", 8.0)),
            "random_state": int(self.params.get("seed", 42)),
            "n_jobs": int(self.params.get("threads", 8)),
            "deterministic": bool(self.params.get("deterministic", True)),
            "force_col_wise": bool(self.params.get("force_col_wise", True)),
            "verbosity": int(self.params.get("verbosity", -1)),
        }

    @staticmethod
    def _as_best(value) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        return value if value >= 0 else None

    def _resolve_iterations(self, n_iterations) -> tuple[int, int]:
        """Числа итераций для (classifier, regressor) из ``n_iterations`` или ``params``.

        ``n_iterations`` — ``None`` (параметры), число (оба) или словарь
        ``{"classifier": …, "regressor": …}``, который вернул
        :attr:`selected_iterations` после early stopping.
        """
        classifier = self.params.get("classifier", {})
        regressor = self.params.get("regressor", {})
        default_clf = int(classifier.get("n_estimators", 2000))
        default_reg = int(regressor.get("n_estimators", 2000))
        if n_iterations is None:
            return default_clf, default_reg
        if isinstance(n_iterations, dict):
            classifier = n_iterations.get("classifier")
            regressor = n_iterations.get("regressor")
            return (
                int(classifier) if classifier is not None else default_clf,
                int(regressor) if regressor is not None else default_reg,
            )
        return int(n_iterations), int(n_iterations)

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.DataFrame] = None,
        early_stopping_rounds: Optional[int] = None,
        n_iterations=None,
    ) -> "HurdleForecastModel":
        if lgb is None:
            raise RuntimeError(self.install_hint)
        classifier_iterations, regressor_iterations = self._resolve_iterations(n_iterations)

        x = self._encode(x_train, fit=True)
        y = np.asarray(y_train, dtype="float64")
        labels = (y > 0).astype(int)
        positive = labels == 1
        if not positive.any():
            raise RuntimeError("В обучающей выборке нет положительных продаж — hurdle не обучится")

        use_early_stopping = bool(
            early_stopping_rounds and x_valid is not None and y_valid is not None
        )
        if use_early_stopping:
            valid_x = self._encode(x_valid, fit=False)
            valid_y = np.asarray(y_valid, dtype="float64")
            valid_labels = (valid_y > 0).astype(int)
        else:
            valid_x, valid_y, valid_labels = None, None, None
        log_period = int(self.params.get("log_period", 200))

        print(
            f"  hurdle, стадия 1/2: классификатор P(продажа > 0), "
            f"{len(x):,} строк, лимит {classifier_iterations} итераций",
            flush=True,
        )
        classifier = lgb.LGBMClassifier(
            **self._common_kwargs(self.params.get("classifier", {}), classifier_iterations)
        )
        progress = iteration_progress_callback(
            log_period, classifier_iterations, label="классификатор"
        )
        if use_early_stopping:
            callbacks = [
                lgb.early_stopping(int(early_stopping_rounds), verbose=False),
                *([lgb.log_evaluation(period=log_period)] if log_period else []),
            ]
            if progress:
                callbacks.append(progress)
            classifier.fit(
                x,
                labels,
                eval_X=valid_x,
                eval_y=valid_labels,
                eval_metric="binary_logloss",
                categorical_feature=CAT_FEATURES,
                callbacks=callbacks,
            )
            classifier_best = self._as_best(getattr(classifier, "best_iteration_", None))
        else:
            classifier.fit(
                x,
                labels,
                categorical_feature=CAT_FEATURES,
                callbacks=[progress] if progress else None,
            )
            classifier_best = None

        regressor_params = self.params.get("regressor", {})
        regressor_kwargs = self._common_kwargs(regressor_params, regressor_iterations)
        regressor_kwargs["objective"] = str(regressor_params.get("objective", "poisson"))
        print(
            f"  hurdle, стадия 2/2: регрессор величины продажи ({regressor_kwargs['objective']}), "
            f"{int(positive.sum()):,} строк с продажами, лимит {regressor_iterations} итераций",
            flush=True,
        )
        regressor = lgb.LGBMRegressor(**regressor_kwargs)
        progress = iteration_progress_callback(log_period, regressor_iterations, label="регрессор")
        positive_valid = valid_labels == 1 if use_early_stopping else None
        if use_early_stopping and positive_valid.any():
            callbacks = [
                lgb.early_stopping(int(early_stopping_rounds), verbose=False),
                *([lgb.log_evaluation(period=log_period)] if log_period else []),
            ]
            if progress:
                callbacks.append(progress)
            regressor.fit(
                x[positive],
                y[positive],
                eval_X=valid_x[positive_valid],
                eval_y=valid_y[positive_valid],
                eval_metric="poisson",
                categorical_feature=CAT_FEATURES,
                callbacks=callbacks,
            )
            regressor_best = self._as_best(getattr(regressor, "best_iteration_", None))
        else:
            regressor.fit(
                x[positive],
                y[positive],
                categorical_feature=CAT_FEATURES,
                callbacks=[progress] if progress else None,
            )
            regressor_best = None

        self.classifier = classifier.booster_
        self.regressor = regressor.booster_
        self._best_iterations = {
            "classifier": classifier_best,
            "regressor": regressor_best,
        }
        return self

    @property
    def selected_iterations(self) -> Optional[dict]:
        """Числа итераций обеих моделей после early stopping (``None`` — без него).

        Возвращает словарь, который :meth:`fit` принимает как ``n_iterations``
        при финальном refit'е.
        """
        classifier = self._best_iterations.get("classifier")
        regressor = self._best_iterations.get("regressor")
        if classifier is None and regressor is None:
            return None
        return {"classifier": classifier, "regressor": regressor}

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Модель не обучена — сначала вызовите fit()")
        x = self._encode(features, fit=False)
        probability = self.classifier.predict(x, num_iteration=self._best_iterations["classifier"], num_threads=int(self.params.get("threads", 8)))
        magnitude = self.regressor.predict(x, num_iteration=self._best_iterations["regressor"], num_threads=int(self.params.get("threads", 8)))
        gate = float(self.params.get("gate", 0.5))
        return np.where(np.asarray(probability) >= gate, magnitude, 0.0).astype("float64")

    def save(self, path: Path) -> None:
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Нечего сохранять — модель не обучена")
        payload = {
            "classifier": self.classifier.model_to_string(),
            "regressor": self.regressor.model_to_string(),
            "categories": self._categories,
            "best_iterations": self._best_iterations,
            "params": self.params,
        }
        Path(path).write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "HurdleForecastModel":
        if lgb is None:
            raise RuntimeError(cls.install_hint)
        payload = pickle.loads(Path(path).read_bytes())
        model = cls(payload.get("params"))
        model.classifier = lgb.Booster(model_str=payload["classifier"])
        model.regressor = lgb.Booster(model_str=payload["regressor"])
        model._categories = {
            col: [str(value) for value in values]
            for col, values in payload["categories"].items()
        }
        model._best_iterations = payload.get("best_iterations") or {
            "classifier": None,
            "regressor": None,
        }
        return model
