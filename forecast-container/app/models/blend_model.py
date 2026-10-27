"""Ансамбль «hurdle + lgbm» — тип ``--model-type blend`` (финальная модель).

Взвешенная смесь двух моделей реестра:

1. ``hurdle`` — Poisson-величина при P(продажи за неделю > 0) ≥ gate, иначе ноль;
2. ``lgbm`` — одиночный регрессор на ``log1p(quantity)``.

Обе учатся на одних и тех же срезах; прогнозы возвращаются в исходном
пространстве величины (у ансамбля ``target_transform = "none"``) и смешиваются
с весом ``weight_hurdle`` (по умолчанию 0.3):

    прогноз = w × hurdle + (1 − w) × lgbm

Вес, гейт и округление проверяются на недельных временных окнах.
Исторические дневные сравнения не определяют качество недельной модели.
Постобработка (``postprocess`` в ``config.json``) применяется вызывающей
стороной как обычно, уже после смешивания.

Параметры (``--model-params``; вложенные словари — переопределения параметров
соответствующей модели реестра)::

    {"weight_hurdle": 0.3,
     "hurdle": {"classifier": {"n_estimators": 2000}},
     "lgbm": {"n_estimators": 3000}}

Артефакт — один pickle: оба бустера hurdle и бустер lgbm (в текстовом формате
LightGBM), кодировки категорий, числа итераций после early stopping и полные
``params``, поэтому поведение после ``save``/``load`` не меняется.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecastModel, inverse_target, transform_target
from .hurdle_model import HurdleForecastModel
from .lightgbm_model import LightGBMForecastModel

try:
    import lightgbm as lgb
except ImportError:  # модель не будет доступна, пока не установлен lightgbm
    lgb = None  # type: ignore[assignment]


class BlendForecastModel(BaseForecastModel):
    """Взвешенная смесь hurdle и одиночного LightGBM (см. модуль)."""

    name = "blend"
    file_suffix = ".pkl"
    install_hint = "pip install lightgbm"
    supports_early_stopping = True
    target_transform = "none"

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        common = {
            "seed": self.params.get("seed", 42),
            "threads": self.params.get("threads", 8),
        }
        self.hurdle = HurdleForecastModel({**common, **(self.params.get("hurdle") or {})})
        self.lgbm = LightGBMForecastModel({**common, **(self.params.get("lgbm") or {})})

    @classmethod
    def default_params(cls) -> dict:
        return {"weight_hurdle": 0.3, "hurdle": {}, "lgbm": {}}

    @classmethod
    def is_available(cls) -> bool:
        return lgb is not None

    @property
    def weight_hurdle(self) -> float:
        """Вес hurdle в смеси (0..1); остаток — вес lgbm."""
        return min(1.0, max(0.0, float(self.params.get("weight_hurdle", 0.3))))

    def journal_params(self) -> dict[str, object]:
        """Колонки журнала: значения обеих моделей через ``+`` (hurdle+lgbm)."""
        hurdle = self.hurdle.journal_params()
        lgbm = self.lgbm.journal_params()
        return {
            "iterations": f"{hurdle.get('iterations', '')}+{lgbm.get('iterations', '')}",
            "depth": f"{hurdle.get('depth', '')}+{lgbm.get('depth', '')}",
            "learning_rate": f"{hurdle.get('learning_rate', '')}+{lgbm.get('learning_rate', '')}",
        }

    def _split_iterations(self, n_iterations) -> tuple[object, Optional[int]]:
        """Разбирает ``n_iterations`` ансамбля на (hurdle, lgbm).

        Принимает ``None`` (значения из ``params``), число (обеим моделям) или
        словарь ``{"hurdle": …, "lgbm": …}``, который вернул
        :attr:`selected_iterations` после early stopping.
        """
        if isinstance(n_iterations, dict):
            return n_iterations.get("hurdle"), n_iterations.get("lgbm")
        if n_iterations is None:
            return None, None
        return int(n_iterations), int(n_iterations)

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        n_iterations=None,
    ) -> "BlendForecastModel":
        hurdle_iterations, lgbm_iterations = self._split_iterations(n_iterations)

        y = np.asarray(y_train, dtype="float64")
        y_lgbm = transform_target(y, self.lgbm.target_transform)
        y_valid_lgbm = (
            transform_target(np.asarray(y_valid, dtype="float64"), self.lgbm.target_transform)
            if y_valid is not None
            else None
        )

        print(f"  blend, стадия 1/2: hurdle (вес {self.weight_hurdle:g})", flush=True)
        self.hurdle.fit(
            x_train, y, x_valid, y_valid, early_stopping_rounds, hurdle_iterations
        )
        print(f"  blend, стадия 2/2: lgbm (вес {1.0 - self.weight_hurdle:g})", flush=True)
        self.lgbm.fit(
            x_train, y_lgbm, x_valid, y_valid_lgbm, early_stopping_rounds, lgbm_iterations
        )
        return self

    @property
    def selected_iterations(self) -> Optional[dict]:
        """Итерации обеих моделей после early stopping (для финального refit'а)."""
        hurdle = self.hurdle.selected_iterations
        lgbm = self.lgbm.selected_iterations
        if hurdle is None and lgbm is None:
            return None
        return {"hurdle": hurdle, "lgbm": lgbm}

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.hurdle.classifier is None or self.lgbm.booster is None:
            raise RuntimeError("Модель не обучена — сначала вызовите fit()")
        hurdle = inverse_target(self.hurdle.predict(features), self.hurdle.target_transform)
        lgbm = inverse_target(self.lgbm.predict(features), self.lgbm.target_transform)
        weight = self.weight_hurdle
        return weight * hurdle + (1.0 - weight) * lgbm

    def save(self, path: Path) -> None:
        if self.hurdle.classifier is None or self.lgbm.booster is None:
            raise RuntimeError("Нечего сохранять — модель не обучена")
        payload = {
            "params": self.params,
            "hurdle": {
                "classifier": self.hurdle.classifier.model_to_string(),
                "regressor": self.hurdle.regressor.model_to_string(),
                "categories": self.hurdle._categories,
                "best_iterations": self.hurdle._best_iterations,
                "params": self.hurdle.params,
            },
            "lgbm": {
                "booster": self.lgbm.booster.model_to_string(),
                "categories": self.lgbm._categories,
                "best_iteration": self.lgbm._best_iteration,
                "params": self.lgbm.params,
            },
        }
        Path(path).write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "BlendForecastModel":
        if lgb is None:
            raise RuntimeError(cls.install_hint)
        payload = pickle.loads(Path(path).read_bytes())
        model = cls(payload.get("params"))

        hurdle = payload["hurdle"]
        model.hurdle.classifier = lgb.Booster(model_str=hurdle["classifier"])
        model.hurdle.regressor = lgb.Booster(model_str=hurdle["regressor"])
        model.hurdle._categories = {
            col: [str(value) for value in values]
            for col, values in hurdle["categories"].items()
        }
        model.hurdle._best_iterations = hurdle.get("best_iterations") or {
            "classifier": None,
            "regressor": None,
        }
        model.hurdle.params = hurdle.get("params") or model.hurdle.params

        lgbm = payload["lgbm"]
        model.lgbm.booster = lgb.Booster(model_str=lgbm["booster"])
        model.lgbm._categories = {
            col: [str(value) for value in values]
            for col, values in lgbm["categories"].items()
        }
        model.lgbm._best_iteration = lgbm.get("best_iteration")
        model.lgbm.params = lgbm.get("params") or model.lgbm.params
        return model
