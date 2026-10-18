"""Загрузка обученной модели (тип — из config.json, реестр в ``app.models``) и инференс."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    HORIZON,
    KEY,
    LOOKBACK_DAYS,
    build_panel,
    make_features,
)
from .models import BaseForecastModel, get_model_class, inverse_target


class ModelNotReadyError(RuntimeError):
    """Файл обученной модели отсутствует, тип неизвестен или библиотека не установлена."""


class ForecastNotReadyError(RuntimeError):
    """GET-эндпоинты вызваны до первого POST /predict."""


class Predictor:
    """Обёртка над моделью: загрузка, инференс, кэш последнего прогноза."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.config: dict = {}
        self.model: Optional[BaseForecastModel] = None
        self._cached_forecast: Optional[pd.DataFrame] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "Predictor":
        """Загружает модель и конфиг из ``model_dir``.

        Тип модели берётся из поля ``model_type`` конфига (по умолчанию
        ``catboost`` — обратная совместимость с конфигами первой версии).
        Модель подгружается лениво: приложение стартует и отвечает на
        ``/health`` даже до обучения, а ``POST /predict`` вернёт 503
        с понятным сообщением, пока модель не обучена.
        """
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise ModelNotReadyError(f"Не найден конфиг модели: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        model_type = str(self.config.get("model_type", "catboost"))
        try:
            model_cls = get_model_class(model_type)
        except KeyError as exc:
            raise ModelNotReadyError(str(exc)) from None

        if not model_cls.is_available():
            raise ModelNotReadyError(
                f"Библиотека для модели '{model_type}' не установлена: {model_cls.install_hint}"
            )

        model_file = self.config.get("model_file") or f"model{model_cls.file_suffix}"
        model_path = self.model_dir / model_file
        if not model_path.exists():
            raise ModelNotReadyError(
                f"Модель не найдена: {model_path}. "
                f"Сначала обучите её: python train.py --model-type {model_type} --sales ... --stock ..."
            )

        self.model = model_cls.load(model_path)
        self._check_preprocessor()
        self._cached_forecast = None
        return self

    def _check_preprocessor(self) -> None:
        """Сверяет ``preprocessor.pkl`` с кодом признаков и ``config.json``.

        Файл создаёт ``train.py`` (ТЗ п. 10.5). Если его нет — пропускаем
        (конфиги первой версии). Если он есть и параметры пайплайна не
        совпали — модель обучена на других признаках, чем считает сервис:
        прогноз был бы молча испорчен, поэтому лучше явная ошибка.
        """
        path = self.model_dir / "preprocessor.pkl"
        if not path.exists():
            return
        try:
            spec = pickle.loads(path.read_bytes())
        except Exception as exc:
            raise ModelNotReadyError(f"Не удалось прочитать {path}: {exc}") from None
        checks = {
            "lookback_days": int(self.config.get("lookback_days", LOOKBACK_DAYS)),
            "horizon": int(self.config.get("horizon", HORIZON)),
            "feature_columns": list(FEATURE_COLUMNS),
            "cat_features": list(CAT_FEATURES),
        }
        problems = [
            name
            for name, expected in checks.items()
            if (list(spec[name]) if isinstance(expected, list) else int(spec.get(name, -1)))
            != expected
        ]
        if problems:
            raise ModelNotReadyError(
                "preprocessor.pkl не совпадает с кодом признаков/config.json "
                f"(расхождение: {', '.join(problems)}). Переобучите модель: python train.py"
            )

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def predict(self, df_sales: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
        """Прогноз на 64 дня от последней даты продаж; результат кэшируется."""
        self.ensure_loaded()

        panel = build_panel(df_sales, df_stock)
        if panel.empty:
            raise ValueError("В переданных данных нет ни одной пары «склад × товар»")

        cutoff = panel["date"].max()
        horizon = int(self.config.get("horizon", HORIZON))
        lookback = int(self.config.get("lookback_days", LOOKBACK_DAYS))

        features = make_features(panel, cutoff, range(1, horizon + 1), lookback)
        predictions = inverse_target(
            self.model.predict(features),
            str(self.config.get("target_transform", "log1p")),
        )
        predictions = np.clip(predictions, 0.0, None)

        forecast = features[[*KEY, "date"]].copy()
        forecast["quantity"] = predictions.astype(float)
        forecast = forecast.sort_values(["date", *KEY]).reset_index(drop=True)
        forecast = forecast[["date", *KEY, "quantity"]]
        forecast["date"] = forecast["date"].dt.strftime("%Y-%m-%d")

        self._cached_forecast = forecast
        return forecast

    def get_cached_forecast(self) -> pd.DataFrame:
        """Последний посчитанный прогноз (для GET-эндпоинтов)."""
        if self._cached_forecast is None:
            raise ForecastNotReadyError(
                "Прогноз не рассчитан. Сначала вызовите POST /predict."
            )
        return self._cached_forecast
