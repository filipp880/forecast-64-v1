"""Загрузка обученной модели (тип — из config.json, реестр в ``app.models``) и инференс."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

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
from .models import (
    BaseForecastModel,
    get_model_class,
    inverse_target,
    postprocess_forecast,
)


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
        # Изолированное состояние: ошибка любого этапа не оставляет рабочую модель.
        self.model = None
        self.config = {}
        self._cached_forecast = None
        candidate = type(self)(self.model_dir)
        try:
            candidate._load_checked()
        except Exception as exc:
            raise ModelNotReadyError(f"Не удалось загрузить модель: {exc}") from exc
        self.config, self.model = candidate.config, candidate.model
        return self

    def _load_checked(self) -> "Predictor":
        """Загружает модель и конфиг из ``model_dir``.

        Тип модели берётся из поля ``model_type`` конфига (по умолчанию
        ``lgbm`` — обратная совместимость с конфигами без поля).
        Модель подгружается лениво: приложение стартует и отвечает на
        ``/health`` даже до обучения, а ``POST /predict`` вернёт 503
        с понятным сообщением, пока модель не обучена.
        """
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise ModelNotReadyError(f"Не найден конфиг модели: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        model_type = str(self.config.get("model_type", "lgbm"))
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

        import hashlib
        expected_hash = self.config.get("model_artifact_sha256")
        if expected_hash and hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_hash:
            raise ModelNotReadyError("SHA-256 модели не соответствует config")
        self.model = model_cls.load(model_path)
        for name, expected in (("horizon", HORIZON), ("feature_columns", FEATURE_COLUMNS), ("cat_features", CAT_FEATURES)):
            if self.config.get(name, expected) != expected:
                raise ModelNotReadyError(f"Несовместимый config: {name}")
        if model_type == "lgbm":
            # Текстовый Booster не хранит внешнее преобразование таргета.
            transform = self.config.get("target_transform", "log1p")
            if transform not in ("log1p", "none"):
                raise ModelNotReadyError("Неизвестное преобразование таргета")
            self.model.target_transform = transform
        if self.config.get("target_transform", "log1p") != self.model.target_transform:
            raise ModelNotReadyError("Несовместимое преобразование таргета")
        boosters = []
        for part in (self.model, getattr(self.model, "lgbm", None), getattr(self.model, "hurdle", None)):
            for attr in ("booster", "classifier", "regressor"):
                booster = getattr(part, attr, None)
                if booster is not None:
                    boosters.append(booster)
        if any(b.feature_name() != FEATURE_COLUMNS for b in boosters):
            raise ModelNotReadyError("Признаки артефакта не соответствуют коду")
        catboost = getattr(self.model, "model", None)
        if catboost is not None and getattr(catboost, "feature_names_", FEATURE_COLUMNS) != FEATURE_COLUMNS:
            raise ModelNotReadyError("Признаки CatBoost не соответствуют коду")
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
        path = self.model_dir / self.config.get("preprocessor_file", "preprocessor.pkl")
        if not path.exists():
            if self.config.get("preprocessor_file"):
                raise ModelNotReadyError(f"Нет объявленного препроцессора: {path}")
            return
        try:
            payload = path.read_bytes()
            import hashlib
            expected_hash = self.config.get("preprocessor_sha256")
            if expected_hash and hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ModelNotReadyError("SHA-256 препроцессора не соответствует config")
            spec = pickle.loads(payload)
        except Exception as exc:
            raise ModelNotReadyError(f"Не удалось прочитать {path}: {exc}") from None
        checks = {
            "lookback_days": int(self.config.get("lookback_days", LOOKBACK_DAYS)),
            "horizon": int(self.config.get("horizon", HORIZON)),
            "feature_columns": list(FEATURE_COLUMNS),
            "cat_features": list(CAT_FEATURES),
        }
        from .features import PREPROCESSING_VERSION
        if self.config.get("stock_policy", "legacy") == "window" and spec.get("schema_version") != PREPROCESSING_VERSION:
            raise ModelNotReadyError("Несовместимая версия window-препроцессора")
        from .features import HISTORY_WINDOWS, NEVER_SOLD
        if list(spec.get("history_windows", HISTORY_WINDOWS)) != list(HISTORY_WINDOWS) or spec.get("never_sold", NEVER_SOLD) != NEVER_SOLD:
            raise ModelNotReadyError("Несовместимые параметры исторических признаков")
        if spec.get("target_transform", self.config.get("target_transform")) != self.config.get("target_transform"):
            raise ModelNotReadyError("Преобразование таргета в препроцессоре не совпадает с config")
        if spec.get("stock_policy", "legacy") != self.config.get("stock_policy", "legacy"):
            raise ModelNotReadyError("Несовместимая политика исторических stock")
        problems = [
            name
            for name, expected in checks.items()
            if (list(spec.get(name, [])) if isinstance(expected, list) else int(spec.get(name, -1)))
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
            raise ValueError("В переданных данных нет ни одной пары «склад x товар»")

        cutoff = panel["date"].max()
        horizon = int(self.config.get("horizon", HORIZON))
        lookback = int(self.config.get("lookback_days", LOOKBACK_DAYS))

        features = make_features(panel, cutoff, range(1, horizon + 1), lookback, self.config.get("stock_policy", "legacy"))
        post = self.config.get("postprocess") or {}
        if self.config.get("calibration"):
            from .calibration import predict_calibrated
            predictions = predict_calibrated(self.model, features, self.config["calibration"])
        else:
            predictions = postprocess_forecast(
                inverse_target(
                    self.model.predict(features),
                    str(self.config.get("target_transform", "log1p")),
                ),
                zero_threshold=float(post.get("zero_threshold", 0.0)),
                round_int=bool(post.get("round_int", False)),
            )
        forecast = features[[*KEY, "date"]].copy()
        import numpy as np
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("Модель вернула некорректный quantity")
        forecast["quantity"] = predictions.astype(float)
        forecast = forecast.sort_values(["date", *KEY]).reset_index(drop=True)
        forecast = forecast[["date", *KEY, "quantity"]]
        forecast["date"] = forecast["date"].dt.strftime("%Y-%m-%d")

        from .submission import validate_forecast
        validate_forecast(forecast, df_sales)
        self._cached_forecast = forecast
        return forecast

    def get_cached_forecast(self) -> pd.DataFrame:
        """Последний посчитанный прогноз (для GET-эндпоинтов)."""
        if self._cached_forecast is None:
            raise ForecastNotReadyError(
                "Прогноз не рассчитан. Сначала вызовите POST /predict."
            )
        return self._cached_forecast
