"""Checked weekly model loading and shared API/competition inference."""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .features import (CAT_FEATURES, FEATURE_COLUMNS, FORECAST_WEEKS, KEY,
                       LOOKBACK_DAYS, PREPROCESSING_VERSION, build_panel,
                       make_features, next_week_start, preprocessing_spec, truncate_history)
from .models import get_model_class, inverse_target, postprocess_forecast
from .submission import normalized_pairs, validate_forecast


class ModelNotReadyError(RuntimeError):
    pass


class ForecastNotReadyError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.config = {}
        self.model = None
        self._cached_forecast = None

    @property
    def is_loaded(self):
        return self.model is not None

    def load(self):
        self.model, self.config, self._cached_forecast = None, {}, None
        candidate = type(self)(self.model_dir)
        try:
            candidate._load_checked()
        except Exception as exc:
            raise ModelNotReadyError(f"Не удалось загрузить недельную модель: {exc}") from exc
        self.model, self.config = candidate.model, candidate.config
        return self

    def _load_checked(self):
        self.config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        required = dict(granularity="week", horizon=FORECAST_WEEKS, forecast_weeks=FORECAST_WEEKS,
                        preprocessing_version=PREPROCESSING_VERSION, stock_aggregation="sum",
                        negative_stock_policy="preserve", stock_policy="window",
                        forecast_alignment="next_calendar_monday",
                        feature_columns=FEATURE_COLUMNS, cat_features=CAT_FEATURES)
        if self.config.get("model_type") == "research":
            from .enhancements import enhanced_spec, feature_columns
            required.update(preprocessing_version=4,feature_columns=feature_columns(),enhancements=enhanced_spec())
        if self.config.get("model_type") == "ideas":
            from .enhancements import enhanced_spec, feature_columns
            from .ideas_features import GROUPS, ideas_columns, ideas_spec
            required.update(preprocessing_version=5,feature_columns=feature_columns()+ideas_columns(GROUPS),
                            enhancements=enhanced_spec(),ideas=ideas_spec())
        for name, expected in required.items():
            if self.config.get(name) != expected:
                raise ValueError(f"Несовместимый config: {name}. Требуется переобученная weekly модель")
        model_type = self.config["model_type"]
        cls = get_model_class(model_type)
        if not cls.is_available():
            raise ValueError(cls.install_hint)
        model_path = self.model_dir / self.config["model_file"]
        payload = model_path.read_bytes()
        if self.config.get("model_artifact_sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("SHA-256 модели не соответствует config")
        preprocessor = (self.model_dir / self.config["preprocessor_file"]).read_bytes()
        if self.config.get("preprocessor_sha256") != hashlib.sha256(preprocessor).hexdigest():
            raise ValueError("SHA-256 препроцессора не соответствует config")
        spec = pickle.loads(preprocessor)
        expected_spec = preprocessing_spec(lookback_days=int(self.config.get("lookback_days", LOOKBACK_DAYS)))
        if model_type == "research":
            expected_spec.update(schema_version=4,feature_columns=required["feature_columns"],enhancements=required["enhancements"])
        if model_type == "ideas":
            expected_spec.update(schema_version=5,feature_columns=required['feature_columns'],
                                 enhancements=required['enhancements'],ideas=required['ideas'])
        if any(spec.get(k) != v for k, v in expected_spec.items()):
            raise ValueError("Спецификация препроцессора не соответствует weekly коду")
        if spec.get("target_transform") != self.config.get("target_transform"):
            raise ValueError("Преобразование таргета не совпадает")
        # Date provenance is mandatory even when an artifact is used only via API.
        for field in ("training_data_end", "selection_data_end"):
            if not self.config.get(field) or pd.isna(pd.to_datetime(self.config[field], errors="coerce")):
                raise ValueError(f"Отсутствует дата происхождения модели: {field}")
        self.model = cls.load(model_path)
        if model_type in ("research", "ideas"):
            self.model.check_schema()
            if self.model.params != self.config["model_params"]:
                raise ValueError("Research parameters differ from model config")
        if model_type == "lgbm":
            self.model.target_transform = self.config["target_transform"]
        if self.model.target_transform != self.config["target_transform"]:
            raise ValueError("Несовместимое преобразование таргета модели")
        for part in (self.model, getattr(self.model, "lgbm", None), getattr(self.model, "hurdle", None)):
            for attr in ("booster", "classifier", "regressor"):
                booster = getattr(part, attr, None)
                if booster is not None and booster.feature_name() != FEATURE_COLUMNS:
                    raise ValueError("Признаки бустера не соответствуют коду")
        cat = getattr(self.model, "model", None)
        if cat is not None and list(cat.feature_names_) != FEATURE_COLUMNS:
            raise ValueError("Признаки CatBoost не соответствуют коду")

    def ensure_loaded(self):
        if not self.is_loaded:
            self.load()

    def assert_history_compatible(self, cutoff):
        for field in ("training_data_end", "selection_data_end"):
            date = self.config.get(field)
            if not date or pd.Timestamp(date) > pd.Timestamp(cutoff):
                raise ValueError(f"Модель использовала данные после разрешённой отсечки: {field}={date}")

    def predict(self, df_sales, df_stock, *, history_cutoff=None, first_week=None, required_pairs=None):
        self.ensure_loaded()
        if history_cutoff is not None:
            self.assert_history_compatible(history_cutoff)
            df_sales, df_stock = truncate_history(df_sales, df_stock, history_cutoff)
        if df_sales.empty:
            raise ValueError("Нет продаж в разрешённой истории")
        cutoff = pd.Timestamp(history_cutoff) if history_cutoff is not None else pd.to_datetime(df_sales.date).max()
        # Future inventory is ignored in both API and competition modes.
        df_stock = df_stock.loc[pd.to_datetime(df_stock.date) <= cutoff]
        first = pd.Timestamp(first_week) if first_week is not None else next_week_start(cutoff)
        pairs = normalized_pairs(required_pairs if required_pairs is not None else df_sales[KEY].drop_duplicates())
        panel = build_panel(df_sales, df_stock, history_end=cutoff)
        if getattr(self.model,"name",None) in ("research", "ideas"):
            from .enhancements import add_money
            panel = add_money(panel, df_sales)
        features = make_features(panel, cutoff, lookback_days=int(self.config.get("lookback_days", LOOKBACK_DAYS)),
                                 first_week=first, required_pairs=pairs)
        if getattr(self.model,"name",None) in ("research", "ideas"):
            from .enhancements import augment_features
            features = augment_features(panel, features, cutoff)
        if getattr(self.model,"name",None) == "ideas":
            from .ideas_features import augment_ideas
            features = augment_ideas(panel, features, cutoff)
            features['feature_origin'] = pd.Timestamp(cutoff)
        from .calibration import forecast_quantities
        post = self.config.get("postprocess", {})
        predictions = forecast_quantities(self.model, features, float(post.get("zero_threshold", 0)),
                                          bool(post.get("round_int", False)), self.config.get("calibration"))
        forecast = features[["week", *KEY]].copy()
        forecast["week"] = forecast.week.dt.strftime("%Y-%m-%d")
        forecast["quantity"] = predictions.astype(float)
        forecast = forecast.sort_values(["week", *KEY]).reset_index(drop=True)
        validate_forecast(forecast, pairs, first)
        self._cached_forecast = forecast
        return forecast

    def get_cached_forecast(self):
        if self._cached_forecast is None:
            raise ForecastNotReadyError("Прогноз не рассчитан. Сначала вызовите POST /predict.")
        return self._cached_forecast.copy()
