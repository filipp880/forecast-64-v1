"""Реестр моделей: ``имя типа`` (config.json / CLI) → класс модели.

Добавление новой модели: создать класс-наследник ``BaseForecastModel``
в отдельном модуле пакета и дописать его в ``MODEL_REGISTRY`` ниже.
Ни ``train.py``, ни ``predictor.py`` править не нужно.
"""

from __future__ import annotations

from .base import BaseForecastModel, inverse_target, transform_target
from .catboost_model import CatBoostForecastModel

MODEL_REGISTRY: dict[str, type[BaseForecastModel]] = {
    cls.name: cls
    for cls in (CatBoostForecastModel,)
}


def available_model_types() -> list[str]:
    """Имена зарегистрированных типов моделей (для справки CLI и ошибок)."""
    return sorted(MODEL_REGISTRY)


def get_model_class(name: str) -> type[BaseForecastModel]:
    """Класс модели по имени типа; ``KeyError`` со списком доступных — если нет."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Неизвестный тип модели {name!r}. Доступны: {', '.join(available_model_types())}"
        ) from None


__all__ = [
    "BaseForecastModel",
    "MODEL_REGISTRY",
    "available_model_types",
    "get_model_class",
    "inverse_target",
    "transform_target",
]
