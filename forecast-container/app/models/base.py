"""Базовый интерфейс модели прогноза — точка расширения проекта.

Чтобы добавить новую модель (например, HistGradientBoosting, hurdle-композицию
«классификатор + регрессор» или нейросеть), достаточно:

1. создать класс-наследник :class:`BaseForecastModel` в этом пакете;
2. реализовать ``fit`` / ``predict`` / ``save`` / ``load`` и объявить
   ``name``, ``file_suffix``, ``default_params()`` (если имена гиперпараметров
   отличаются от колонок журнала — ``JOURNAL_PARAM_KEYS``, см.
   :meth:`BaseForecastModel.journal_params`);
3. зарегистрировать класс в ``MODEL_REGISTRY`` (``app/models/__init__.py``);
4. при необходимости добавить зависимость в ``requirements.txt``;
5. обучить модель: ``python train.py --model-type <имя>`` — ``train.py`` и
   ``app/predictor.py`` ничего не знают о конкретном классе и работают через
   конфиг ``config.json`` (поле ``model_type``).

Общий контракт данных (его обеспечивает ``app/features.py``):

- на вход ``fit``/``predict`` приходит ``DataFrame``, содержащий минимум все
  колонки из ``FEATURE_COLUMNS`` (лишние — ``date``, ключи серий — допустимы,
  модель обязана сама выбрать нужные);
- категориальные колонки — строки, числовые — float без NaN;
- таргет передаётся в ``fit`` уже в пространстве ``target_transform``
  (по умолчанию ``log1p``); обратное преобразование и постобработка
  (клиппинг, порог обнуления, округление) делают вызывающая сторона через
  :func:`inverse_target` и :func:`postprocess_forecast`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

TARGET_TRANSFORMS = ("log1p", "none")


def format_duration(seconds: float) -> str:
    """Длительность для вывода прогресса в терминал: ``45 с`` / ``3 мин 05 с`` / ``1 ч 12 мин``."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} с"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {sec:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


def transform_target(values: np.ndarray, kind: str) -> np.ndarray:
    """Преобразование таргета перед обучением (прогноз неотрицательной величины)."""
    values = np.asarray(values, dtype="float64")
    if kind == "log1p":
        return np.log1p(values)
    if kind == "none":
        return values
    raise ValueError(f"Неизвестное преобразование таргета: {kind!r} (есть: {TARGET_TRANSFORMS})")


def inverse_target(values: np.ndarray, kind: str) -> np.ndarray:
    """Обратное преобразование прогноза, сделанного в пространстве ``target_transform``."""
    values = np.asarray(values, dtype="float64")
    if kind == "log1p":
        return np.expm1(values)
    if kind == "none":
        return values
    raise ValueError(f"Неизвестное преобразование таргета: {kind!r} (есть: {TARGET_TRANSFORMS})")


def postprocess_forecast(
    values: np.ndarray,
    zero_threshold: float = 0.0,
    round_int: bool = False,
) -> np.ndarray:
    """Постобработка прогноза после обратного трансформа (сторона вызывающего).

    Настройка задаётся в ``config.json`` (``postprocess``) и переопределяется
    флагами ``train.py``/``evaluate.py``; инференс API применяет ровно те же
    шаги, что и оценка. Под метрику SMAPE п. 5.5 ТЗ (дни «факт=0, прогноз=0»
    входят как 0%) маленькие положительные прогнозы на «спящих» сериях
    максимально штрафуются, поэтому их выгодно обнулять:

    - отрицательные значения клиппятся в 0;
    - прогнозы меньше ``zero_threshold`` обнуляются (``0`` — выключено);
    - при ``round_int=True`` значения округляются до целых (``np.rint``).
    """
    result = np.clip(np.asarray(values, dtype="float64"), 0.0, None)
    if zero_threshold and zero_threshold > 0.0:
        result = np.where(result < float(zero_threshold), 0.0, result)
    if round_int:
        result = np.rint(result)
    return result


class BaseForecastModel(ABC):
    """Интерфейс модели прогноза на 64 дня.

    Классы моделей не знают ни про API, ни про структуру CSV — только про
    ``FEATURE_COLUMNS`` и трансформ таргета. Выбор модели при обучении:
    ``python train.py --model-type <name>``; при инференсе — поле
    ``model_type`` в ``model/config.json``.
    """

    name: ClassVar[str] = ""
    file_suffix: ClassVar[str] = ".bin"
    install_hint: ClassVar[str] = "pip install -r requirements.txt"
    supports_early_stopping: ClassVar[bool] = False
    target_transform: ClassVar[str] = "log1p"
    JOURNAL_PARAM_KEYS: ClassVar[dict[str, str]] = {
        "iterations": "iterations",
        "depth": "depth",
        "learning_rate": "learning_rate",
    }

    def journal_params(self) -> dict[str, object]:
        """Гиперпараметры модели для колонок журнала ``experiments.csv``.

        Ключи результата — поля журнала (``iterations``, ``depth``,
        ``learning_rate``), значения — соответствующие ключи ``self.params``.
        По умолчанию имена совпадают; модель с другими именами параметров
        переопределяет ``JOURNAL_PARAM_KEYS`` (например, LightGBM:
        ``n_estimators``/``num_leaves``). Отсутствующий параметр — ``""``.
        """
        return {
            field: self.params.get(key, "")
            for field, key in self.JOURNAL_PARAM_KEYS.items()
        }

    def __init__(self, params: Optional[dict] = None) -> None:
        merged = self.default_params()
        merged.update(params or {})
        self.params: dict = merged

    @classmethod
    def default_params(cls) -> dict:
        """Параметры по умолчанию (переопределяются через ``--model-params``)."""
        return {}

    @classmethod
    def is_available(cls) -> bool:
        """Установлена ли библиотека-бэкенд (в образе/окружении)."""
        return True

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        n_iterations: Optional[int] = None,
    ) -> "BaseForecastModel":
        """Обучает модель; ``y`` уже в пространстве ``target_transform``.

        ``n_iterations`` — жёстко зафиксировать число итераций (используется
        для финального refit'а на всех данных после early stopping); ``None`` —
        число берётся из ``params``. Модели без понятия итераций параметр
        игнорируют. Повторный вызов ``fit`` полностью заменяет обученную модель.
        """

    @property
    def selected_iterations(self) -> Optional[int]:
        """Число итераций лучшей модели после early stopping (``None`` — нет).

        Значение передаётся в :meth:`fit` как ``n_iterations`` при финальном
        refit'е, чтобы боевая модель повторила отобранную раннюю остановку.
        """
        return None

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Прогноз (в пространстве ``target_transform``) для строк ``features``."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Сохраняет артефакт(ы) модели в файл ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseForecastModel":
        """Загружает модель из файла, созданного :meth:`save`."""
