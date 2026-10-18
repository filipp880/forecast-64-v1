#!/usr/bin/env python3
"""Обучение модели прогноза продаж (регрессор на 64 дня — простая база).

Тип модели выбирается через ``--model-type`` (реестр в ``app/models``),
по умолчанию ``lgbm``. Параметры модели
задаются JSON-строкой ``--model-params`` поверх значений по умолчанию из
класса модели. Новая модель (HistGradientBoosting, hurdle-композиция и т.п.)
добавляется как новый класс в ``app/models`` — этот скрипт править не нужно.

Таргет трансформируется по правилу класса модели (``target_transform``,
по умолчанию ``log1p(quantity)``).

Постобработка прогноза — ``--zero-threshold`` (обнулять прогнозы ниже порога)
и ``--round-int`` (округлять до целых): применяется и к метрикам валидации,
и при инференсе API (записывается в ``config.json`` → ``postprocess``).
Подбирается без переобучения: ``python evaluate.py --sweep-thresholds ...``.

Схема обучения — строгий holdout по времени (training_scheme=strict-holdout):

1. обучающие срезы — только те, чьи таргеты заканчиваются до окна отбора
   (иначе утечка: модель «видела» бы конец проверяемых периодов);
2. early stopping — на окне отбора (предпоследние ``--horizon`` дней данных);
3. пять метрик ТЗ (SMAPE/WAPE/RMSE/|Bias|/RMSLE, см. ``app/metrics.py``)
   считаются на тестовом окне — последних ``--horizon`` днях данных, которые
   ранняя остановка не видела; метрики и параметры пишутся в ``config.json``
   (``validation``);
4. финальный refit — модель переобучается на ВСЕХ срезах (включая последние
   дни) с зафиксированным числом итераций из шага 2; это модель «в бой»
   (``--no-refit`` отключает).

Если данных мало для строгой схемы, происходит автоматический откат к простой
схеме: train = все срезы кроме последнего, валидация = последний срез.
Честная оценка по нескольким срезам истории — ``evaluate.py``.

Признаки считаются по истории не глубже ``--lookback`` (по умолчанию 180)
дней до каждого среза обучения — в точности как при инференсе в API, куда
приходят только последние ~180 дней.

Примеры запуска (из директории forecast-container/):
    python train.py \
        --sales ../data/sales_dataset_hakaton_train.csv \
        --stock ../data/stock_dataset_hakaton_train.csv \
        --out-dir model

Быстрая проба (несколько срезов и мало итераций):
    python train.py --max-cutoffs 4 --model-params '{"n_estimators": 300}'

Каждый прогон дописывает строку в журнал экспериментов (``experiments.csv``):
тег (``--tag``), конфигурация, метрики, длительность — для сравнения вариантов
(``--no-log`` отключает запись, ``--log-file`` меняет файл).

Детерминизм: фиксированы seed и thread_count, обучение на CPU.
Повторный запуск на тех же данных даёт ту же модель и тот же прогноз.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    HORIZON,
    KEY,
    LOOKBACK_DAYS,
    build_panel,
    make_training_frame,
    preprocessing_spec,
)
from app.models import (
    available_model_types,
    get_model_class,
    inverse_target,
    postprocess_forecast,
    transform_target,
)
from app.models.base import format_duration
from app.metrics import compute_all_metrics, format_metrics


def make_cutoffs(
    panel: pd.DataFrame,
    lookback_days: int,
    horizon: int,
    step_days: int,
) -> list[pd.Timestamp]:
    """Сетка дат-отсечек: от ``первая дата + lookback`` до ``последняя - 1``."""
    first = panel["date"].min()
    last = panel["date"].max()
    start = first + pd.Timedelta(days=lookback_days)
    end = last - pd.Timedelta(days=1)
    if start > end:
        raise ValueError(
            f"Истории не хватает: нужно минимум {lookback_days + horizon} дней данных"
        )
    return list(pd.date_range(start, end, freq=f"{step_days}D"))


def holdout_cutoffs(panel: pd.DataFrame, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Срезы строгой схемы: ``(срез отбора, срез теста)``.

    Тестовое окно — последние ``horizon`` дней данных, окно отбора —
    ``horizon`` дней перед ним. Оба среза дают полный горизонт 1..horizon.
    """
    last = panel["date"].max()
    test_cutoff = last - pd.Timedelta(days=horizon)
    select_cutoff = test_cutoff - pd.Timedelta(days=horizon)
    return select_cutoff, test_cutoff


def build_dataset(
    panel: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    horizon: int,
    lookback_days: int,
) -> list[pd.DataFrame]:
    """По каждому срезу — рамка признаков + таргет (горизонты только внутри данных)."""
    frames: list[pd.DataFrame] = []
    last_date = panel["date"].max()
    started = time.time()
    for index, cutoff in enumerate(cutoffs, start=1):
        horizons = [
            h
            for h in range(1, horizon + 1)
            if cutoff + pd.Timedelta(days=h) <= last_date
        ]
        if not horizons:
            continue
        frame = make_training_frame(panel, cutoff, horizons, lookback_days)
        elapsed = time.time() - started
        eta = elapsed / index * (len(cutoffs) - index)
        print(
            f"  [{index}/{len(cutoffs)}] срез {cutoff.date()}: {len(frame):,} строк, "
            f"горизонт {len(horizons)} дн. (прошло {format_duration(elapsed)}, "
            f"осталось ~{format_duration(eta)})",
            flush=True,
        )
        frames.append(frame)
    if not frames:
        raise ValueError("Не удалось собрать обучающую выборку: слишком мало данных")
    return frames


def evaluate_frame(
    model,
    model_cls,
    frame: pd.DataFrame,
    zero_threshold: float = 0.0,
    round_int: bool = False,
) -> dict:
    """Пять метрик ТЗ на рамке обучения по её собственному таргету.

    Постобработка — та же, что будет на инференсе (``postprocess`` в config.json).
    """
    truth = frame[["date", *KEY]].copy()
    truth["quantity"] = frame["target"].to_numpy(dtype="float64")
    forecast = frame[["date", *KEY]].copy()
    forecast["quantity"] = postprocess_forecast(
        inverse_target(model.predict(frame), model_cls.target_transform),
        zero_threshold=zero_threshold,
        round_int=round_int,
    )
    return compute_all_metrics(truth, forecast)


EXP_LOG_COLUMNS = [
    "timestamp",
    "tag",
    "scheme",
    "test_cutoff",
    "model_type",
    "max_cutoffs",
    "cutoff_step",
    "iterations",
    "depth",
    "learning_rate",
    "refit",
    "early_stopping_rounds",
    "seed",
    "lookback",
    "horizon",
    "es_iterations",
    "zero_threshold",
    "round_int",
    "n_cutoffs",
    "train_rows",
    "smape",
    "wape",
    "rmse",
    "bias",
    "rmsle",
    "duration_sec",
    "out_dir",
]


def log_experiment(
    args, model, validation_info: dict | None, n_cutoffs: int, train_rows: int, duration_sec: float
) -> Path:
    """Дописывает строку эксперимента в CSV-журнал (сравнение прогонов)."""
    metrics = (validation_info or {}).get("metrics") or {}
    early_stopping = (validation_info or {}).get("early_stopping") or {}
    journal = model.journal_params()

    def metric(name: str):
        value = metrics.get(name)
        return round(float(value), 4) if value is not None else ""

    row = [
        datetime.now().isoformat(timespec="seconds"),
        args.tag,
        (validation_info or {}).get("scheme", ""),
        (validation_info or {}).get("cutoff", ""),
        model.name,
        args.max_cutoffs if args.max_cutoffs else "all",
        args.cutoff_step,
        journal.get("iterations", ""),
        journal.get("depth", ""),
        journal.get("learning_rate", ""),
        args.refit,
        args.early_stopping_rounds,
        args.seed,
        args.lookback,
        args.horizon,
        early_stopping.get("iterations", ""),
        args.zero_threshold,
        args.round_int,
        n_cutoffs,
        train_rows,
        metric("smape"),
        metric("wape"),
        metric("rmse"),
        metric("bias"),
        metric("rmsle"),
        round(duration_sec),
        args.out_dir,
    ]
    path = Path(args.log_file)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        header = path.open("r", encoding="utf-8").readline().strip()
        if header != ",".join(EXP_LOG_COLUMNS):
            backup = path.with_name(path.name + ".bak")
            path.replace(backup)
            print(f"Схема журнала изменилась — старая версия сохранена: {backup}")
            new_file = True
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(EXP_LOG_COLUMNS)
        writer.writerow(row)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели прогноза (64 дня)")
    parser.add_argument(
        "--sales",
        default="../data/sales_dataset_hakaton_train.csv",
        help="CSV с продажами (формат sales_dataset_hakaton.csv)",
    )
    parser.add_argument(
        "--stock",
        default="../data/stock_dataset_hakaton_train.csv",
        help="CSV с остатками (формат stock_dataset_hakaton.csv)",
    )
    parser.add_argument("--out-dir", default="model", help="Куда сохранить модель и config.json")
    parser.add_argument(
        "--model-type",
        default="lgbm",
        choices=available_model_types(),
        help="Тип модели из реестра app/models (по умолчанию lgbm)",
    )
    parser.add_argument(
        "--model-params",
        default="{}",
        help='JSON-объект с параметрами модели поверх значений по умолчанию: \'{"n_estimators": 300}\'',
    )
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Горизонт прогноза, дней")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Глубина истории для признаков")
    parser.add_argument("--cutoff-step", type=int, default=28, help="Шаг между срезами обучения, дней")
    parser.add_argument(
        "--max-cutoffs",
        type=int,
        default=None,
        help="Оставить только N последних обучающих срезов (и срезов refit'а) — для быстрой пробы",
    )
    parser.add_argument(
        "--refit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Финальный refit на всех срезах с отобранным числом итераций (--no-refit отключает)",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.0,
        help="Обнулять прогнозы ниже порога (постобработка под SMAPE п. 5.5; 0 — выключено)",
    )
    parser.add_argument(
        "--round-int",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Округлять прогноз до целых после обнуления (--no-round-int отключает)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8, help="Число потоков (фиксировано для детерминизма)")
    parser.add_argument("--tag", default="", help="Метка эксперимента для журнала (что изменено)")
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Дописывать результат в CSV-журнал экспериментов (--no-log отключает)",
    )
    parser.add_argument(
        "--log-file",
        default="experiments.csv",
        help="Файл журнала экспериментов (по умолчанию experiments.csv рядом с train.py)",
    )

    args = parser.parse_args(argv)
    try:
        params = json.loads(args.model_params)
    except json.JSONDecodeError as exc:
        parser.error(f"--model-params: некорректный JSON ({exc})")
    if not isinstance(params, dict):
        parser.error('--model-params должен быть JSON-объектом, например \'{"n_estimators": 300}\'')
    args.model_params = params
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    def step(title: str) -> None:
        """Отметка этапа обучения с общим прошедшим временем — «ноутбучный» прогресс."""
        print(f"--- {title} (прошло {format_duration(time.time() - started)}) ---", flush=True)

    model_cls = get_model_class(args.model_type)
    model = model_cls({"seed": args.seed, "threads": args.threads, **args.model_params})
    print(f"Тип модели: {model_cls.name}, параметры: {model.params}")
    print(f"Данные: sales={args.sales}, stock={args.stock}")
    print("(обучение идёт на полном train-наборе; сэмплы sample_*_6mo.csv — только для тестов API)")

    step("Загрузка данных и сборка панели")
    sales = pd.read_csv(args.sales, parse_dates=["date"])
    stock = pd.read_csv(args.stock, parse_dates=["date"])
    panel = build_panel(sales, stock)
    n_series = panel.groupby(["warehouse_name", "article"]).ngroups
    print(
        f"  панель: {len(panel):,} строк, {panel['date'].nunique()} дней, "
        f"{n_series:,} серий ({panel['date'].min().date()} .. {panel['date'].max().date()})"
    )

    cutoffs = make_cutoffs(panel, args.lookback, args.horizon, args.cutoff_step)
    select_cutoff, test_cutoff = holdout_cutoffs(panel, args.horizon)
    eligible = [
        c for c in cutoffs if c + pd.Timedelta(days=args.horizon) <= select_cutoff
    ]
    strict = bool(eligible) and select_cutoff > panel["date"].min()

    validation_info = None
    if strict:
        last_date = panel["date"].max()
        train_cutoffs = eligible[-args.max_cutoffs :] if args.max_cutoffs else eligible
        day = pd.Timedelta(days=1)
        print(
            f"Схема: строгий holdout — отбор {select_cutoff.date() + day}..{test_cutoff.date()}, "
            f"тест {test_cutoff.date() + day}..{last_date.date()}",
            flush=True,
        )
        step(
            f"Сборка обучающей выборки: {len(train_cutoffs)} срезов, "
            f"таргеты заканчиваются не позже {select_cutoff.date()}"
        )
        train_frames = build_dataset(panel, train_cutoffs, args.horizon, args.lookback)
        if any(frame["date"].max() > select_cutoff for frame in train_frames):
            raise RuntimeError("Утечка: таргеты обучающих срезов заходят в окно отбора")
        train = pd.concat(train_frames, ignore_index=True)

        step(f"Сборка окон отбора и теста (h=1..{args.horizon})")
        valid_select = build_dataset(panel, [select_cutoff], args.horizon, args.lookback)[0]
        valid_test = build_dataset(panel, [test_cutoff], args.horizon, args.lookback)[0]

        y_train = transform_target(train["target"].to_numpy(dtype="float64"), model_cls.target_transform)
        y_select = transform_target(valid_select["target"].to_numpy(dtype="float64"), model_cls.target_transform)

        if args.early_stopping_rounds and not model_cls.supports_early_stopping:
            print(f"  прим.: модель '{model_cls.name}' не поддерживает early stopping — параметр пропущен")
        step(
            f"Обучение с early stopping: train {len(train):,} строк, "
            f"окно отбора {len(valid_select):,} строк"
        )
        model.fit(
            train,
            y_train,
            x_valid=valid_select,
            y_valid=y_select,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        n_iterations = model.selected_iterations

        step(f"Метрики ТЗ на тестовом окне {test_cutoff.date() + day}..{last_date.date()}")
        metrics = evaluate_frame(
            model, model_cls, valid_test, args.zero_threshold, args.round_int
        )
        detail = f"  (iterations={n_iterations})" if n_iterations is not None else ""
        print(
            f"  {format_metrics(metrics)}{detail}",
            flush=True,
        )

        train_rows_final, n_cutoffs_final = len(train), len(train_frames)
        if args.refit:
            refit_cutoffs = cutoffs[-args.max_cutoffs :] if args.max_cutoffs else cutoffs
            step(f"Refit на всех срезах ({len(refit_cutoffs)} шт.) — боевая модель")
            refit_frames = build_dataset(panel, refit_cutoffs, args.horizon, args.lookback)
            refit = pd.concat(refit_frames, ignore_index=True)
            y_refit = transform_target(refit["target"].to_numpy(dtype="float64"), model_cls.target_transform)
            refit_iterations = (
                n_iterations
                if n_iterations is not None
                else model.journal_params().get("iterations", "")
            )
            print(
                f"  refit {len(refit):,} строк, iterations={refit_iterations}",
                flush=True,
            )
            model.fit(refit, y_refit, n_iterations=n_iterations)
            train_rows_final, n_cutoffs_final = len(refit), len(refit_frames)

        validation_info = {
            "scheme": "strict-holdout",
            "cutoff": str(test_cutoff.date()),
            "max_horizon": args.horizon,
            "rows": int(len(valid_test)),
            "metrics": metrics,
            "early_stopping": {
                "cutoff": str(select_cutoff.date()),
                "rows": int(len(valid_select)),
                "iterations": n_iterations,
            },
        }
    else:
        if args.max_cutoffs:
            cutoffs = cutoffs[-args.max_cutoffs :]
        print(
            "Прим.: данных мало для строгой схемы — простая схема "
            "(train = все срезы кроме последнего).",
            flush=True,
        )
        step(f"Сборка обучающей выборки: {len(cutoffs)} срезов")
        frames = build_dataset(panel, cutoffs, args.horizon, args.lookback)

        if len(frames) >= 2:
            train = pd.concat(frames[:-1], ignore_index=True)
            valid = frames[-1]
        else:
            train, valid = frames[0], None

        y_train = transform_target(train["target"].to_numpy(dtype="float64"), model_cls.target_transform)
        step(
            f"Обучение: train {len(train):,} строк"
            + (f", valid {len(valid):,} строк" if valid is not None else "")
        )
        if valid is not None:
            if args.early_stopping_rounds and not model_cls.supports_early_stopping:
                print(f"  прим.: модель '{model_cls.name}' не поддерживает early stopping — параметр пропущен")
            y_valid = transform_target(valid["target"].to_numpy(dtype="float64"), model_cls.target_transform)
            model.fit(
                train,
                y_train,
                x_valid=valid,
                y_valid=y_valid,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            metrics = evaluate_frame(
                model, model_cls, valid, args.zero_threshold, args.round_int
            )
            valid_cutoff = valid["date"].min() - pd.Timedelta(days=1)
            max_horizon = int(valid["h"].max())
            print(f"Метрики ТЗ на срезе {valid_cutoff.date()} (h=1..{max_horizon}): {format_metrics(metrics)}", flush=True)
            if max_horizon < args.horizon:
                print(f"  прим.: срез покрывает не весь горизонт 1..{args.horizon} — для честной оценки запустите evaluate.py")
            validation_info = {
                "scheme": "legacy-split",
                "cutoff": str(valid_cutoff.date()),
                "max_horizon": max_horizon,
                "rows": int(len(valid)),
                "metrics": metrics,
            }
        else:
            model.fit(train, y_train)
        train_rows_final, n_cutoffs_final = len(train), len(frames)

    step("Сохранение модели, препроцессора и конфига")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model{model_cls.file_suffix}"
    model.save(model_path)

    preprocessor_path = out_dir / "preprocessor.pkl"
    preprocessor_path.write_bytes(
        pickle.dumps(
            {
                **preprocessing_spec(),
                "target_transform": model_cls.target_transform,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    )

    config = {
        "model_type": model_cls.name,
        "model_file": model_path.name,
        "preprocessor_file": preprocessor_path.name,
        "horizon": args.horizon,
        "lookback_days": args.lookback,
        "target_transform": model_cls.target_transform,
        "postprocess": {
            "zero_threshold": args.zero_threshold,
            "round_int": args.round_int,
        },
        "random_seed": args.seed,
        "thread_count": args.threads,
        "feature_columns": FEATURE_COLUMNS,
        "cat_features": CAT_FEATURES,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_cutoffs": n_cutoffs_final,
        "train_rows": int(train_rows_final),
        "model_params": model.params,
        "validation": validation_info,
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Готово за {time.time() - started:.0f} с.")
    print(f"Модель: {model_path}")
    print(f"Препроцессор: {preprocessor_path}")
    print(f"Конфиг: {config_path}")
    if args.log:
        log_path = log_experiment(
            args, model, validation_info, n_cutoffs_final, train_rows_final, time.time() - started
        )
        print(f"Журнал: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
