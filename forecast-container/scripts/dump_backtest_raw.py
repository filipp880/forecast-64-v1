"""Выгрузка сырых дневных компонент бленда на бэктесте — вход для ``scripts/postproc_lab.py``.

Для каждого полного среза сохраняет parquet-файл: ключ (date, склад, артикул),
горизонт ``h``, факт ``quantity``, компоненты прогноза (``p`` — вероятность
классификатора hurdle, ``m`` — величина его регрессора, ``lgbm_raw`` — прогноз
lgbm в исходном пространстве) и сегментные признаки на дату отсечки.

Сами прогнозы не собираются: варианты бленда/постобработки проверяет
лаборатория, а этот шаг нужен один раз (сборка панели и признаков — самая
дорогая часть).

Запуск (из директории forecast-container/)::

    python scripts/dump_backtest_raw.py                      # model_blend -> logs/raw/
    python scripts/dump_backtest_raw.py --model-dir model_blend --out-dir logs/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import HORIZON, LOOKBACK_DAYS, build_panel, make_features
from app.models import get_model_class, inverse_target
from app.models.base import format_duration
from evaluate import full_cutoffs

DATA_KEY = ["date", "warehouse_name", "article"]
SEGMENT_COLUMNS = ["mean_28", "zero_share_90", "stock_last", "days_of_cover", "n_sale_days_90"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Выгрузка сырых компонент бленда для лаборатории постобработки")
    parser.add_argument("--model-dir", default="model_blend", help="Директория модели (ожидается blend)")
    parser.add_argument(
        "--extra-model",
        default=None,
        help="Директория дополнительной модели (например, model_tweedie) — колонка extra_raw",
    )
    parser.add_argument(
        "--sales", default="../data/sales_dataset_hakaton_train.csv", help="CSV с продажами"
    )
    parser.add_argument(
        "--stock", default="../data/stock_dataset_hakaton_train.csv", help="CSV с остатками"
    )
    parser.add_argument("--cutoff-step", type=int, default=28, help="Шаг между срезами, дней")
    parser.add_argument("--max-cutoffs", type=int, default=8, help="Сколько последних полных срезов")
    parser.add_argument("--out-dir", default="logs/raw", help="Куда писать parquet-файлы срезов")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "lgbm"))
    model_cls = get_model_class(model_type)
    model = model_cls.load(model_dir / (config.get("model_file") or f"model{model_cls.file_suffix}"))
    if not hasattr(model, "hurdle") or not hasattr(model, "lgbm"):
        print(f"Ошибка: скрипт ожидает модель blend, а загружена {model_type!r}", file=sys.stderr)
        return 2
    horizon = int(config.get("horizon", HORIZON))
    lookback = int(config.get("lookback_days", LOOKBACK_DAYS))
    print(f"Модель: {model_type} из {model_dir} (горизонт {horizon}, lookback {lookback})", flush=True)

    extra_model = None
    extra_transform = "log1p"
    extra_dir = None
    if args.extra_model:
        extra_dir = Path(args.extra_model)
        extra_config = json.loads((extra_dir / "config.json").read_text(encoding="utf-8"))
        extra_type = str(extra_config.get("model_type", "lgbm"))
        extra_cls = get_model_class(extra_type)
        extra_model = extra_cls.load(
            extra_dir / (extra_config.get("model_file") or f"model{extra_cls.file_suffix}")
        )
        extra_transform = str(extra_config.get("target_transform", "log1p"))
        print(f"Доп. компонента: {extra_type} из {extra_dir} (transform {extra_transform})", flush=True)

    print("Загрузка данных...", flush=True)
    sales = pd.read_csv(args.sales, parse_dates=["date"])
    stock = pd.read_csv(args.stock, parse_dates=["date"])
    panel = build_panel(sales, stock)
    truth = panel[[*DATA_KEY, "quantity"]]

    cutoffs = full_cutoffs(panel, lookback, horizon, args.cutoff_step, args.max_cutoffs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Выгрузка {len(cutoffs)} срезов в {out_dir}...", flush=True)

    started_eval = time.time()
    for index, cutoff in enumerate(cutoffs, start=1):
        features = make_features(panel, cutoff, range(1, horizon + 1), lookback)

        encoded = model.hurdle._encode(features, fit=False)
        probability = model.hurdle.classifier.predict(
            encoded, num_iteration=model.hurdle._best_iterations.get("classifier")
        )
        magnitude = model.hurdle.regressor.predict(
            encoded, num_iteration=model.hurdle._best_iterations.get("regressor")
        )
        lgbm_raw = inverse_target(model.lgbm.predict(features), model.lgbm.target_transform)

        frame = features[[*DATA_KEY, "h", *SEGMENT_COLUMNS]].copy()
        frame["p"] = np.asarray(probability, dtype="float32")
        frame["m"] = np.asarray(magnitude, dtype="float32")
        frame["lgbm_raw"] = np.asarray(lgbm_raw, dtype="float32")
        if extra_model is not None:
            extra_raw = inverse_target(extra_model.predict(features), extra_transform)
            frame["extra_raw"] = np.asarray(extra_raw, dtype="float32")
        frame = frame.merge(truth, on=DATA_KEY, how="left")
        frame["quantity"] = frame["quantity"].fillna(0.0).astype("float32")
        frame["cutoff"] = cutoff.strftime("%Y-%m-%d")

        path = out_dir / f"raw_{cutoff.date()}.parquet"
        frame.to_parquet(path, index=False)

        elapsed = time.time() - started_eval
        eta = elapsed / index * (len(cutoffs) - index)
        print(
            f"  [{index}/{len(cutoffs)}] срез {cutoff.date()}: {len(frame):,} строк -> {path.name} "
            f"(осталось ~{format_duration(eta)})",
            flush=True,
        )

    meta = {
        "model_dir": str(model_dir),
        "model_type": model_type,
        "weight_hurdle": float(model.weight_hurdle),
        "gate": float(model.hurdle.params.get("gate", 0.5)),
        "horizon": horizon,
        "lookback_days": lookback,
        "postprocess": config.get("postprocess") or {},
        "cutoffs": [str(cutoff.date()) for cutoff in cutoffs],
        "extra_model_dir": str(extra_dir) if extra_dir else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Метаданные: {out_dir / 'meta.json'}")
    print(f"Готово за {time.time() - started:.0f} с.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
