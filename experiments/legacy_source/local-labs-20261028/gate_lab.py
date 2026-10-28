"""Динамический гейт hurdle (docs/333, вектор 3) — проверка на дампе бэктеста.

Аудит предлагает опускать порог гейта для регулярных серий:
``theta = max(floor, 0.5 - alpha * log1p(mean_28))``, чтобы не терять дни с
``p`` чуть ниже 0.5 на сериях с большим ``m`` (штраф ``+/0`` = 200% SMAPE).

Стенд берёт готовый дамп ``scripts/dump_backtest_raw.py`` (p, m, lgbm_raw,
mean_28) и пересчитывает бленд ``0.3 * gated + 0.7 * lgbm`` с порогом,
зависящим от ``mean_28`` серии. Сетка (floor, alpha) + LOO-выбор по срезам;
сравнение с базой по композитному индексу ТЗ.

Запуск (из директории forecast-container/, ~1 мин)::

    python scripts/gate_lab.py                 # отчёт logs/gate_lab.json
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

from app.metrics import METRIC_NAMES, METRIC_WEIGHTS, bias_abs, rmsle, rmse, smape, wape
from app.models.base import postprocess_forecast

FLOORS = (0.1, 0.2)
ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="docs/333: динамический гейт по mean_28")
    parser.add_argument("--raw-dir", default="logs/raw")
    parser.add_argument("--out", default="logs/gate_lab.json")
    return parser.parse_args(argv)


def load_data(raw_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    data: list[dict] = []
    for cutoff in meta["cutoffs"]:
        frame = pd.read_parquet(raw_dir / f"raw_{cutoff}.parquet")
        series = (frame["warehouse_name"].astype(str) + "|" + frame["article"].astype(str)).to_numpy()
        data.append(
            {
                "cutoff": cutoff,
                "y": frame["quantity"].to_numpy("float64"),
                "p": frame["p"].to_numpy("float64"),
                "m": frame["m"].to_numpy("float64"),
                "lgbm": frame["lgbm_raw"].to_numpy("float64"),
                "mean_28": frame["mean_28"].to_numpy("float64"),
                "series": series,
            }
        )
    return meta, data


def cutoff_metrics(item: dict, pred: np.ndarray) -> dict:
    y, series = item["y"], item["series"]
    return {
        "smape": smape(y, pred, series),
        "wape": wape(y, pred),
        "rmse": rmse(y, pred),
        "bias": bias_abs(y, pred, series),
        "rmsle": rmsle(y, pred),
    }


def predict_with_theta(item: dict, weight: float, theta) -> np.ndarray:
    gated = np.where(item["p"] >= theta, item["m"], 0.0)
    raw = weight * gated + (1.0 - weight) * item["lgbm"]
    return postprocess_forecast(raw, 0.0, True)


def constant_theta(value: float):
    return value


def dynamic_theta(floor: float, alpha: float):
    def threshold(item: dict) -> np.ndarray:
        return np.maximum(floor, 0.5 - alpha * np.log1p(item["mean_28"]))

    return threshold


def composite(metrics: dict, baseline: dict) -> float:
    return float(sum(METRIC_WEIGHTS[name] * metrics[name] / baseline[name] for name in METRIC_NAMES))


def evaluate_variant(data: list[dict], weight: float, theta) -> tuple[list[dict], dict]:
    rows = []
    for item in data:
        threshold = theta(item) if callable(theta) else np.full(len(item["y"]), theta)
        rows.append(cutoff_metrics(item, predict_with_theta(item, weight, threshold)))
    means = {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES}
    return rows, means


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    meta, data = load_data(Path(args.raw_dir))
    weight = float(meta["weight_hurdle"])
    print(f"Срезов: {len(data)}, вес hurdle {weight:g}, гейт по умолчанию {meta['gate']:g}")

    base_rows, baseline = evaluate_variant(data, weight, constant_theta(float(meta["gate"])))
    print("База:", {name: round(baseline[name], 4) for name in METRIC_NAMES})

    variants: dict[str, dict] = {}

    def add_variant(label: str, theta) -> None:
        rows, means = evaluate_variant(data, weight, theta)
        index = composite(means, baseline)
        variants[label] = {"metrics": means, "index": index, "rows": rows}
        print(
            f"  {label:<28} SMAPE {means['smape']:.2f} WAPE {means['wape']:.2f} "
            f"RMSE {means['rmse']:.4f} Bias {means['bias']:.2f} RMSLE {means['rmsle']:.4f} "
            f"индекс {index:.4f}"
        )

    print("\nВарианты:")
    add_variant("theta 0.5 (база)", constant_theta(0.5))
    add_variant("theta 0.4 (глоб.)", constant_theta(0.4))
    add_variant("theta 0.3 (глоб.)", constant_theta(0.3))
    for floor in FLOORS:
        for alpha in ALPHAS:
            if alpha == 0.0 and floor == FLOORS[0]:
                continue
            add_variant(f"theta>={floor:g}, alpha={alpha:g}", dynamic_theta(floor, alpha))

    labels = [label for label in variants if label.startswith("theta>=")]
    folds: list[str] = []
    fold_metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for fold in range(len(data)):
        train = [index for index in range(len(data)) if index != fold]
        best_label, best_index = labels[0], None
        for label in labels:
            train_means = {
                name: float(np.mean([variants[label]["rows"][i][name] for i in train]))
                for name in METRIC_NAMES
            }
            base_means = {
                name: float(np.mean([base_rows[i][name] for i in train]))
                for name in METRIC_NAMES
            }
            value = composite(train_means, base_means)
            if best_index is None or value < best_index:
                best_index, best_label = value, label
        folds.append(best_label)
        for name in METRIC_NAMES:
            fold_metrics[name].append(variants[best_label]["rows"][fold][name])

    loo_means = {name: float(np.mean(values)) for name, values in fold_metrics.items()}
    loo_index = composite(loo_means, baseline)
    print(f"\nLOO по (floor, alpha): индекс {loo_index:.4f} vs база 1.0")
    print(f"  выбор по фолдам: {folds}")

    report = {
        "meta": meta,
        "baseline": baseline,
        "variants": {
            label: {"metrics": entry["metrics"], "index": entry["index"]}
            for label, entry in variants.items()
        },
        "loo": {
            "index": loo_index,
            "metrics": loo_means,
            "folds": folds,
        },
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён: {args.out} (готово за {time.time() - started:.0f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
