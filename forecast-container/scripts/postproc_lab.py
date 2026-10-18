"""Лаборатория постобработки: локирование вариантов «непрерывный прогноз -> целые».

Идеи — `docs/222` (очередь 1): суммарный прогноз по серии + распределение
единиц по наиболее вероятным дням (top-K), пороги обнуления, отказ от фиксированного
``gate`` в hurdle (переход к ожидаемому значению ``p * m`` и его калибровка),
smearing-множитель для lgbm-компоненты.

Вход — parquet-срезы из ``scripts/dump_backtest_raw.py`` (сырые компоненты
бленда и факт на 8 полных срезах). Выход — таблица из 5 метрик ТЗ и
композитного индекса по каждому варианту (1.0 = текущая прод-постобработка),
диагностика по сегментам, confusion matrix дневных нулей и калибровка ``p``.

Запуск (из директории forecast-container/, после dump_backtest_raw.py)::

    python scripts/postproc_lab.py                       # logs/raw -> logs/postproc_lab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics import METRIC_WEIGHTS, bias_abs, rmsle, rmse, smape, wape
from app.models.base import postprocess_forecast

METRIC_NAMES = ("smape", "wape", "rmse", "bias", "rmsle")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лаборатория постобработки: варианты целочисленного выхода")
    parser.add_argument("--raw-dir", default="logs/raw", help="Каталог с raw_*.parquet и meta.json")
    parser.add_argument("--out", default="logs/postproc_lab.json", help="JSON-отчёт лаборатории")
    return parser.parse_args(argv)


def load_cutoffs(raw_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    data: list[dict] = []
    for cutoff in meta["cutoffs"]:
        frame = pd.read_parquet(raw_dir / f"raw_{cutoff}.parquet")
        series = (
            frame["warehouse_name"].astype(str) + "|" + frame["article"].astype(str)
        ).to_numpy()
        data.append(
            {
                "cutoff": cutoff,
                "y": frame["quantity"].to_numpy("float64"),
                "series": series,
                "p": frame["p"].to_numpy("float64"),
                "m": frame["m"].to_numpy("float64"),
                "lgbm": frame["lgbm_raw"].to_numpy("float64"),
                "h": frame["h"].to_numpy("int16"),
                "mean_28": frame["mean_28"].to_numpy("float64"),
                "zero_share_90": frame["zero_share_90"].to_numpy("float64"),
                "stock_last": frame["stock_last"].to_numpy("float64"),
                "extra": frame["extra_raw"].to_numpy("float64") if "extra_raw" in frame.columns else None,
            }
        )
    return meta, data


def gated(item: dict, gate: float) -> np.ndarray:
    return np.where(item["p"] >= gate, item["m"], 0.0)


def blend_raw(item: dict, weight: float, gate: float, use_expected: bool, smear: float) -> np.ndarray:
    hurdle = item["p"] * item["m"] if use_expected else gated(item, gate)
    return weight * hurdle + (1.0 - weight) * item["lgbm"] * smear


def apportion(item: dict, raw: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Суммарный прогноз T -> число единиц K -> распределение по дням (наибольшие остатки).

    K = 0 при T < t0; K = 1 при T < t1; иначе K = round(T). Единицы: floor по дню,
    остаток — дням с наибольшей дробной частью (для редких серий это равносильно
    top-K по ``e_h``: floor = 0, единицы получают дни с максимальным прогнозом).
    """
    series_codes = item["series_codes"]
    n_series = item["n_series"]
    n = len(raw)

    totals = np.bincount(series_codes, weights=raw, minlength=n_series)
    units = np.floor(totals + 0.5)
    units = np.where(totals < t1, 1.0, units)
    units = np.where(totals < t0, 0.0, units)

    floors = np.floor(raw)
    floor_sum = np.bincount(series_codes, weights=floors, minlength=n_series)
    extra = np.maximum(units - floor_sum, 0.0)

    order = np.lexsort((-raw, series_codes))
    positions = np.arange(n)
    sorted_codes = series_codes[order]
    first_index = np.full(n_series, n, dtype=np.int64)
    first_index[sorted_codes[::-1]] = positions[::-1]
    rank = positions - first_index[sorted_codes]
    add = rank < extra[sorted_codes]

    result = floors.copy()
    result[order[add]] += 1.0
    return result


def quantize(item: dict, raw: np.ndarray, mode: str, tau: float, t1: float | None) -> np.ndarray:
    if mode == "round":
        return postprocess_forecast(raw, zero_threshold=tau, round_int=True)
    if mode == "unrounded":
        return postprocess_forecast(raw, zero_threshold=tau, round_int=False)
    if mode == "floor1":
        return np.where(np.clip(raw, 0.0, None) >= tau, 1.0, 0.0)
    if mode == "apportion":
        assert t1 is not None
        return apportion(item, np.clip(raw, 0.0, None), tau, t1)
    raise ValueError(f"Неизвестный режим постобработки: {mode!r}")


def evaluate(data: list[dict], predictions: list[np.ndarray]) -> dict:
    means: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for item, pred in zip(data, predictions):
        means["smape"].append(smape(item["y"], pred, item["series"]))
        means["wape"].append(wape(item["y"], pred))
        means["rmse"].append(rmse(item["y"], pred))
        means["bias"].append(bias_abs(item["y"], pred, item["series"]))
        means["rmsle"].append(rmsle(item["y"], pred))
    return {name: float(np.mean(values)) for name, values in means.items()}


def composite(metrics: dict, baseline: dict) -> float:
    return float(
        sum(METRIC_WEIGHTS[name] * metrics[name] / baseline[name] for name in METRIC_NAMES)
    )


def build_variants(meta: dict) -> list[tuple[str, object]]:
    weight = float(meta["weight_hurdle"])
    gate = float(meta["gate"])
    variants: list[tuple[str, object]] = []

    def make(weight: float, gate: float, expected: bool, smear: float, mode: str, tau: float, t1: float | None = None):
        def builder(data: list[dict]) -> list[np.ndarray]:
            return [
                quantize(item, blend_raw(item, weight, gate, expected, smear), mode, tau, t1)
                for item in data
            ]

        return builder

    variants.append(("base (gated 0.5, round)", make(weight, gate, False, 1.0, "round", 0.0)))
    variants.append(("base без округления", make(weight, gate, False, 1.0, "unrounded", 0.0)))
    for smear in (1.05, 1.1, 1.15, 1.25):
        variants.append((f"base smearing x{smear:g}", make(weight, gate, False, smear, "round", 0.0)))
    for g in (0.3, 0.4, 0.6, 0.7):
        variants.append((f"gate={g:g}", make(weight, g, False, 1.0, "round", 0.0)))
    for w in (0.15, 0.3, 0.45):
        variants.append((f"expected p*m, w={w:g}", make(w, gate, True, 1.0, "round", 0.0)))
    for tau in (0.25, 0.5, 0.75, 1.0, 1.25):
        variants.append((f"round, порог {tau:g}", make(weight, gate, False, 1.0, "round", tau)))
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        variants.append((f"floor1 порог {t:g}", make(weight, gate, False, 1.0, "floor1", t)))
    for t0, t1 in ((0.5, 1.5), (0.3, 1.0), (0.7, 1.5), (1.0, 2.0), (0.0, 1.5)):
        variants.append((f"apportion T<{t0:g}->0, <{t1:g}->1", make(weight, gate, False, 1.0, "apportion", t0, t1)))
    for t0, t1 in ((0.5, 1.5), (0.0, 1.5), (0.7, 1.5)):
        variants.append((f"expected + apportion {t0:g}/{t1:g}", make(weight, gate, True, 1.0, "apportion", t0, t1)))
    return variants


def loo_alpha_variant(meta: dict, data: list[dict]) -> tuple[str, list[np.ndarray]]:
    """Калибровка уровня: alpha по остальным срезам (leave-one-out), затем apportion."""
    weight = float(meta["weight_hurdle"])
    gate = float(meta["gate"])
    raws = [blend_raw(item, weight, gate, True, 1.0) for item in data]
    totals_y = np.array([float(item["y"].sum()) for item in data])
    totals_raw = np.array([float(raw.sum()) for raw in raws])
    predictions = []
    for index, raw in enumerate(raws):
        alpha = (totals_y.sum() - totals_y[index]) / max(totals_raw.sum() - totals_raw[index], 1e-9)
        predictions.append(apportion(data[index], np.clip(raw * alpha, 0.0, None), 0.0, 1.5))
    return "expected + LOO alpha + apportion", predictions


def pooled(items_key: str, data: list[dict], predictions: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([item[items_key] for item in data]), np.concatenate(predictions)


def subset_metrics(item: dict, pred: np.ndarray, mask: np.ndarray) -> dict:
    y = item["y"][mask]
    series = item["series"][mask]
    return {
        "smape": smape(y, pred[mask], series),
        "wape": wape(y, pred[mask]),
        "rmse": rmse(y, pred[mask]),
        "bias": bias_abs(y, pred[mask], series),
        "rmsle": rmsle(y, pred[mask]),
    }


def weight_families() -> dict[str, list[tuple[str, object]]]:
    bands = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, np.inf)]
    return {
        "horizon": [
            ("h 1-7", lambda item: item["h"] <= 7),
            ("h 8-28", lambda item: (item["h"] >= 8) & (item["h"] <= 28)),
            ("h 29-64", lambda item: item["h"] >= 29),
        ],
        "horizon_doc": [
            ("h 1-14", lambda item: item["h"] <= 14),
            ("h 15-28", lambda item: (item["h"] >= 15) & (item["h"] <= 28)),
            ("h 29-64", lambda item: item["h"] >= 29),
        ],
        "mean_28": [
            (
                f"m [{low:g}, {high:g})",
                (lambda lo, hi: lambda item: (item["mean_28"] >= lo) & (item["mean_28"] < hi))(low, high),
            )
            for low, high in bands
        ],
    }


def tune_group_weights(meta: dict, data: list[dict], base_predictions: list[np.ndarray]) -> dict:
    """Сегментные веса бленда (leave-one-out по срезам) — честная оценка выгоды.

    Для каждого семейства сегментов (горизонт, класс спроса по mean_28) и каждого
    веса w сетки считается композит по 7 срезам, лучший w применяется к
    отложенному срезу; итог — среднее по 8 отложенным срезам.
    """
    weight = float(meta["weight_hurdle"])
    gate = float(meta["gate"])
    w_grid = tuple(round(0.1 * i, 1) for i in range(11))

    gated_all = [gated(item, gate) for item in data]
    lgbm_all = [item["lgbm"] for item in data]
    raw_by_w = {
        w: [w * g + (1.0 - w) * l for g, l in zip(gated_all, lgbm_all)] for w in w_grid
    }

    report: dict = {"grid": list(w_grid), "families": {}}
    for family, groups in weight_families().items():
        group_report: dict = {}
        # метрики по (группа, срез, w) — считаются один раз
        metrics_cache: dict[tuple[str, int, float], dict] = {}
        masks: dict[tuple[int, str], np.ndarray] = {}
        for index, item in enumerate(data):
            for name, mask_fn in groups:
                mask = mask_fn(item)
                masks[(index, name)] = mask
                if not mask.any():
                    continue
                for w in w_grid:
                    metrics_cache[(name, index, w)] = subset_metrics(
                        item, postprocess_forecast(raw_by_w[w][index], 0.0, True), mask
                    )
        for name, _ in groups:
            folds: list[float] = []
            for fold in range(len(data)):
                train = [i for i in range(len(data)) if i != fold and (name, i, weight) in metrics_cache]
                if not train:
                    folds.append(weight)
                    continue
                baseline_means = {
                    metric: float(np.mean([metrics_cache[(name, i, weight)][metric] for i in train]))
                    for metric in METRIC_NAMES
                }
                best_w, best_index = weight, None
                for w in w_grid:
                    if (name, train[0], w) not in metrics_cache:
                        continue
                    means = {
                        metric: float(np.mean([metrics_cache[(name, i, w)][metric] for i in train]))
                        for metric in METRIC_NAMES
                    }
                    index_value = composite(means, baseline_means)
                    if best_index is None or index_value < best_index:
                        best_index, best_w = index_value, w
                folds.append(best_w)
            group_report[name] = {"chosen_weights": folds, "median_weight": float(np.median(folds))}

        # честный LOO-композит: применяем выбранные веса к отложенному срезу
        fold_metrics: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
        for fold in range(len(data)):
            predictions = base_predictions[fold].copy()
            for name, _ in groups:
                w = group_report[name]["chosen_weights"][fold]
                mask = masks[(fold, name)]
                if mask.any():
                    predictions[mask] = postprocess_forecast(raw_by_w[w][fold], 0.0, True)[mask]
            metrics = subset_metrics(
                data[fold], predictions, np.ones(data[fold]["y"].shape[0], dtype=bool)
            )
            for metric in METRIC_NAMES:
                fold_metrics[metric].append(metrics[metric])
        means = {metric: float(np.mean(values)) for metric, values in fold_metrics.items()}
        baseline_means = evaluate(data, base_predictions)
        report["families"][family] = {
            "groups": group_report,
            "loo_metrics": means,
            "loo_index": composite(means, baseline_means),
        }
        print(f"\nВеса внутри «{family}» (LOO по срезам): индекс {report['families'][family]['loo_index']:.4f} "
              f"vs база 1.0")
        for name, _ in groups:
            entry = group_report[name]
            print(f"  {name:<12} медианный w={entry['median_weight']:g}  по срезам: {entry['chosen_weights']}")
    return report


def smearing_variants(meta: dict, data: list[dict], base_predictions: list[np.ndarray]) -> dict:
    """Duan smearing для log1p-компоненты: E[exp(resid)] (mean/median/trim10), LOO по срезам.

    Для y_hat = exp(z) - 1 корректная оценка E[y] = exp(z_pred) * E[exp(resid)] - 1,
    где resid = log1p(y) - z_pred, а E[exp(resid)] = mean((1 + y) / (1 + lgbm_raw)).
    Множитель считается на 7 срезах и применяется к отложенному (LOO).
    """
    weight = float(meta["weight_hurdle"])
    gate = float(meta["gate"])
    gated_all = [gated(item, gate) for item in data]
    lgbm_all = [item["lgbm"] for item in data]
    y_all = [item["y"] for item in data]
    baseline = evaluate(data, base_predictions)

    report: dict = {}
    for method in ("mean", "median", "trim10"):
        folds: list[float] = []
        predictions: list[np.ndarray] = []
        for fold in range(len(data)):
            ratios = np.concatenate(
                [(y_all[j] + 1.0) / (lgbm_all[j] + 1.0) for j in range(len(data)) if j != fold]
            )
            if method == "mean":
                factor = float(np.mean(ratios))
            elif method == "median":
                factor = float(np.median(ratios))
            else:
                low, high = np.quantile(ratios, [0.05, 0.95])
                factor = float(np.mean(ratios[(ratios >= low) & (ratios <= high)]))
            folds.append(factor)
            lgbm_smeared = (lgbm_all[fold] + 1.0) * factor - 1.0
            raw = weight * gated_all[fold] + (1.0 - weight) * lgbm_smeared
            predictions.append(postprocess_forecast(raw, 0.0, True))
        metrics = evaluate(data, predictions)
        report[method] = {
            "folds": folds,
            "metrics": metrics,
            "index": composite(metrics, baseline),
        }
    return report


def three_way_weights(meta: dict, data: list[dict], base_predictions: list[np.ndarray]) -> dict | None:
    """Пропорции трёх компонент (gated hurdle + lgbm_log + extra) — сетка + LOO по срезам.

    Сетка шагом 0.1, сумма весов = 1 (66 комбинаций); для каждого среза метрики
    считаются один раз, выбор комбинации — на 7 срезах, применение — к отложенному.
    """
    if any(item.get("extra") is None for item in data):
        return None
    weight = float(meta["weight_hurdle"])
    gate = float(meta["gate"])
    w_grid = tuple(round(0.1 * i, 1) for i in range(11))
    combos: list[tuple[float, float, float]] = []
    for a in w_grid:
        for c in w_grid:
            b = round(1.0 - a - c, 1)
            if b < -1e-9:
                continue
            combos.append((a, max(c, 0.0), max(b, 0.0)))

    gated_all = [gated(item, gate) for item in data]
    lgbm_all = [item["lgbm"] for item in data]
    extra_all = [item["extra"] for item in data]
    all_rows = [np.ones(item["y"].shape[0], dtype=bool) for item in data]

    metrics_cache: dict[tuple[float, float, int], dict] = {}
    for a, c, b in combos:
        for index in range(len(data)):
            raw = a * gated_all[index] + b * lgbm_all[index] + c * extra_all[index]
            metrics_cache[(a, c, index)] = subset_metrics(
                data[index], postprocess_forecast(raw, 0.0, True), all_rows[index]
            )

    base_combo = (weight, 0.0)

    def combo_index(combo: tuple[float, float], cutoffs: list[int]) -> float:
        means = {
            metric: float(np.mean([metrics_cache[(combo[0], combo[1], i)][metric] for i in cutoffs]))
            for metric in METRIC_NAMES
        }
        base_means = {
            metric: float(np.mean([metrics_cache[(base_combo[0], base_combo[1], i)][metric] for i in cutoffs]))
            for metric in METRIC_NAMES
        }
        return composite(means, base_means)

    folds: list[tuple[float, float, float]] = []
    for fold in range(len(data)):
        train = [i for i in range(len(data)) if i != fold]
        best, best_index = combos[0], None
        for combo in combos:
            if (combo[0], combo[1], train[0]) not in metrics_cache:
                continue
            value = combo_index((combo[0], combo[1]), train)
            if best_index is None or value < best_index:
                best_index, best = value, combo
        folds.append(best)

    fold_metrics: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    for fold in range(len(data)):
        a, c, b = folds[fold]
        raw = a * gated_all[fold] + b * lgbm_all[fold] + c * extra_all[fold]
        metrics = subset_metrics(
            data[fold], postprocess_forecast(raw, 0.0, True), all_rows[fold]
        )
        for metric in METRIC_NAMES:
            fold_metrics[metric].append(metrics[metric])
    means = {metric: float(np.mean(values)) for metric, values in fold_metrics.items()}
    top_full = sorted(
        ((combo_index((combo[0], combo[1]), list(range(len(data)))), combo) for combo in combos),
        key=lambda pair: pair[0],
    )[:5]
    report = {
        "grid_combos": [list(combo) for combo in combos],
        "folds": [list(combo) for combo in folds],
        "loo_metrics": means,
        "loo_index": composite(means, evaluate(data, base_predictions)),
        "top_full_data": [
            {"index": round(index, 4), "hurdle": combo[0], "lgbm": combo[2], "extra": combo[1]}
            for index, combo in top_full
        ],
    }
    print(f"\nТри компоненты (LOO по срезам): индекс {report['loo_index']:.4f} vs база 1.0")
    print(f"  комбинации по фолдам (hurdle / lgbm / extra): {report['folds']}")
    print("  лучшие на всех 8 срезах (для справки, оптимистично):")
    for row in report["top_full_data"]:
        print(f"    hurdle {row['hurdle']:g} + lgbm {row['lgbm']:g} + extra {row['extra']:g}: индекс {row['index']}")
    return report


def confusion_block(data: list[dict], predictions: list[np.ndarray]) -> dict:
    y, pred = pooled("y", data, predictions)
    zero_zero = int(np.sum((y == 0) & (pred == 0)))
    zero_pos = int(np.sum((y == 0) & (pred > 0)))
    pos_zero = int(np.sum((y > 0) & (pred == 0)))
    pos_pos = int(np.sum((y > 0) & (pred > 0)))
    total = zero_zero + zero_pos + pos_zero + pos_pos
    return {
        "fact0_pred0": zero_zero,
        "fact0_pred_pos": zero_pos,
        "fact_pos_pred0": pos_zero,
        "fact_pos_pred_pos": pos_pos,
        "share": {k: v / total for k, v in zip(("00", "0+", "+0", "++"), (zero_zero, zero_pos, pos_zero, pos_pos))},
    }


def segment_block(data: list[dict], predictions: list[np.ndarray], label: str) -> list[dict]:
    bands = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, np.inf)]
    rows: list[dict] = []
    for low, high in bands:
        metrics_parts: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        for item, pred in zip(data, predictions):
            mask = (item["mean_28"] >= low) & (item["mean_28"] < high)
            if not mask.any():
                continue
            metrics_parts["smape"].append(smape(item["y"][mask], pred[mask], item["series"][mask]))
            metrics_parts["wape"].append(wape(item["y"][mask], pred[mask]))
            metrics_parts["bias"].append(bias_abs(item["y"][mask], pred[mask], item["series"][mask]))
        if not metrics_parts["smape"]:
            continue
        rows.append(
            {
                "variant": label,
                "mean_28_band": f"[{low:g}, {high:g})",
                "smape": float(np.mean(metrics_parts["smape"])),
                "wape": float(np.mean(metrics_parts["wape"])),
                "bias": float(np.mean(metrics_parts["bias"])),
            }
        )
    return rows


def calibration_block(data: list[dict]) -> list[dict]:
    p = np.concatenate([item["p"] for item in data])
    y = np.concatenate([item["y"] for item in data])
    edges = np.quantile(p, np.linspace(0.0, 1.0, 11))
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p <= high if high == edges[-1] else p < high)
        if not mask.any():
            continue
        rows.append(
            {
                "p_range": f"[{low:.3f}, {high:.3f}]",
                "mean_p": float(p[mask].mean()),
                "sale_day_share": float((y[mask] > 0).mean()),
                "n_days": int(mask.sum()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    meta, data = load_cutoffs(raw_dir)
    for item in data:
        codes, uniques = pd.factorize(item["series"], sort=False)
        item["series_codes"] = codes
        item["n_series"] = len(uniques)
    total_rows = sum(len(item["y"]) for item in data)
    print(f"Срезов: {len(data)}, строк: {total_rows:,}, вес hurdle {meta['weight_hurdle']:g}, gate {meta['gate']:g}")

    variants = build_variants(meta)
    results: list[dict] = []
    predictions_by_label: dict[str, list[np.ndarray]] = {}
    for label, builder in variants:
        predictions = builder(data)  # type: ignore[operator]
        metrics = evaluate(data, predictions)
        results.append({"variant": label, "metrics": metrics})
        predictions_by_label[label] = predictions

    alpha_label, alpha_predictions = loo_alpha_variant(meta, data)
    results.append({"variant": alpha_label, "metrics": evaluate(data, alpha_predictions)})
    predictions_by_label[alpha_label] = alpha_predictions

    baseline = results[0]["metrics"]
    for entry in results:
        entry["index"] = composite(entry["metrics"], baseline)
    results.sort(key=lambda entry: entry["index"])

    print(f"\n{'вариант':<34} {'SMAPE':>7} {'WAPE':>7} {'RMSE':>7} {'Bias':>7} {'RMSLE':>7} {'индекс':>7}")
    for entry in results:
        m = entry["metrics"]
        print(
            f"{entry['variant']:<34} {m['smape']:>7.2f} {m['wape']:>7.2f} {m['rmse']:>7.4f} "
            f"{m['bias']:>7.2f} {m['rmsle']:>7.4f} {entry['index']:>7.4f}"
        )

    best_label = results[0]["variant"]
    best_predictions = predictions_by_label[best_label]
    base_predictions = predictions_by_label["base (gated 0.5, round)"]

    weight_report = tune_group_weights(meta, data, base_predictions)

    smearing_report = smearing_variants(meta, data, base_predictions)
    for label, entry in smearing_report.items():
        m = entry["metrics"]
        print(
            f"smearing {label} (LOO): SMAPE {m['smape']:.2f} WAPE {m['wape']:.2f} RMSE {m['rmse']:.4f} "
            f"Bias {m['bias']:.2f} RMSLE {m['rmsle']:.4f} индекс {entry['index']:.4f}"
        )

    three_way_report = three_way_weights(meta, data, base_predictions)

    segments = segment_block(data, base_predictions, "base") + segment_block(
        data, best_predictions, best_label
    )
    print(f"\nСегменты (base vs лучший — {best_label}):")
    for row in segments:
        print(
            f"  mean_28 {row['mean_28_band']:<12} {row['variant']:<28} "
            f"SMAPE={row['smape']:6.2f} WAPE={row['wape']:6.2f} Bias={row['bias']:6.2f}"
        )

    confusion = {
        "base": confusion_block(data, base_predictions),
        best_label: confusion_block(data, best_predictions),
    }
    for label, block in confusion.items():
        share = block["share"]
        print(
            f"\nДневные нули [{label}]: 0/0 {share['00']:.3f}, 0/+ {share['0+']:.3f}, "
            f"+/0 {share['+0']:.3f}, +/+ {share['++']:.3f}"
        )

    calibration = calibration_block(data)
    print("\nКалибровка вероятности p (децили):")
    for row in calibration:
        print(
            f"  p {row['p_range']:<16} mean_p={row['mean_p']:.3f}  доля продажных дней={row['sale_day_share']:.3f}"
        )

    report = {
        "meta": meta,
        "rows": total_rows,
        "variants": results,
        "best": best_label,
        "segments": segments,
        "confusion": confusion,
        "calibration": calibration,
        "weight_tuning": weight_report,
        "smearing": smearing_report,
        "three_way": three_way_report,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
