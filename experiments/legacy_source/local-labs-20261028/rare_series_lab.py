"""Очередь 3 (docs/222): редкие серии — классы спроса, TSB/SBA и правило P_any_64.

Офлайн-эксперимент на 8 полных срезах бэктеста (компоненты бленда — из
``scripts/dump_backtest_raw.py``). Проверяются три идеи внешнего аудита:

1. **Классы спроса** (п. 3.1): ADI / CV2 по истории 90 дней ->
   regular / erratic / intermittent / lumpy (+ спящие) — диагностика метрик
   текущей базы по классам.
2. **TSB и SBA** (п. 3.3): классические методы интермиттирующего спроса на
   дневной истории 180 дней; проверяются как индикатор «серии нужна хотя бы
   одна единица» (64 * TSB >= t) для серий с нулевым прогнозом базы.
3. **P_any_64** (п. 3.2/Шаг 4): вероятность хотя бы одной продажи за 64 дня
   (из откалиброванных p классификатора hurdle) — если база ставит все нули,
   а P_any_64 >= theta, добавляется 1 единица в день с максимальным p (или e=p*m).

Пороги theta/t подбираются leave-one-out по срезам (выбор на 7, проверка на
отложенном), итог сравнивается с базой по композитному индексу ТЗ.

Запуск (из директории forecast-container/, после dump_backtest_raw.py)::

    python scripts/rare_series_lab.py                # logs/raw -> logs/rare_lab.json
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

from app.features import LOOKBACK_DAYS, build_panel
from app.metrics import METRIC_WEIGHTS, bias_abs, rmsle, rmse, smape, wape
from app.models.base import postprocess_forecast

METRIC_NAMES = ("smape", "wape", "rmse", "bias", "rmsle")
P_ANY_GRID = (0.3, 0.5, 0.7, 0.8, 0.9)
TSB_GRID = (0.5, 1.0, 1.5)
TSB_PARAMS = ((0.1, 0.1), (0.2, 0.1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Очередь 3: редкие серии, TSB/SBA, P_any_64")
    parser.add_argument("--raw-dir", default="logs/raw", help="Каталог с raw_*.parquet и meta.json")
    parser.add_argument("--sales", default="../data/sales_dataset_hakaton_train.csv")
    parser.add_argument("--stock", default="../data/stock_dataset_hakaton_train.csv")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--out", default="logs/rare_lab.json")
    return parser.parse_args(argv)


def tsb_forecast(matrix: np.ndarray, p0: np.ndarray, z0: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """TSB по всем сериям сразу: вероятность продажи p и средний размер z обновляются ежедневно.

    ``matrix`` — (series, days) количества; ``p0``/``z0`` — стартовые оценки.
    Возвращает прогноз ``p * z`` на каждый день (плоский по горизонту).
    """
    p = p0.copy()
    z = z0.copy()
    for step in range(matrix.shape[1]):
        sold = matrix[:, step] > 0.0
        p = np.where(sold, p + alpha * (1.0 - p), p - alpha * p)
        z = np.where(sold, z + beta * (matrix[:, step] - z), z)
    return p * z


def sba_forecast(matrix: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Croston-SBA (1/2 сглаженного интервала): прогноз = z / (2 * q)."""
    n_series, n_days = matrix.shape
    p_size = np.zeros(n_series)
    q_interval = np.ones(n_series) * 2.0
    last_sale = np.full(n_series, -1, dtype="float64")
    seen = np.zeros(n_series, dtype=bool)
    for step in range(n_days):
        sold = matrix[:, step] > 0.0
        if sold.any():
            sizes = matrix[sold, step]
            p_size[sold] = np.where(seen[sold], p_size[sold] + alpha * (sizes - p_size[sold]), sizes)
            intervals = step - last_sale[sold]
            q_interval[sold] = np.where(
                seen[sold], q_interval[sold] + alpha * (intervals - q_interval[sold]), intervals
            )
            last_sale[sold] = step
            seen[sold] = True
    result = np.zeros(n_series)
    valid = seen & (q_interval > 0)
    result[valid] = p_size[valid] / (2.0 * q_interval[valid])
    return result


def demand_classes(adi: np.ndarray, cv2: np.ndarray, n_sale_90: np.ndarray) -> np.ndarray:
    """Классификация серий по ADI/CV2 (docs/222, п. 3.1)."""
    classes = np.full(len(adi), "sleeping", dtype=object)
    known = n_sale_90 >= 2
    low_adi = adi < 1.32
    low_cv2 = cv2 < 0.49
    classes[known & low_adi & low_cv2] = "regular"
    classes[known & low_adi & ~low_cv2] = "erratic"
    classes[known & ~low_adi & low_cv2] = "intermittent"
    classes[known & ~low_adi & ~low_cv2] = "lumpy"
    return classes


def series_stats(panel: pd.DataFrame, cutoff: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """Статистики серий на дату отсечки по окну lookback: ADI, CV2, mean_28, zero_share_90, TSB/SBA."""
    window = panel[(panel["date"] > cutoff - pd.Timedelta(days=lookback)) & (panel["date"] <= cutoff)]
    keys = window[["warehouse_name", "article"]].drop_duplicates()
    window = window.sort_values("date")
    pivot = window.pivot_table(
        index=["warehouse_name", "article"], columns="date", values="quantity", aggfunc="sum", fill_value=0.0
    )
    matrix = pivot.to_numpy(dtype="float64")
    days = matrix.shape[1]

    positive = matrix > 0.0
    n_sale = positive.sum(axis=1)
    n_sale_90 = positive[:, -90:].sum(axis=1) if days >= 90 else n_sale.copy()
    n_sale_28 = positive[:, -28:].sum(axis=1) if days >= 28 else n_sale.copy()
    mean_28 = matrix[:, -28:].sum(axis=1) / 28.0 if days >= 28 else matrix.mean(axis=1)
    zero_share_90 = 1.0 - n_sale_90 / 90.0
    adi = 90.0 / np.maximum(n_sale_90, 1)
    mean_pos = np.divide(
        matrix[:, -90:].sum(axis=1),
        np.maximum(n_sale_90, 1),
    )
    sumsq = (np.where(positive[:, -90:], matrix[:, -90:], 0.0) ** 2).sum(axis=1)
    var_pos = np.divide(sumsq, np.maximum(n_sale_90, 1)) - mean_pos**2
    cv2 = np.where(mean_pos > 0, np.maximum(var_pos, 0.0) / np.maximum(mean_pos**2, 1e-12), 0.0)

    p0 = np.clip(n_sale_90 / 90.0, 0.01, 0.99)
    z0 = np.where(n_sale >= 1, matrix.sum(axis=1) / np.maximum(n_sale, 1), 0.0)
    stats = keys.copy()
    stats["mean_28"] = mean_28
    stats["zero_share_90"] = zero_share_90
    stats["n_sale_days_90"] = n_sale_90
    stats["adi_90"] = adi
    stats["cv2_90"] = cv2
    stats["class"] = demand_classes(adi, cv2, n_sale_90)
    for alpha, beta in TSB_PARAMS:
        stats[f"tsb_{alpha:g}_{beta:g}"] = tsb_forecast(matrix, p0, z0, alpha, beta)
    stats["sba"] = sba_forecast(matrix)
    return stats


def subset_metrics(y: np.ndarray, pred: np.ndarray, series: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "smape": smape(y[mask], pred[mask], series[mask]),
        "wape": wape(y[mask], pred[mask]),
        "rmse": rmse(y[mask], pred[mask]),
        "bias": bias_abs(y[mask], pred[mask], series[mask]),
        "rmsle": rmsle(y[mask], pred[mask]),
    }


def composite(metrics: dict, baseline: dict) -> float:
    return float(sum(METRIC_WEIGHTS[name] * metrics[name] / baseline[name] for name in METRIC_NAMES))


def load_cutoff_data(raw_dir: Path, cutoff: str, stats: pd.DataFrame, weight: float, gate: float) -> dict:
    """Компоненты среза + серии + базовый прогноз + сгруппированные структуры."""
    frame = pd.read_parquet(raw_dir / f"raw_{cutoff}.parquet")
    frame = frame.merge(stats, on=["warehouse_name", "article"], how="left")
    series = (frame["warehouse_name"].astype(str) + "|" + frame["article"].astype(str)).to_numpy()
    codes, uniques = pd.factorize(series, sort=False)

    p = frame["p"].to_numpy("float64")
    m = frame["m"].to_numpy("float64")
    lgbm = frame["lgbm_raw"].to_numpy("float64")
    raw = weight * np.where(p >= gate, m, 0.0) + (1.0 - weight) * lgbm
    base_pred = postprocess_forecast(raw, 0.0, True)

    n_series = len(uniques)
    log_one_minus = np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12))
    p_any = -np.expm1(np.bincount(codes, weights=log_one_minus, minlength=n_series))
    k_base = np.bincount(codes, weights=base_pred, minlength=n_series)

    order = np.lexsort((-p, codes))
    first = np.full(n_series, len(p), dtype=np.int64)
    first[codes[order][::-1]] = np.arange(len(p))[::-1]
    best_p_row = order[first]

    first_row = np.zeros(n_series, dtype=np.int64)
    first_row[codes[::-1]] = np.arange(len(codes), dtype=np.int64)[::-1]

    e = p * m
    order_e = np.lexsort((-e, codes))
    first_e = np.full(n_series, len(e), dtype=np.int64)
    first_e[codes[order_e][::-1]] = np.arange(len(e))[::-1]
    best_e_row = order_e[first_e]

    tsb_columns = [f"tsb_{a:g}_{b:g}" for a, b in TSB_PARAMS]
    return {
        "cutoff": cutoff,
        "y": frame["quantity"].to_numpy("float64"),
        "series": series,
        "codes": codes,
        "n_series": n_series,
        "base": base_pred,
        "p_any": p_any,
        "k_base": k_base,
        "best_p_row": best_p_row,
        "best_e_row": best_e_row,
        "class": frame["class"].fillna("sleeping").to_numpy(),
        "tsb_series": frame[tsb_columns].to_numpy("float64")[first_row],
        "sba_series": frame["sba"].to_numpy("float64")[first_row],
        "mean_28": frame["mean_28"].to_numpy("float64"),
    }


def add_unit(item: dict, rows: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Копия базового прогноза + 1 единица в указанный день для активных серий."""
    pred = item["base"].copy()
    pred[rows[active]] += 1.0
    return pred


def per_cutoff_metrics(data: list[dict], predictions: list[np.ndarray]) -> list[dict]:
    rows = []
    for item, pred in zip(data, predictions):
        rows.append(subset_metrics(item["y"], pred, item["series"], np.ones(len(item["y"]), dtype=bool)))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    raw_dir = Path(args.raw_dir)
    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    weight = float(meta["weight_hurdle"])
    gate = float(meta["gate"])
    cutoffs = meta["cutoffs"]
    print(f"Срезов: {len(cutoffs)}, вес hurdle {weight:g}, gate {gate:g}", flush=True)

    print("Загрузка данных...", flush=True)
    sales = pd.read_csv(args.sales, parse_dates=["date"])
    stock = pd.read_csv(args.stock, parse_dates=["date"])
    panel = build_panel(sales, stock)

    data: list[dict] = []
    for index, cutoff in enumerate(cutoffs, start=1):
        stats = series_stats(panel, pd.Timestamp(cutoff), args.lookback)
        stats = stats.drop(columns=["mean_28", "zero_share_90", "n_sale_days_90"], errors="ignore")
        item = load_cutoff_data(raw_dir, cutoff, stats, weight, gate)
        data.append(item)
        sizes = pd.Series(item["class"]).value_counts().to_dict()
        print(f"  [{index}/{len(cutoffs)}] {cutoff}: {item['n_series']} серий, классы {sizes}", flush=True)

    baseline_rows = per_cutoff_metrics(data, [item["base"] for item in data])
    baseline = {name: float(np.mean([row[name] for row in baseline_rows])) for name in METRIC_NAMES}
    print("\nБаза:", {k: round(v, 4) for k, v in baseline.items()})

    print("\nДиагностика по классам спроса (база, SMAPE/WAPE/Bias):")
    class_report: dict[str, dict] = {}
    for klass in ("regular", "erratic", "intermittent", "lumpy", "sleeping"):
        parts: dict[str, list[float]] = {name: [] for name in ("smape", "wape", "bias")}
        n_series_list: list[int] = []
        for item in data:
            mask = item["class"] == klass
            if not mask.any():
                continue
            metrics = subset_metrics(item["y"], item["base"], item["series"], mask)
            for name in parts:
                parts[name].append(metrics[name])
            n_series_list.append(int(len(np.unique(item["codes"][mask]))))
        if not parts["smape"]:
            continue
        class_report[klass] = {
            "smape": float(np.mean(parts["smape"])),
            "wape": float(np.mean(parts["wape"])),
            "bias": float(np.mean(parts["bias"])),
            "series_per_cutoff": float(np.mean(n_series_list)),
        }
        row = class_report[klass]
        print(
            f"  {klass:<13} серий/срез {row['series_per_cutoff']:7.0f}  "
            f"SMAPE={row['smape']:6.2f} WAPE={row['wape']:6.2f} Bias={row['bias']:6.2f}"
        )

    rule_results: dict[str, dict] = {}

    def evaluate_rule(name: str, predictions: list[np.ndarray]) -> None:
        rows = per_cutoff_metrics(data, predictions)
        metrics = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_NAMES}
        index = composite(metrics, baseline)
        rule_results[name] = {"metrics": metrics, "index": index, "rows": rows}
        print(
            f"  {name:<28} SMAPE {metrics['smape']:.2f} WAPE {metrics['wape']:.2f} "
            f"RMSE {metrics['rmse']:.4f} Bias {metrics['bias']:.2f} RMSLE {metrics['rmsle']:.4f} "
            f"индекс {index:.4f}"
        )

    print("\nПравило P_any_64 (1 единица в лучший день для серий с нулевой базой):")
    for theta in P_ANY_GRID:
        for day_key, day_name in (("best_p_row", "p"), ("best_e_row", "e")):
            predictions = []
            for item in data:
                active = (item["k_base"] == 0) & (item["p_any"] >= theta)
                predictions.append(add_unit(item, item[day_key], active))
            evaluate_rule(f"P_any>={theta:g} (день по {day_name})", predictions)

    print("\nПравило TSB/SBA (1 единица, если ожидание за 64 дня выше порога):")
    for param_index, (alpha, beta) in enumerate(TSB_PARAMS):
        for threshold in TSB_GRID:
            predictions = []
            for item in data:
                expected = item["tsb_series"][:, param_index] * 64.0
                active = (item["k_base"] == 0) & (expected >= threshold)
                predictions.append(add_unit(item, item["best_p_row"], active))
            evaluate_rule(f"TSB {alpha:g}/{beta:g} >= {threshold:g}", predictions)
    for threshold in TSB_GRID:
        predictions = []
        for item in data:
            active = (item["k_base"] == 0) & (item["sba_series"] * 64.0 >= threshold)
            predictions.append(add_unit(item, item["best_p_row"], active))
        evaluate_rule(f"SBA >= {threshold:g}", predictions)

    print("\nLOO-подбор порогов по семействам правил (выбор на 7 срезах, проверка на отложенном):")
    loo_report: dict[str, dict] = {}
    families = {
        "P_any (день по p)": [f"P_any>={t:g} (день по p)" for t in P_ANY_GRID],
        "P_any (день по e)": [f"P_any>={t:g} (день по e)" for t in P_ANY_GRID],
        "TSB 0.1/0.1": [f"TSB 0.1/0.1 >= {t:g}" for t in TSB_GRID],
        "TSB 0.2/0.1": [f"TSB 0.2/0.1 >= {t:g}" for t in TSB_GRID],
        "SBA": [f"SBA >= {t:g}" for t in TSB_GRID],
    }
    for family, labels in families.items():
        fold_metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        chosen: list[str] = []
        for fold in range(len(data)):
            train = [i for i in range(len(data)) if i != fold]
            best_label, best_index = labels[0], None
            for label in labels:
                train_metrics = {
                    name: float(np.mean([rule_results[label]["rows"][i][name] for i in train]))
                    for name in METRIC_NAMES
                }
                base_train = {
                    name: float(np.mean([baseline_rows[i][name] for i in train]))
                    for name in METRIC_NAMES
                }
                value = composite(train_metrics, base_train)
                if best_index is None or value < best_index:
                    best_index, best_label = value, label
            chosen.append(best_label)
            for name in METRIC_NAMES:
                fold_metrics[name].append(rule_results[best_label]["rows"][fold][name])
        means = {name: float(np.mean(values)) for name, values in fold_metrics.items()}
        loo_report[family] = {
            "chosen_per_fold": chosen,
            "loo_metrics": means,
            "loo_index": composite(means, baseline),
        }
        print(f"  {family:<20} индекс {loo_report[family]['loo_index']:.4f}  выбор по фолдам: {chosen}")

    report = {
        "meta": meta,
        "baseline": baseline,
        "demand_classes": class_report,
        "rules": {name: {"metrics": entry["metrics"], "index": entry["index"]} for name, entry in rule_results.items()},
        "loo": loo_report,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён: {args.out} (за {time.time() - started:.0f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
