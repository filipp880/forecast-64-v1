#!/usr/bin/env python3
"""Готовит сэмплы данных для локальных тестов API (tests/test_api.sh).

Из полных train-CSV (по умолчанию ../data) вырезаются:
  * sample_sales_6mo.csv / sample_stock_6mo.csv — последние DAYS дней
    по первым ARTICLES артикулам (≥180 дней истории, чтобы POST /predict
    прошёл валидацию по глубине истории);
  * sample_sales.csv / sample_stock.csv — первые 100 строк (пример формата из ТЗ).

Запуск (из директории forecast-container/):
    python scripts/make_samples.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    docker_dir = Path(__file__).resolve().parents[1]
    repo_dir = docker_dir.parent

    parser = argparse.ArgumentParser(description="Сэмплы данных для тестов API")
    parser.add_argument(
        "--sales",
        default=repo_dir / "data" / "sales_dataset_hakaton_train.csv",
        help="Полный CSV продаж",
    )
    parser.add_argument(
        "--stock",
        default=repo_dir / "data" / "stock_dataset_hakaton_train.csv",
        help="Полный CSV остатков",
    )
    parser.add_argument("--out-dir", default=docker_dir / "data", help="Куда положить сэмплы")
    parser.add_argument("--days", type=int, default=184, help="Глубина истории в сэмпле (>=180)")
    parser.add_argument("--articles", type=int, default=50, help="Сколько артикулов оставить")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Чтение продаж...")
    sales = pd.read_csv(args.sales, parse_dates=["date"])
    print("Чтение остатков...")
    stock = pd.read_csv(args.stock, parse_dates=["date"])

    # Примеры формата — первые 100 строк (как в ТЗ)
    sales.head(100).to_csv(out_dir / "sample_sales.csv", index=False)
    stock.head(100).to_csv(out_dir / "sample_stock.csv", index=False)

    # «Полугодовые» сэмплы: последние N дней по первым артикулам
    articles = sorted(sales["article"].unique())[: args.articles]
    last_date = sales["date"].max()
    first_date = last_date - pd.Timedelta(days=args.days - 1)

    sales_sample = sales[(sales["date"] >= first_date) & (sales["article"].isin(articles))].copy()
    stock_sample = stock[(stock["date"] >= first_date) & (stock["article"].isin(articles))].copy()

    negative_stock = int((stock_sample["stock"] < 0).sum())
    if negative_stock:
        print(
            f"  в сэмпле остатков отброшено строк с отрицательным stock: {negative_stock} "
            "(такие строки есть в train-данных, но POST /predict их отклоняет по ТЗ)"
        )
        stock_sample = stock_sample[stock_sample["stock"] >= 0]

    sales_sample.to_csv(out_dir / "sample_sales_6mo.csv", index=False)
    stock_sample.to_csv(out_dir / "sample_stock_6mo.csv", index=False)

    print(
        f"sample_sales_6mo.csv: {len(sales_sample):,} строк, "
        f"{sales_sample['article'].nunique()} артикулов, "
        f"{sales_sample['warehouse_name'].nunique()} складов, "
        f"{sales_sample['date'].min().date()} .. {sales_sample['date'].max().date()}"
    )
    print(f"sample_stock_6mo.csv: {len(stock_sample):,} строк")


if __name__ == "__main__":
    main()
