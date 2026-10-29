"""FastAPI-приложение: прогноз продаж по складам и товарам на 9 недель.

Эндпоинты:
    GET  /health                          — проверка живости сервиса
    POST /predict (multipart: sales, stock) — прогноз по загруженным данным
    GET  /predict?article=&warehouse=     — срез кэшированного прогноза
    GET  /predict/all?format=csv|json     — весь прогноз целиком

CORS открыт для браузерных клиентов: по умолчанию все origins (``*``),
список разрешённых задаётся переменной окружения ``CORS_ORIGINS`` через запятую.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .predictor import ForecastNotReadyError, ModelNotReadyError, Predictor
from .csv_io import CsvHeaderError, read_csv_checked
from .validators import validate_sales_csv, validate_stock_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast_api")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))
CORS_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "*").split(",") if origin.strip()]
CSV_CHUNK_ROWS = 200_000
MAX_CONCURRENT_PREDICTIONS = int(os.environ.get("MAX_CONCURRENT_PREDICTIONS", "1"))
if MAX_CONCURRENT_PREDICTIONS < 1:
    raise ValueError("MAX_CONCURRENT_PREDICTIONS must be positive")
# Includes CSV parsing and validation, which also allocate substantial memory.
prediction_slots = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)


@asynccontextmanager
async def lifespan(application):
    # A semaphore belongs to its event loop. Recreate it for a new app lifecycle
    # (also permits multiple sequential TestClient/server runs in one process).
    global prediction_slots
    prediction_slots = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
    yield


app = FastAPI(
    title="Hackathon Forecast API",
    description="Прогноз недельных сумм продаж на 9 полных недель со следующего понедельника",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor(MODEL_DIR)
try:
    predictor.load()
    logger.info("Модель загружена из %s", MODEL_DIR)
except ModelNotReadyError as exc:
    logger.warning("Модель пока не готова: %s", exc)


def _error(status_code: int, detail: str, code: str = "VALIDATION_ERROR") -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "code": code})


def _iter_csv(df: pd.DataFrame, chunk_rows: int = CSV_CHUNK_ROWS) -> Iterator[str]:
    yield ",".join(df.columns) + "\n"
    for start in range(0, len(df), chunk_rows):
        yield df.iloc[start : start + chunk_rows].to_csv(index=False, header=False)


def _csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    return StreamingResponse(
        _iter_csv(df),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _iter_json(df: pd.DataFrame, chunk_rows: int = CSV_CHUNK_ROWS) -> Iterator[str]:
    yield "["
    first = True
    for start in range(0, len(df), chunk_rows):
        chunk_json = df.iloc[start : start + chunk_rows].to_json(orient="records")
        chunk_json = chunk_json[1:-1]
        if not chunk_json:
            continue
        if not first:
            yield ","
        yield chunk_json
        first = False
    yield "]"


def _read_upload_sync(upload: UploadFile, name: str):
    """Parse the spooled upload in a worker, without copying it to one bytes object."""
    try:
        upload.file.seek(0)
        df = read_csv_checked(upload.file)
    except CsvHeaderError as exc:
        return None, _error(422, f"Неверный заголовок CSV ({name}): {exc}")
    except pd.errors.EmptyDataError:
        return None, _error(400, f"Пустой файл: {name}")
    except Exception as exc:
        return None, _error(400, f"Ошибка парсинга CSV ({name}): {exc}")
    if df.empty:
        return None, _error(400, f"Пустой файл: {name}")
    return df, None


async def _read_upload(upload: UploadFile, name: str):
    return await run_in_threadpool(_read_upload_sync, upload, name)


def _prepare_inputs_sync(sales: UploadFile, stock: UploadFile):
    df_sales, error = _read_upload_sync(sales, "sales")
    if error is not None:
        return None, None, error
    df_stock, error = _read_upload_sync(stock, "stock")
    if error is not None:
        return None, None, error
    errors = validate_sales_csv(df_sales) + validate_stock_csv(df_stock)
    return df_sales, df_stock, _error(422, "; ".join(errors)) if errors else None


@app.exception_handler(RequestValidationError)
async def request_validation_error(request, exc):
    return _error(422, str(exc))


@app.get("/")
async def root():
    return {
        "service": "Hackathon Forecast API",
        "endpoints": [
            "GET /health",
            "GET /ready",
            "POST /predict",
            "GET /predict?article=&warehouse=",
            "GET /predict/all?format=csv|json",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor.is_loaded}


@app.get("/ready")
async def ready():
    """Readiness fails when the process is alive but its model is unavailable."""
    if not predictor.is_loaded:
        return _error(503, "Модель не загружена", code="MODEL_NOT_READY")
    return {"status": "ready", "model_loaded": True}


@app.post("/predict")
async def predict_post(
    sales: UploadFile = File(...),
    stock: UploadFile = File(...),
):
    """Принимает sales + stock и возвращает week,warehouse_name,article,quantity на 9 недель."""
    async with prediction_slots:
        df_sales, df_stock, error = await run_in_threadpool(_prepare_inputs_sync, sales, stock)
        if error is not None:
            return error
        try:
            forecast = await run_in_threadpool(predictor.predict, df_sales, df_stock)
        except ModelNotReadyError as exc:
            return _error(503, str(exc), code="MODEL_NOT_READY")
        except ValueError as exc:
            return _error(400, str(exc))

    return _csv_response(forecast, "forecast.csv")


@app.get("/predict")
async def predict_get(
    article: Optional[str] = Query(None, description="Артикул товара"),
    warehouse: Optional[str] = Query(None, description="Наименование склада"),
):
    """Прогноз по конкретному артикулу и/или складу (из кэша последнего POST)."""
    if not article and not warehouse:
        return _error(400, "Укажите хотя бы один параметр: article или warehouse")

    try:
        forecast = predictor.get_cached_forecast()
    except ForecastNotReadyError as exc:
        return _error(400, str(exc), code="FORECAST_NOT_READY")

    mask = pd.Series(True, index=forecast.index)
    if article:
        mask &= forecast["article"] == article
    if warehouse:
        mask &= forecast["warehouse_name"] == warehouse

    return _csv_response(forecast[mask], "filtered_forecast.csv")


@app.get("/predict/all")
async def predict_all(format: str = Query("csv", description="Формат ответа: csv или json")):
    """Весь последний прогноз, включая требуемые пары без совместной истории."""
    if format not in ("csv", "json"):
        return _error(400, "Параметр format должен быть 'csv' или 'json'")

    try:
        forecast = predictor.get_cached_forecast()
    except ForecastNotReadyError as exc:
        return _error(400, str(exc), code="FORECAST_NOT_READY")

    if format == "csv":
        return _csv_response(forecast, "full_forecast.csv")

    return StreamingResponse(_iter_json(forecast), media_type="application/json")
