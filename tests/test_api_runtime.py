"""Multipart, readiness, concurrency and real-artifact serving regressions."""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
import sys
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.main as main
from app.predictor import Predictor
from app.csv_io import CsvHeaderError, read_csv_checked


def input_files():
    dates = ["2026-01-28", "2026-07-26"]
    sales = pd.DataFrame({
        "date": dates, "composite_division": "d", "cities_name": "c",
        "warehouse_name": "001", "category_lvl_1": "c1", "category_lvl_2": "c2",
        "category_lvl_3": "c3", "article": "002", "characteristic": "x",
        "quantity": [1.0, 2.0], "sales_sum": 1.0, "discount_sum": float("nan"),
    })
    stock = pd.DataFrame({"date": dates, "warehouse_name": "001", "char_article": "002", "stock": -1.0})
    return sales.to_csv(index=False).encode(), stock.to_csv(index=False).encode()


class StubPredictor:
    is_loaded = True

    def predict(self, sales, stock):
        assert sales.warehouse_name.iloc[0] == "001"
        assert sales.article.iloc[0] == "002"
        self.forecast = pd.DataFrame({
            "week": ["2026-07-27"], "warehouse_name": ["001"], "article": ["002"], "quantity": [1.0],
        })
        return self.forecast


@pytest.mark.parametrize("quoted,leading_blanks", [(False, False), (True, False), (True, True)])
def test_multipart_duplicate_header_is_422_before_pandas_mangles_it(monkeypatch, quoted, leading_blanks):
    monkeypatch.setattr(main, "predictor", StubPredictor())
    sales, stock = input_files()
    lines = sales.decode().splitlines()
    lines[0] += ',"quantity"' if quoted else ",quantity"
    lines[1:] = [line + ",999" for line in lines[1:]]
    sales = ("\ufeff" + ("\n  \n" if leading_blanks else "") + "\n".join(lines)).encode("utf-8")
    with TestClient(main.app) as client:
        response = client.post("/predict", files={"sales": ("s.csv", sales), "stock": ("st.csv", stock)})
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_readiness_tracks_model_but_health_remains_liveness(monkeypatch):
    predictor = StubPredictor()
    predictor.is_loaded = False
    monkeypatch.setattr(main, "predictor", predictor)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["model_loaded"] is False
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["code"] == "MODEL_NOT_READY"
        predictor.is_loaded = True
        assert client.get("/ready").status_code == 200


def test_text_csv_bom_and_blank_lines_do_not_hide_duplicate_header():
    with pytest.raises(CsvHeaderError):
        read_csv_checked(io.StringIO('\ufeff\n  \n"quantity",quantity\n1,999\n'))


def test_delimiter_and_multiline_quote_header_validation_matches_parser():
    with pytest.raises(CsvHeaderError):
        read_csv_checked(io.BytesIO(b'"quantity";quantity\n1;999\n'), sep=";")
    frame = read_csv_checked(io.BytesIO(b'"multi\nline",article\nvalue,002\n'))
    assert frame.columns.tolist() == ["multi\nline", "article"]
    assert frame.article.iloc[0] == "002"


def test_parsing_and_validation_run_in_worker_and_upload_is_not_read_whole(monkeypatch):
    monkeypatch.setattr(main, "predictor", StubPredictor())
    original = main._prepare_inputs_sync
    worker_ids = []

    def checked(*args):
        worker_ids.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(main, "_prepare_inputs_sync", checked)

    async def run():
        loop_thread = threading.get_ident()
        sales, stock = input_files()
        uploads = [UploadFile(io.BytesIO(data), filename=name) for data, name in [(sales, "s.csv"), (stock, "st.csv")]]
        # Parsing should operate on the spooled file, not await UploadFile.read().
        async def disallowed_read(*args, **kwargs):
            raise AssertionError("UploadFile.read would duplicate the complete file")
        for upload in uploads:
            upload.read = disallowed_read
        response = await main.predict_post(*uploads)
        assert response.status_code == 200
        assert worker_ids and all(thread_id != loop_thread for thread_id in worker_ids)

    asyncio.run(run())


def test_queued_post_does_not_parse_or_block_health(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []
    predictor = StubPredictor()
    monkeypatch.setattr(main, "predictor", predictor)

    def prepare(*args):
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(3), "Test did not release blocked CSV preparation"
        return pd.DataFrame({"warehouse_name": ["001"], "article": ["002"]}), pd.DataFrame(), None

    monkeypatch.setattr(main, "_prepare_inputs_sync", prepare)

    async def run():
        monkeypatch.setattr(main, "prediction_slots", asyncio.Semaphore(1))
        first = asyncio.create_task(main.predict_post(None, None))
        second = None
        try:
            assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), 2.5)
            second = asyncio.create_task(main.predict_post(None, None))
            await asyncio.sleep(0)
            assert len(calls) == 1  # queued request has not allocated/parsing its DataFrames
            assert (await asyncio.wait_for(main.health(), .25))["status"] == "ok"
        finally:
            release.set()
            responses = await asyncio.gather(first, *([second] if second else []))
        assert [r.status_code for r in responses] == [200, 200]
        assert len(calls) == 2

    asyncio.run(run())


@pytest.mark.integration
def test_bundled_weights_multipart_roundtrip(monkeypatch):
    monkeypatch.setattr(main, "predictor", Predictor(ROOT / "model").load())
    files = {
        "sales": ("sales.csv", (ROOT / "data/sample_sales_6mo.csv").read_bytes()),
        "stock": ("stock.csv", (ROOT / "data/sample_stock_6mo.csv").read_bytes()),
    }
    with TestClient(main.app) as client:
        assert client.get("/ready").status_code == 200
        first = client.post("/predict", files=files)
        repeat = client.post("/predict", files=files)
        assert first.status_code == repeat.status_code == 200
        assert first.content == repeat.content
        frame = pd.read_csv(io.BytesIO(first.content))
        assert len(frame) == 11 * 50 * 9
        assert not frame.duplicated(["week", "warehouse_name", "article"]).any()
        csv = client.get("/predict/all").content
        assert csv == first.content
        json_frame = pd.DataFrame(client.get("/predict/all?format=json").json())
        pd.testing.assert_frame_equal(frame, json_frame, check_dtype=False)
