"""Full native HTTP parity, timing and process-memory verification (offline model)."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

CONTAINER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTAINER))
from app.features import COMPETITION_CUTOFF, truncate_history
from app.submission import validate_forecast


def multipart(sales, stock):
    boundary = "forecast-weekly-fixed-boundary-20261027"
    blocks = []
    for name, payload in (("sales", sales), ("stock", stock)):
        blocks += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{name}.csv\"\r\nContent-Type: text/csv\r\n\r\n".encode(), payload, b"\r\n"]
    blocks.append(f"--{boundary}--\r\n".encode())
    return b"".join(blocks), f"multipart/form-data; boundary={boundary}"


def main(argv=None):
    root = CONTAINER.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sales", type=Path, default=root / "data/sales_dataset_hakaton_train.csv")
    ap.add_argument("--stock", type=Path, default=root / "data/stock_dataset_hakaton_train.csv")
    ap.add_argument("--model-dir", type=Path, default=CONTAINER / "model")
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=CONTAINER / "logs/weekly_runtime.json")
    ap.add_argument("--base-url", help="Use an existing service, e.g. Docker, instead of starting native uvicorn")
    ap.add_argument("--bash", type=Path, help="Optionally also execute tests/test_api.sh using this Bash executable")
    args = ap.parse_args(argv)
    sales, stock = truncate_history(pd.read_csv(args.sales), pd.read_csv(args.stock), COMPETITION_CUTOFF)
    body, content_type = multipart(sales.to_csv(index=False).encode(), stock.to_csv(index=False).encode())
    expected = pd.read_csv(args.expected)
    process, done, peak = None, threading.Event(), [0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.with_suffix(".server.log").open("w", encoding="utf-8")
    if args.base_url:
        base = args.base_url.rstrip("/")
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        env = {**os.environ, "MODEL_DIR": str(args.model_dir.resolve()), "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=CONTAINER, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        def monitor():
            while not done.wait(.1):
                try:
                    info = psutil.Process(process.pid).memory_info()
                    peak[0] = max(peak[0], getattr(info, "peak_wset", info.rss))
                except psutil.Error:
                    return
        threading.Thread(target=monitor, daemon=True).start()
    def request(path, data=None):
        begin = time.perf_counter()
        headers = {"Content-Type": content_type} if data is not None else {}
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=3600) as response:
            payload = response.read()
        return payload, time.perf_counter() - begin
    try:
        deadline = time.monotonic() + 45
        while True:
            try:
                health = json.loads(request("/health")[0])
                if health.get("model_loaded"):
                    break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Service failed to load the weekly model; inspect server log")
            time.sleep(.25)
        first, sec1 = request("/predict", body)
        second, sec2 = request("/predict", body)
        if first != second:
            raise AssertionError("Repeated POST CSV differs byte-for-byte")
        forecast = pd.read_csv(io.BytesIO(first))
        validate_forecast(forecast, sales, "2026-07-27")
        def parity(actual, wanted):
            a = actual.sort_values(["week", "warehouse_name", "article"]).reset_index(drop=True)
            b = wanted.sort_values(["week", "warehouse_name", "article"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(a.iloc[:, :3], b.iloc[:, :3])
            np.testing.assert_allclose(a.quantity, b.quantity, rtol=0, atol=1e-8)
        parity(forecast, expected)
        full, get_csv = request("/predict/all?format=csv")
        if full != first:
            raise AssertionError("GET CSV differs from POST")
        objects, get_json = request("/predict/all?format=json")
        parity(pd.DataFrame(json.loads(objects)), forecast)
        pair = forecast.iloc[0]
        for filters in ({"article": pair.article}, {"warehouse": pair.warehouse_name},
                        {"article": pair.article, "warehouse": pair.warehouse_name}):
            payload, _ = request("/predict?" + urllib.parse.urlencode(filters))
            wanted = forecast
            if "article" in filters:
                wanted = wanted[wanted.article == filters["article"]]
            if "warehouse" in filters:
                wanted = wanted[wanted.warehouse_name == filters["warehouse"]]
            parity(pd.read_csv(io.BytesIO(payload)), wanted)
        report = dict(runtime="existing_http_service" if args.base_url else "native_windows_http",
                      rows=len(forecast), weeks=9, post_seconds=[sec1, sec2], get_csv_seconds=get_csv,
                      get_json_seconds=get_json, peak_service_gib=peak[0]/2**30 if process else None,
                      deterministic_bytes=True, submission_parity=True, get_csv_json_filters_parity=True,
                      forecast_sha256=hashlib.sha256(first).hexdigest(), history_cutoff="2026-07-26",
                      network_isolation_verified=False,
                      network_note="Native HTTP verification; Docker --network none must be verified separately.")
        if args.bash:
            env = {**os.environ, "FORECAST_TEST_DIR": str((args.out.parent / "http-weekly-output").resolve()),
                   "PYTHON": Path(sys.executable).as_posix(),
                   "PATH": str(args.bash.parent) + os.pathsep + os.environ.get("PATH", "")}
            subprocess.run([str(args.bash), "tests/test_api.sh", base], cwd=CONTAINER, env=env,
                           check=True, stdout=log, stderr=log)
            report["shell_api_test_passed"] = True
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        done.set()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log.close()


if __name__ == "__main__":
    main()
