"""Exercise the saved production model with Python networking disabled."""
from __future__ import annotations
import argparse
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd

CONTAINER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTAINER))
from app.predictor import Predictor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=CONTAINER / "model")
    ap.add_argument("--sales", type=Path, default=CONTAINER / "data/sample_sales_6mo.csv")
    ap.add_argument("--stock", type=Path, default=CONTAINER / "data/sample_stock_6mo.csv")
    ap.add_argument("--out", type=Path, default=CONTAINER.parent / "artifacts/weekly-verified-20261027/offline.json")
    args = ap.parse_args(argv)
    sales, stock = pd.read_csv(args.sales), pd.read_csv(args.stock)
    attempted = []
    def denied(*arguments, **kwargs):
        attempted.append("network call")
        raise AssertionError("Network call during model load/inference")
    with patch.object(socket, "create_connection", denied), patch.object(socket.socket, "connect", denied), \
         patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "getaddrinfo", denied):
        model = Predictor(args.model_dir).load()
        forecast = model.predict(sales, stock)
    report = dict(production_model_sha256=model.config["model_artifact_sha256"], rows=len(forecast),
                  python_network_calls=len(attempted), network_guard_passed=True,
                  scope="Python socket/DNS calls disabled during saved-model loading and sample inference; not an OS/Docker network-isolation proof")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
