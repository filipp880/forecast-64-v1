"""Build and verify the image over loopback inside a network-none container."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd

CONTAINER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTAINER))
from app.features import truncate_history
from verify_runtime import multipart

CLIENT = r'''
import hashlib,io,json,time,urllib.request,urllib.parse,urllib.error
import pandas as pd
base='http://127.0.0.1:8000'
body=open('/tmp/request.body','rb').read()
def request(path,data=None):
 start=time.perf_counter()
 headers={'Content-Type':'multipart/form-data; boundary=forecast-weekly-fixed-boundary-20261027'} if data is not None else {}
 with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers),timeout=3600) as r: value=r.read()
 return value,time.perf_counter()-start
deadline=time.monotonic()+60
while True:
 try:
  if json.loads(request('/health')[0]).get('model_loaded'):break
 except OSError:pass
 if time.monotonic()>deadline:raise RuntimeError('Model not ready')
 time.sleep(.2)
one,t1=request('/predict',body)
two,t2=request('/predict',body)
assert one==two,'Nondeterministic POST'
csv,tc=request('/predict/all?format=csv')
assert csv==one,'GET/POST mismatch'
objects,tj=request('/predict/all?format=json')
f=pd.read_csv(io.BytesIO(one));j=pd.DataFrame(json.loads(objects))
pd.testing.assert_frame_equal(f,j,check_exact=False,rtol=0,atol=1e-8)
pair=f.iloc[0]
for q in ({'article':pair.article},{'warehouse':pair.warehouse_name},{'article':pair.article,'warehouse':pair.warehouse_name}):
 raw,_=request('/predict?'+urllib.parse.urlencode(q));wanted=f
 if 'article' in q:wanted=wanted[wanted.article==q['article']]
 if 'warehouse' in q:wanted=wanted[wanted.warehouse_name==q['warehouse']]
 pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(raw)),wanted.reset_index(drop=True))
open('/tmp/forecast.csv','wb').write(one)
print(json.dumps(dict(rows=len(f),post_seconds=[t1,t2],get_csv_seconds=tc,get_json_seconds=tj,deterministic_bytes=True,csv_json_filters_parity=True,forecast_sha256=hashlib.sha256(one).hexdigest())))
'''


def main(argv=None):
    root = CONTAINER.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--image", default="forecast-api-weekly")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--out", type=Path, default=CONTAINER / "logs/weekly_docker.json")
    args = ap.parse_args(argv)
    def docker(*parts, **kwargs):
        return subprocess.run(["docker", *map(str, parts)], check=True, capture_output=True, text=True, **kwargs).stdout.strip()
    docker("version", "--format", "{{.Server.Version}}")
    started = time.perf_counter()
    if not args.skip_build:
        subprocess.run(["docker", "build", "-t", args.image, str(CONTAINER)], check=True)
    build_seconds = time.perf_counter() - started
    image_bytes = int(docker("image", "inspect", args.image, "--format", "{{.Size}}"))
    if image_bytes > 30 * 1024**3 or build_seconds > 3600:
        raise AssertionError("Build time or image size limit exceeded")
    ident = docker("run", "-d", "--network", "none", "--memory", "7g", "--cpus", "8", args.image)
    try:
        inspection = json.loads(docker("inspect", ident))[0]
        assert inspection["HostConfig"]["NetworkMode"] == "none"
        # Explicitly copy uploaded input into /tmp; the image itself has no train CSV.
        with tempfile.TemporaryDirectory(prefix="weekly_docker_") as temp:
            temp = Path(temp).resolve()
            if temp.parent != Path(tempfile.gettempdir()).resolve() or not temp.name.startswith("weekly_docker_"):
                raise ValueError("Unexpected temporary workspace")
            sales, stock = truncate_history(pd.read_csv(root / "data/sales_dataset_hakaton_train.csv"),
                                            pd.read_csv(root / "data/stock_dataset_hakaton_train.csv"))
            body, _ = multipart(sales.to_csv(index=False).encode(), stock.to_csv(index=False).encode())
            payload = temp / "request.body"
            payload.write_bytes(body)
            docker("cp", payload, f"{ident}:/tmp/request.body")
            report = json.loads(docker("exec", "-i", ident, "python", "-", input=CLIENT))
            docker("cp", f"{ident}:/tmp/forecast.csv", temp / "forecast.csv")
            actual, expected = pd.read_csv(temp / "forecast.csv"), pd.read_csv(args.expected)
            pd.testing.assert_frame_equal(actual.iloc[:, :3], expected.iloc[:, :3])
            np.testing.assert_allclose(actual.quantity, expected.quantity, rtol=0, atol=1e-8)
            if max(report["post_seconds"]) > 3600 or max(report["get_csv_seconds"], report["get_json_seconds"]) > 600:
                raise AssertionError("HTTP runtime limits exceeded")
            report.update(runtime="docker_linux", network_isolation_verified=True,
                          network_mode="none", memory_limit_gib=7, cpus=8,
                          image_bytes=image_bytes, build_seconds=None if args.skip_build else build_seconds,
                          build_skipped=bool(args.skip_build), submission_parity=True,
                          peak_memory_note="Container completed under a hard 7 GiB limit; exact peak not sampled.")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
    finally:
        subprocess.run(["docker", "rm", "-f", ident], check=False, capture_output=True)


if __name__ == "__main__":
    main()
