#!/usr/bin/env bash
# Weekly API self-check. Run from forecast-container after starting the service.
# Usage: bash tests/test_api.sh [BASE_URL]
# sample_*_6mo.csv provide >=180 calendar days; the first-100-row samples do not.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SALES_FILE="${SALES_FILE:-data/sample_sales_6mo.csv}"
STOCK_FILE="${STOCK_FILE:-data/sample_stock_6mo.csv}"
TMP_DIR="${FORECAST_TEST_DIR:-.test-output}"
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)' "$TMP_DIR"
for file in "$SALES_FILE" "$STOCK_FILE"; do
  if [ ! -f "$file" ]; then
    echo "Missing $file; run scripts/make_samples.py first" >&2
    exit 1
  fi
done

echo "1. Health and weekly POST"
curl -sf "$BASE_URL/health" -o "$TMP_DIR/health.json"
"$PYTHON" - "$TMP_DIR/health.json" <<'PY'
import json,sys
health = json.load(open(sys.argv[1], encoding="utf-8"))
assert health["status"] == "ok" and health["model_loaded"], health
PY
curl -sf -X POST "$BASE_URL/predict" -F "sales=@$SALES_FILE" -F "stock=@$STOCK_FILE" -o "$TMP_DIR/forecast.csv"
"$PYTHON" - "$TMP_DIR/forecast.csv" "$SALES_FILE" <<'PY'
import csv,sys,math
from datetime import date,timedelta
with open(sys.argv[2], newline="", encoding="utf-8-sig") as stream:
    sales = list(csv.DictReader(stream))
pairs = {(r["warehouse_name"], r["article"]) for r in sales}
last = max(date.fromisoformat(r["date"]) for r in sales)
first = last - timedelta(days=last.weekday()) + timedelta(days=7)
weeks = {(first + timedelta(weeks=i)).isoformat() for i in range(9)}
with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    reader = csv.DictReader(stream)
    assert reader.fieldnames == ["week", "warehouse_name", "article", "quantity"], reader.fieldnames
    rows = list(reader)
actual = {(r["week"], r["warehouse_name"], r["article"]) for r in rows}
expected = {(week, warehouse, article) for week in weeks for warehouse, article in pairs}
assert actual == expected and len(actual) == len(rows), "Missing/extra/duplicate pair-week keys"
assert len(rows) == 9 * len(pairs)
for row in rows:
    quantity = float(row["quantity"])
    assert math.isfinite(quantity) and quantity >= 0, row
print(f"Validated {len(rows)} rows, {len(pairs)} input pairs and nine complete weeks from {first}")
PY

ARTICLE="$("$PYTHON" -c 'import csv,sys; print(next(csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8-sig")))["article"])' "$TMP_DIR/forecast.csv")"
WAREHOUSE="$("$PYTHON" -c 'import csv,sys; print(next(csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8-sig")))["warehouse_name"])' "$TMP_DIR/forecast.csv")"

echo "2. Filtered GET, complete CSV and JSON parity"
curl -sf -G "$BASE_URL/predict" --data-urlencode "article=$ARTICLE" -o "$TMP_DIR/get_article.csv"
curl -sf -G "$BASE_URL/predict" --data-urlencode "warehouse=$WAREHOUSE" -o "$TMP_DIR/get_warehouse.csv"
curl -sf -G "$BASE_URL/predict" --data-urlencode "article=$ARTICLE" --data-urlencode "warehouse=$WAREHOUSE" -o "$TMP_DIR/get_both.csv"
curl -sf "$BASE_URL/predict/all?format=csv" -o "$TMP_DIR/full_forecast.csv"
curl -sf "$BASE_URL/predict/all?format=json" -o "$TMP_DIR/full_forecast.json"
"$PYTHON" - "$TMP_DIR" "$ARTICLE" "$WAREHOUSE" <<'PY'
import csv,json,sys
from pathlib import Path
root,article,warehouse = Path(sys.argv[1]),sys.argv[2],sys.argv[3]
def read(name):
    with (root/name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))
full = read("forecast.csv")
assert read("full_forecast.csv") == full
assert read("get_article.csv") == [r for r in full if r["article"] == article]
assert read("get_warehouse.csv") == [r for r in full if r["warehouse_name"] == warehouse]
assert read("get_both.csv") == [r for r in full if r["article"] == article and r["warehouse_name"] == warehouse]
objects = json.loads((root/"full_forecast.json").read_text(encoding="utf-8"))
assert len(objects) == len(full)
for obj,row in zip(objects,full):
    assert set(obj) == {"week","warehouse_name","article","quantity"}
    assert all(obj[key] == row[key] for key in ("week","warehouse_name","article"))
    assert abs(float(obj["quantity"]) - float(row["quantity"])) <= 1e-8 * max(1,abs(float(row["quantity"])))
PY

echo "3. Validation errors and CORS"
code="$(curl -s -o "$TMP_DIR/error.json" -w '%{http_code}' "$BASE_URL/predict")"
test "$code" = "400"
"$PYTHON" - "$TMP_DIR/error.json" <<'PY'
import json,sys
error = json.load(open(sys.argv[1], encoding="utf-8"))
assert error["detail"] and error["code"] == "VALIDATION_ERROR",error
PY
code="$(curl -s -o "$TMP_DIR/error.json" -w '%{http_code}' -X POST "$BASE_URL/predict")"
test "$code" = "422"
code="$(curl -s -o "$TMP_DIR/error.json" -w '%{http_code}' "$BASE_URL/predict/all?format=invalid")"
test "$code" = "400"
curl -sf -D "$TMP_DIR/cors.headers" -o /dev/null -X OPTIONS "$BASE_URL/predict" \
  -H 'Origin: http://localhost:5173' -H 'Access-Control-Request-Method: POST'
grep -qi 'access-control-allow-origin' "$TMP_DIR/cors.headers"

echo "4. Determinism and signed stock accepted"
curl -sf -X POST "$BASE_URL/predict" -F "sales=@$SALES_FILE" -F "stock=@$STOCK_FILE" -o "$TMP_DIR/repeat.csv"
cmp "$TMP_DIR/forecast.csv" "$TMP_DIR/repeat.csv"
"$PYTHON" - "$STOCK_FILE" "$TMP_DIR/stock_negative.csv" <<'PY'
import csv,sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    reader = csv.DictReader(stream)
    fields,rows = reader.fieldnames,list(reader)
rows[0]["stock"] = "-2"
with open(sys.argv[2],"w",newline="",encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
curl -sf -X POST "$BASE_URL/predict" -F "sales=@$SALES_FILE" -F "stock=@$TMP_DIR/stock_negative.csv" -o "$TMP_DIR/negative_stock_forecast.csv"
echo "All weekly API checks passed"
