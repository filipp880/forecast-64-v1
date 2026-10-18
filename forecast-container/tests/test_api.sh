#!/usr/bin/env bash
# Самопроверка API (п. 10.10 ТЗ). Запускать после старта сервиса.
#
# Usage: bash tests/test_api.sh [BASE_URL]
#
# Тяжёлые сэмплы data/sample_*_6mo.csv (>=180 дней истории) готовятся скриптом:
#     python scripts/make_samples.py
# Лёгкие data/sample_sales.csv / sample_stock.csv — примеры формата (первые 100 строк),
# они НЕ подходят для POST из-за короткой истории.

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SALES_FILE="${SALES_FILE:-data/sample_sales_6mo.csv}"
STOCK_FILE="${STOCK_FILE:-data/sample_stock_6mo.csv}"
TMP_DIR="${TMPDIR:-/tmp}/forecast_api_test"
mkdir -p "$TMP_DIR"

for f in "$SALES_FILE" "$STOCK_FILE"; do
  if [ ! -f "$f" ]; then
    echo "Нет файла $f — сгенерируйте сэмплы: python scripts/make_samples.py"
    exit 1
  fi
done

echo "=== 1. Health check ==="
curl -sf "$BASE_URL/health"
echo

echo "=== 2. POST /predict ==="
curl -sf -X POST "$BASE_URL/predict" \
  -F "sales=@$SALES_FILE" \
  -F "stock=@$STOCK_FILE" \
  -o "$TMP_DIR/forecast.csv"
python - "$TMP_DIR/forecast.csv" "$SALES_FILE" <<'PY'
import csv
import sys
from datetime import date, timedelta

forecast_path, sales_path = sys.argv[1], sys.argv[2]

with open(sales_path, newline="", encoding="utf-8") as f:
    combos = {(r["warehouse_name"], r["article"]) for r in csv.DictReader(f)}

with open(forecast_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == ["date", "warehouse_name", "article", "quantity"], reader.fieldnames
    rows = list(reader)

dates = sorted({r["date"] for r in rows})
assert len(dates) == 64, f"дат {len(dates)}, ожидалось 64"
expected = date.fromisoformat(dates[0])
for d in dates:
    assert date.fromisoformat(d) == expected, f"даты не подряд: {d}"
    expected += timedelta(days=1)

assert len(rows) == len(combos) * 64, f"строк {len(rows)}, ожидалось {len(combos) * 64}"
for r in rows:
    q = float(r["quantity"])
    assert q == q and q >= 0, f"плохой quantity: {r}"
print(f"  формат OK: {len(rows):,} строк, 64 даты, {len(combos):,} пар склад×товар")
PY

ARTICLE="$(python -c "import csv,sys; print(next(csv.DictReader(open('$TMP_DIR/forecast.csv', encoding='utf-8')))['article'])")"
WAREHOUSE="$(python -c "import csv,sys; print(next(csv.DictReader(open('$TMP_DIR/forecast.csv', encoding='utf-8')))['warehouse_name'])")"

echo "=== 3. GET /predict?article=$ARTICLE ==="
curl -sf "$BASE_URL/predict?article=$ARTICLE" | head -3
echo "OK"

echo "=== 4. GET /predict?warehouse=$WAREHOUSE ==="
curl -sf "$BASE_URL/predict?warehouse=$WAREHOUSE" | head -3
echo "OK"

echo "=== 5. GET /predict?article=&warehouse= ==="
curl -sf "$BASE_URL/predict?article=$ARTICLE&warehouse=$WAREHOUSE" | head -3
echo "OK"

echo "=== 6. GET /predict/all?format=csv ==="
curl -sf "$BASE_URL/predict/all?format=csv" -o "$TMP_DIR/full_forecast.csv"
lines="$(wc -l < "$TMP_DIR/full_forecast.csv")"
echo "Получен файл: $lines строк"
echo "OK"

echo "=== 7. GET /predict/all?format=json ==="
curl -sf "$BASE_URL/predict/all?format=json" -o "$TMP_DIR/full_forecast.json"
python -m json.tool "$TMP_DIR/full_forecast.json" > /dev/null
echo "OK"

echo "=== 8. Error: GET /predict без параметров (ожидается 400) ==="
code="$(curl -s -o "$TMP_DIR/error.json" -w "%{http_code}" "$BASE_URL/predict")"
if [ "$code" != "400" ]; then
  echo "Провал: ожидался 400, получен $code"
  cat "$TMP_DIR/error.json"
  exit 1
fi
cat "$TMP_DIR/error.json"
echo
echo "OK"

echo "=== 9. CORS preflight (заголовки для браузера) ==="
headers="$(curl -s -i -X OPTIONS "$BASE_URL/predict" \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: POST")"
if ! grep -qi "access-control-allow-origin" <<< "$headers"; then
  echo "Провал: нет заголовка Access-Control-Allow-Origin"
  echo "$headers"
  exit 1
fi
grep -i "access-control-allow-origin" <<< "$headers"
echo "OK"

echo "=== Все тесты пройдены! ==="
