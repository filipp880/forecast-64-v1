# Forecast Container — 9 недель

## Описание решения

Глобальный direct weekly ансамбль: hurdle на LightGBM (вероятность ненулевых недельных продаж и Poisson-величина) и LightGBM на log1p недельного quantity. Вес, gate и округление выбираются на заранее фиксированной сетке первого временного окна и проверяются на втором. Окончательные параметры и метрики находятся в `model/config.json` и `logs/weekly_summary.json`.

Одна строка признаков — пара склада и товара × номер будущей недели. Target — сумма семи дней понедельник–воскресенье. Используются категории, дневные средние 7/14/28/56/90, частота и давность спроса, полные прошлые недели, календарь целевой недели и исторические остатки. mean_all использует всю переданную историю, включая данные старше 180 дней. Будущие факты не подставляются.

Stock — signed остаток на начало дня: дубли одного дневного ключа суммируются, отрицательные значения сохраняются. Исторический stock переносится только вперёд; наблюдения вне окна не превращаются в текущий запас. Неизвестные остатки сохраняют NaN в признаках. Продажи при stock≤0 не удаляются.

Финансовые поля не входят в боевую модель. Вспомогательная monetary_weekly_features обращает np.log через exp до суммы, с нулевым вкладом исходной null скидки. Собственное log1p таргета обращается через expm1.

## Сборка

Требуются Docker Engine с Linux-контейнерами; проверенная версия клиента и движка — 29.7.2, Compose — 5.5.0. В образе используется Python 3.11; GPU не нужен. Зависимости закреплены в requirements.lock.txt. Сеть нужна для первой сборки; при инференсе модель автономна. Команды этой инструкции выполняются из каталога forecast-container/.

```bash
docker build -t forecast-api-weekly .
```

## Запуск

```bash
docker run --rm -p 127.0.0.1:8000:8000 forecast-api-weekly
# либо Docker Compose v2:
docker compose up --build
```

Для локального запуска с установленными зависимостями:

```bash
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

MODEL_DIR задаёт каталог артефакта (по умолчанию model), CORS_ORIGINS — разрешённые origin через запятую. Swagger: http://localhost:8000/docs. Health: http://localhost:8000/health.

Лимиты задания: сборка ≤60 минут, образ ≤30 GB, RAM ≤64 GB, POST ≤60 минут, GET ≤10 минут. Измерения от 27.10.2026: Docker/Linux проверен без сети под лимитом 7 GiB и 8 CPU, POST 36,26/34,98 с, GET CSV 0,130 с, JSON 0,077 с; образ 305,5 MB. Точный пик контейнера не измерялся. [Подробный отчёт](../docs/reports/IMPLEMENTATION_WEEKLY_RU.md).

## API

| Маршрут | Вход | Ответ |
|---|---|---|
| GET /health | — | status, model_loaded |
| POST /predict | multipart sales, stock | Полный CSV на 9 недель |
| GET /predict | article и/или warehouse | Срез последнего успешного прогноза |
| GET /predict/all | format=csv или json | Последний полный прогноз |

```bash
curl -f -X POST http://localhost:8000/predict -F "sales=@data/sample_sales_6mo.csv" -F "stock=@data/sample_stock_6mo.csv" -o forecast.csv
curl -f "http://localhost:8000/predict?article=char_article_0" -o article.csv
curl -f "http://localhost:8000/predict?warehouse=warehouse_name_0" -o warehouse.csv
curl -f "http://localhost:8000/predict?article=char_article_0&warehouse=warehouse_name_0" -o pair.csv
curl -f "http://localhost:8000/predict/all?format=json" -o forecast.json
```

Выходные поля строго `week,warehouse_name,article,quantity`. Первая неделя API — понедельник следующей календарной недели после max(sales.date), даже если последняя дата сама является понедельником. Вход до 31.07.2026 даёт начало 03.08.2026. Результат покрывает наблюдаемые входные sales-пары, без скрытого расширения ассортимента.

Sales-схема соответствует разделу 2 ТЗ; cost_price/volume_weight не обязательны. Stock принимает article или char_article, конфликт отклоняется. Требуются даты YYYY-MM-DD, ≥180 календарных дней каждого файла, конечные quantity/stock, непустые ключи. Отрицательный quantity запрещён, отрицательный stock допустим. Null discount_sum не означает отсутствие продаж. Дубли полного исходного ключа sales отклоняются; агрегация разных департаментов/городов выполняется отдельно.

Пустой или нечитаемый файл →400, нарушение схемы/истории →422, незагруженная модель →503, GET без фильтра/кэша →400. Ошибка содержит detail/code. Неизвестный идентификатор GET даёт CSV с заголовком. Неудачная валидация POST не заменяет предыдущий успешный кэш.

## Обучение и проверка качества

Полные CSV отдельно разместить в ../data/. Конкурсная отсечка применяется к обоим источникам до панели и обучения.

```bash
python scripts/verified_backtest.py --out ../artifacts/weekly-verified-20261027 --max-cutoffs 16 --estimators 600 --threads 8
python scripts/select_candidate.py --out ../artifacts/weekly-verified-20261027
python scripts/select_candidate.py --out ../artifacts/weekly-verified-20261027 --confirm
python train.py --out-dir ../artifacts/weekly-final-20261027 --max-cutoffs 16 --estimators 600 --threads 8 --calibration-file ../artifacts/weekly-verified-20261027/decision.json
```

Новый протокол использует origins 15.02 и 24.05.2026. Все оценочные веса обучены только на прошлом своего origin. Первый прогнозный период — 16.02–19.04, второй — 25.05–26.07. Кандидат выбирается только на первом окне; на втором проверяется один выбранный вариант относительно исходного, без повторного выбора. Считаются все пять недельных метрик и baseline zero/last_week/mean4/mean8/mean13.

Локальная min-max нормализация относительно last_week не является официальным leaderboard. score.py дополнительно реализует ранговую нормализацию; равные значения получают средний ранг. Недостающие, лишние, отрицательные и нечисловые submitted значения не исправляются молча.

Независимое недельное подтверждение: SMAPE 59,8408%, WAPE 46,8954%, RMSE 3,3251, |Bias| по сериям 72,8448%, RMSLE 0,5066. Суммарный volume_bias −18,8262% — другая величина. Дневные 28–30% SMAPE и недельные 59,84% относятся к разным точкам оценки и временным окнам; [объяснение сравнения](../docs/reports/METRICS_COMPARISON_RU.md). Числа характеризуют историческое окно, а не неизвестный факт конечного CSV.

Протокол фиксирует хеши CSV, ML-кода и сетки до расчёта. Изменение источников/OOF/весов запрещает незаметное возобновление. Для другого протокола нужен новый каталог. Сохранённые даты, версия схемы, семантика stock, признаки и SHA проверяются при загрузке. Старые дневные веса не принимаются.

```bash
python baseline.py --out ../artifacts/weekly-baselines.json
python evaluate.py --model-dir ../artifacts/weekly-verified-20261027/2026-05-24 --origins 2026-05-24 --out ../artifacts/weekly-evaluation.json
python score.py --truth truth.csv --submissions a.csv b.csv --method rank
```

Финальные refit-веса нельзя честно оценивать на уже виденных ими датах; evaluate.py это запрещает без --allow-in-sample. Такая разрешённая диагностика явно маркируется.

## Генерация и упаковка

```bash
# Пока официальный каталог отсутствует — только явно помеченный provisional:
python generate_submission.py --allow-observed-pairs
# С официальным каталогом или шаблоном:
python generate_submission.py --required-pairs ../data/official_pairs.csv
python scripts/verify_runtime.py --expected ../submission/weekly_observed/forecast.csv
python scripts/verify_docker.py --expected ../submission/weekly_observed/forecast.csv
python scripts/package_weekly.py --forecast ../submission/weekly_observed/forecast.csv --out ../submission/weekly_observed/forecast_weekly_submission-20261028.zip
```

Основной прогноз всегда начинается 27.07.2026 и использует историю до 26.07. Артефакт, обученный или отобранный на более поздних фактах, отклоняется. Проверяются ровно девять недель, точное покрытие указанного каталога, отсутствие дублей/NaN/inf/отрицательных значений и повторяемость. Manifest содержит хеши и явный статус покрытия. Существующие файлы не перезаписываются без --overwrite.

Для 8 777 наблюдаемых пар получится 78 993 строки. Условные 79 101 строки относятся к полному произведению 11 × 799; официальный состав пар по имеющимся файлам подтвердить нельзя. Для заданных каталогом пар без продаж используется общий для API/оценки fallback: средний недельный объём артикула по складам, затем склада, затем общий средний; округление применяется согласованно.

## Тесты и структура

```bash
python -m pytest tests -q
bash tests/test_api.sh http://localhost:8000
```

Первые 100 строк sample_*.csv — примеры формата, короткой истории для POST недостаточно. sample_*_6mo.csv предназначены для HTTP-проверок. make_samples.py принимает оба названия артикула в stock и сохраняет резервы.

```text
forecast-container/
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── requirements.lock.txt
├── requirements-dev.txt
├── README.md
├── app/
│   ├── main.py                # FastAPI и HTTP-контракт
│   ├── predictor.py           # загрузка артефакта, weekly-инференс, кэш
│   ├── validators.py          # входные CSV
│   ├── features.py            # общие признаки и недельный target
│   ├── metrics.py             # пять метрик
│   ├── submission.py          # строгий контракт CSV
│   ├── temporal.py            # временные границы
│   ├── calibration.py         # общая постобработка
│   └── models/                # реестр и реализации моделей
├── model/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── config.json
├── data/
│   ├── sample_sales.csv       # первые 100 строк
│   ├── sample_stock.csv       # первые 100 строк
│   ├── sample_sales_6mo.csv   # вход для успешной HTTP-проверки
│   └── sample_stock_6mo.csv
├── tests/
│   ├── test_api.sh
│   ├── test_regressions.py
│   ├── test_weekly_contract.py
│   ├── test_provenance.py
│   └── test_weekdays_backtest.py
├── scripts/                   # воспроизведение, проверки и упаковка
├── logs/                      # компактные отчёты; старый verified_summary исторический
├── train.py
├── evaluate.py
├── baseline.py
├── score.py
└── generate_submission.py
```

Обязательные файлы п. 10.5 ТЗ сохранены на ожидаемых местах. Дополнительные модули нужны для общего train/serve-кода, строгого скоринга и воспроизводимости. Полные CSV, OOF, промежуточные модели и старые лаборатории остаются локально вне Git. [ТЗ и отчёты](../docs/README.md).

## Известные ограничения

- Официальный каталог пар не предоставлен; локальный observed submission предварительный по покрытию.
- Российский календарь праздников и переносов пока не реализован; присутствуют общие month/week_of_year/year.
- Кэш один на процесс, очищается перезапуском; очереди и пользовательской истории нет.
- Отдельного веб-интерфейса в этом репозитории нет.
- Два временных окна не доказывают устойчивость по всем сезонам. Проверка семи weekday отдельно отмечает перекрытие окон факта и не используется для выбора параметров.
- Исторические demo и *_lab.py не являются новым weekly pipeline. Прежние дневные отчёты сохранены как история.
