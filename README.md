# Прогноз недельных продаж на 9 недель

## 1. Описание решения

Глобальная модель прогнозирует сумму `quantity` за неделю для пары `warehouse_name × article`. Обученные веса находятся в `model/`; переобучение для запуска API не требуется.

Итоговое решение — ансамбль LightGBM: 0,3 × hurdle (вероятность положительной недели и Poisson-величина) + 0,7 × регрессия `log1p(quantity)`. Используются 49 признаков: история продаж, доли нулей, остатки, недельные EWMA, вариативность и агрегаты по артикулам/складам/категориям. Число деревьев — 120/188/88. Собственное log1p таргета обращается через expm1; затем применяется выбранная калибровка и округление недельного прогноза.

Продажи суммируются через города и департаменты. Дубли остатков внутри дня суммируются, отрицательные остатки сохраняются. Будущие продажи и остатки не используются. Денежные поля не входят в выбранные 49 признаков.

Историческая оценка на периоде 25.05–26.07.2026:

| SMAPE | WAPE | RMSE | Средний абсолютный Bias по сериям | RMSLE |
|---:|---:|---:|---:|---:|
| 59,3214% | 46,3284% | 3,2149 | 72,6117% | 0,5030 |

Период ранее просматривался; это историческая оценка, а не новый независимый holdout. Готовый прогноз — **[forecast.csv](forecast.csv)**. Параметры и SHA-256 — [forecast.manifest.json](forecast.manifest.json), полный состав пар — [required_pairs.csv](required_pairs.csv). Исторически наблюдавшиеся пары перечислены отдельно в `observed_pairs.csv`.

Готовый CSV содержит **79 101 строку: 11 складов × 799 артикулов × 9 недель**. Для 12 пар без истории используется предусмотренный fallback по среднему спросу артикула за последние 28 дней. Прогнозы по 8 777 наблюдавшимся парам сохранены без изменений. Для повторной генерации полного набора: `python generate_submission.py --all-combinations --out forecast_new.csv`.

Обучение реализовано в `train.py`; воспроизводимый исследовательский цикл — `scripts/research_weekly.py`. Полные исходные CSV размещаются локально в `data/` и не входят в Git/образ. Для нового исследовательского запуска:

```bash
python scripts/research_weekly.py --out artifacts/research-new --stage development
python scripts/research_weekly.py --out artifacts/research-new --stage confirm
python scripts/research_weekly.py --out artifacts/research-new --stage finalize
```

## 2. Инструкция по сборке

Требуется Docker Engine с Linux-контейнерами. Используется Python 3.11, зависимости закреплены в `requirements.lock.txt`. GPU не нужен. Интернет требуется при первой сборке. Все команды выполняются из корня репозитория:

```bash
docker build -t forecast-api .
```

`.dockerignore` исключает полные данные, локальные артефакты, кэши и секреты. В образ включены код приложения и готовая модель.

## 3. Инструкция по запуску

```bash
docker run --rm --name forecast-api -p 127.0.0.1:8000:8000 forecast-api
```

Альтернатива: `docker compose up --build`. Swagger: http://localhost:8000/docs. Готовность: http://localhost:8000/health, поле `model_loaded` должно быть `true`.

Модель работает автономно. Для запуска с отключённой сетью доступен `docker run --rm --network none forecast-api`; публикация порта хоста в таком режиме недоступна. CUDA-флаги не требуются.

## 4. Описание API

| Метод | Путь | Результат |
|---|---|---|
| GET | `/health` | состояние загрузки модели |
| POST | `/predict` | полный прогноз по двум multipart-файлам |
| GET | `/predict?article=…` | CSV по артикулу |
| GET | `/predict?warehouse=…` | CSV по складу |
| GET | `/predict?article=…&warehouse=…` | CSV пары |
| GET | `/predict/all?format=csv` | весь последний прогноз |
| GET | `/predict/all?format=json` | тот же прогноз в JSON |

```bash
curl -f http://localhost:8000/health
curl -f -X POST http://localhost:8000/predict \
  -F "sales=@data/sample_sales_6mo.csv" \
  -F "stock=@data/sample_stock_6mo.csv" -o forecast_result.csv
curl -f "http://localhost:8000/predict?article=char_article_0" -o article_result.csv
curl -f "http://localhost:8000/predict?warehouse=warehouse_name_0" -o warehouse_result.csv
curl -f "http://localhost:8000/predict?article=char_article_0&warehouse=warehouse_name_0" -o pair_result.csv
curl -f "http://localhost:8000/predict/all?format=csv" -o full_result.csv
curl -f "http://localhost:8000/predict/all?format=json" -o forecast_result.json
```

sales: `date,composite_division,cities_name,warehouse_name,category_lvl_1,category_lvl_2,category_lvl_3,article,characteristic,quantity,sales_sum,discount_sum`.

stock: `date,warehouse_name,char_article,stock`. Пустая скидка допустима. В обоих файлах требуется минимум 180 календарных дней; более длинная история учитывается целиком.

Выход: `week,warehouse_name,article,quantity`. API формирует полное произведение складов и артикулов входных sales, включая пары без совместной истории с fallback. Прогноз начинается с понедельника следующей календарной недели после максимальной даты sales и содержит девять полных недель. Например, после 31.07.2026 первая неделя — 03.08.2026. Quantity — неотрицательное конечное число, сумма за полную неделю. GET читает кэш последнего успешного POST; до POST кэш отсутствует. Неверная схема/глубина истории — 422, отсутствие GET-фильтра или ещё не рассчитанный прогноз — 400.

`sample_sales.csv` и `sample_stock.csv` содержат первые 100 строк для ознакомления со схемой. Для POST предназначены `sample_*_6mo.csv`.

## 5. Структура репозитория

```text
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.lock.txt
├── README.md
├── app/
│   ├── main.py                 # FastAPI
│   ├── predictor.py            # загрузка модели и прогноз
│   ├── validators.py           # валидация CSV
│   └── …                      # признаки, модели, метрики
├── model/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── config.json
├── data/
│   ├── sample_sales.csv
│   ├── sample_stock.csv
│   ├── sample_sales_6mo.csv
│   └── sample_stock_6mo.csv
├── tests/
│   ├── test_api.sh
│   └── test_*.py
├── scripts/                   # обучение и подготовка примеров
├── train.py
├── evaluate.py
├── baseline.py
├── score.py
├── generate_submission.py
├── forecast.csv               # готовый прогноз
├── observed_pairs.csv
├── required_pairs.csv
└── forecast.manifest.json
```

## 6. Известные ограничения

- Состав 79 101 ключей восстановлен полным произведением 11 × 799 × 9 согласно ожидаемой размерности внешней проверки. Машиночитаемый эталон не предоставлен; точное совпадение ключей должен подтвердить повторный внешний прогон.
- Готовый прогноз использует историю только по 26.07.2026 и охватывает 27.07–27.09.2026; последняя метка week — 21.09.2026. API использует даты загруженных данных.
- Именованные праздничные признаки не входят в выбранные 49 признаков; календарная сезонность представлена общими признаками месяца/недели/года. Региональная привязка анонимных складов неизвестна.
- Кэш находится в памяти одного процесса и очищается при перезапуске. Отдельный пользовательский веб-интерфейс не предусмотрен.
