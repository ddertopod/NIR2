# НИР: прогноз решения по ипотечной заявке (HMDA 2024) + анализ справедливости и интерпретируемости

Проект оформлен в виде Jupyter/Colab-ноутбука **`NIR.ipynb`** и реализует полный пайплайн:

1. загрузка публичного датасета HMDA LAR (Loan Application Register) за 2024 год;
2. очистка и фильтрация записей;
3. генерация признаков (в т.ч. обработка пропусков и гео-фичей);
4. разбиение на train/val/test со стратификацией по пересечению **таргета** и **sensitive-групп**;
5. обучение табличной нейросети (**PyTorch + PyTorch Lightning**) на *несенситивных* признаках;
6. оценка качества и **fairness-метрик** по чувствительным атрибутам (включая пересечение групп);
7. интерпретация модели через **SHAP**.

---

## Данные

Источник данных скачивается автоматически из официального хранилища FFIEC/CFPB:

- ZIP: `https://files.ffiec.cfpb.gov/static-data/snapshot/2024/2024_public_lar_csv.zip`
- внутри ZIP извлекается CSV и сохраняется как: `data/2024_public_lar_csv.csv`

### Целевая переменная

В ноутбуке используется колонка:

- **`action_taken`** — приводится к бинарной разметке:
  - `1` → `1` (loan originated)
  - `3` → `0` (application denied)

### Подвыборка

Перед обучением данные фильтруются:

- `loan_purpose == 1` (дом/покупка жилья)
- `action_taken ∈ {1, 3}`
- обязательное наличие: `loan_amount`, `income`

### Чувствительные признаки

В проекте есть два набора:

- **`SENSITIVE_COLS`** — используются для построения *групп* при стратификации и для audit:
  - `derived_ethnicity`, `derived_race`, `derived_sex`

- **`SENSITIVE`** — более широкий список (включая прямые атрибуты и сильные прокси), который **удаляется из обучающего датасета** `df_model_*`:
  - derived_* (этничность/раса/пол)
  - applicant/co_applicant race/sex/ethnicity
  - возрастные признаки
  - `tract_minority_population_percent` (демографический прокси)
  - `derived_msa_md` (гео/рынок как прокси)

При этом **audit-таблицы** (`df_audit_*`) сохраняют чувствительные колонки для последующего fairness-анализа.

---

## Шаги работы

### 1) Загрузка датасета

- Скачивает ZIP (если нет локально).
- Извлекает CSV в папку `data/`.

### 2) Очистка (`TODROP`)

Ключевые действия:

- фильтрация строк по условиям выше;
- бинаризация `action_taken`;
- удаление потенциальных утечек/лишних колонок:
  - все `denial_reason_*`;
  - константы (`loan_purpose`, `activity_year`, `multifamily_affordable_units`);
  - все колонки с суффиксом `_2/_3/_4/_5`;
  - ряд дополнительных колонок (в т.ч. некоторые cost/rate поля и др.).
- фильтрация строк с «невалидными/неуказанными» значениями в sensitive-полях (например, коды `3/4/6/7`, возраста `8888` и т.д.).

### 3) Генерация признаков (`FEATENG`)

Добавляются/преобразуются признаки:

- **`has_coapplicant`** — флаг наличия созаёмщика (по нескольким полям co-applicant).
- **`special_loan_conditions`** — агрегированный признак по набору флагов:
  - `negative_amortization`, `interest_only_payment`, `balloon_payment`, `other_nonamortizing_features`,
    `reverse_mortgage`, `open_end_line_of_credit`.
- **Гео/цензусные фичи**:
  - `owner_occ_ratio`, `homes_per_capita`, `tract_median_income_est`, `rating_income_tract`
  - флаг «плохих/пропущенных гео-полей» агрегируется далее.
- Приведение строковых числовых полей к float + индикаторы пропусков:
  - `combined_loan_to_value_ratio`, `loan_term`, `property_value` (обрабатываются токены вроде `Exempt`).
- **`missing_info`** — компактный категориальный код паттерна пропусков (`g/c/l/p`), затем исходные флаги удаляются.
- **`ltv`** — `loan_amount / property_value`.
- удаляются некоторые «технические/гео идентификаторы» (`state_code`, `county_code`, `census_tract`) и потенциальная утечка `preapproval`.

Дополнительно в ноутбуке есть утилиты для анализа «пустых строк», специальных токенов (`__NA__`, `Exempt`) и доли нулей в разрезе классов.

### 4) Сохранение очищенного датасета

Очищенный CSV сохраняется по пути `CLEAN_CSV_PATH`.

> В исходном ноутбуке `CLEAN_CSV_PATH` настроен на Google Drive (`/content/drive/MyDrive/...`).
> Если вы запускаете **локально**, замените путь на обычный файл в проекте, например `data/2024_public_lar_csv_clean.csv`.

### 5) Train/Val/Test split

Разбиение выполняется **только один раз на полном df**, затем строятся представления:

- `df_audit_*` — полный набор колонок (включая sensitive) для анализа;
- `df_model_*` — данные для обучения модели (удаляются колонки из `SENSITIVE`).

Стратификация делается по **пересечению**:

- `y` (таргет `action_taken`) × `SENSITIVE_COLS` (derived_ethnicity/race/sex)

Редкие страты (меньше `RARE_MIN_COUNT`) сворачиваются в `__RARE__`, чтобы разбиение не «падало».

### 6) Подготовка данных для модели

- признаки делятся на **категориальные** и **числовые**:
  - категориальные — по dtype (`object/string`) + список принудительно-категориальных колонок (например `debt_to_income_ratio`, `conforming_loan_limit`, `missing_info` и др.).
- `CatEncoder` строит словари категорий на train, добавляя токен `__UNK__` для неизвестных.
- `NumScaler` стандартизирует числовые признаки (z-score), NaN/Inf → 0.
- группы для fairness/стратегий обучения кодируются по `SENSITIVE_COLS` (редкие группы → `__RARE__`).

### 7) Модель

**`TabularNet`**:

- embeddings для категориальных признаков;
- конкатенация `[embeddings || numeric]`;
- MLP (Linear → BatchNorm → Activation → Dropout);
- выход: логит (BCEWithLogits).

### 8) Обучение (PyTorch Lightning)

Lightning-модуль **`HMDALitFull`** умеет несколько стратегий:

- `fairness_strategy="none"` — обычное обучение;
- `fairness_strategy="reweight"` — перевзвешивание лосса по inverse-frequency групп;
- `fairness_strategy="group_dro"` — GroupDRO (робастность к «худшей» группе).

Валидация:

- считается `val_auc`;
- подбирается порог `thr`, который **максимизирует F1 для класса 0** (`val_f1_0_best`) на сетке порогов;
- сохраняется best-checkpoint по `val_f1_0_best` + EarlyStopping по тому же критерию.

Логи пишутся в `logs/` через `CSVLogger`.

### 9) Тестирование + fairness

На тесте выводятся:

- confusion matrix, ROC-AUC, PR-AUC, classification report;
- fairness-таблицы и разрывы по каждому признаку из `SENSITIVE`, который присутствует в audit:
  - **Demographic parity gap** (разница доли предсказанных позитивов между группами)
  - **TPR gap / FPR gap**
  - **Equalized odds gap** (max(TPR_gap, FPR_gap))
- аналогично — для **пересечения всех доступных sensitive-признаков**.

Для `tract_minority_population_percent` используется биннинг в 10 квантилей, чтобы избежать слишком большого числа групп.

### 10) SHAP интерпретация

Используется `shap.GradientExplainer`:

- считается SHAP на представлении `z = [emb(cat) || x_num]`;
- для категориальных признаков вклад агрегируется суммой по размерностям embedding;
- строится:
  - глобальная важность (mean |SHAP|);
  - summary plot для числовых фич;
  - (опционально) сравнение вкладов FP vs TN среди `y=0`.

---

## Быстрый старт

### Вариант A — Google Colab (рекомендуется)

1. Откройте `NIR.ipynb` в Colab.
2. Выполните ячейку установки зависимостей.
3. (Опционально) примонтируйте Google Drive (ячейка `drive.mount`).
4. Запускайте ячейки последовательно сверху вниз.

### Вариант B — локальный запуск

1. Создайте окружение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2. Поставьте зависимости (минимальный набор):

```bash
pip install pandas numpy scikit-learn matplotlib shap torch pytorch-lightning
```

3. В ноутбуке:

- закомментируйте/удалите `from google.colab import drive` и `drive.mount(...)`;
- задайте локальный путь для `CLEAN_CSV_PATH`, например:

```python
CLEAN_CSV_PATH = Path("data/2024_public_lar_csv_clean.csv")
```

4. Запустите ноутбук через Jupyter:

```bash
pip install notebook
jupyter notebook
```

---

## Настройка экспериментов

Полезные параметры в ноутбуке:

- `TEST_SIZE`, `VAL_SIZE`, `RANDOM_STATE`, `RARE_MIN_COUNT` — разбиение данных;
- `SENSITIVE_COLS` — какие sensitive-колонки используются для стратификации и групп;
- `SENSITIVE` — какие колонки вырезаются из `df_model_*` (чтобы модель не обучалась на sensitive/proxy);
- `BATCH_SIZE`, `NW_*` — производительность DataLoader;
- `HMDALitFull(... fairness_strategy=...)` — включение reweight / group_dro;
- `train_loader_full_oversampled` — опциональный oversampling (в ноутбуке создан отдельный loader).

---

## Выходные артефакты

В процессе работы ноутбук создаёт:

- `data/` — raw данные и (опционально) cleaned CSV;
- `logs/` — CSV-логи обучения;
- checkpoint лучшей модели (путь печатается после `trainer.fit(...)`).

---

## Ограничения и этика

- HMDA — публичные данные, но модель может улавливать социальные и географические прокси.
- Даже при удалении sensitive-признаков **полностью убрать дискриминационные эффекты сложно** из-за корреляций.
- Используйте результаты только для исследовательских задач, анализа качества/справедливости и интерпретируемости.


---

## Контакты / автор

Софронов М.Д. @ddertopod
