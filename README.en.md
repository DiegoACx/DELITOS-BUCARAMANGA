<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<h1 align="center">Bucaramanga Crime Report Classifier</h1>

<p align="center">A temporally validated classifier of crime report types, built on open data from datos.gov.co.</p>

<p align="center">🇪🇸 <a href="README.md">Leer en español</a></p>

## About

This project classifies the **type of a crime report** in Bucaramanga (5 classes of the `tipologia` field) from its context: district, month, weekday, time slot, place, and the victim's sex, age and mobility. It uses a reproducible scikit-learn pipeline over the open dataset `x46e-abhz` from [datos.gov.co](https://www.datos.gov.co).

It downloads and cleans the data, trains on 2016-2021, tests on 2022, and saves a model and a metrics report. It also includes a descriptive notebook and a Streamlit app that shows the probability of each class.

### What it does NOT do

- It does **not measure risk** and does not predict future crimes, or when or where they will happen.
- It does not assess a person or an area. Every row is an already reported crime: there are no "no crime" cases, so risk cannot be estimated.

The app states it: *"Clasifica el tipo de un reporte según su contexto; no mide riesgo"* (it classifies the type of a report from its context; it does not measure risk).

## Data

- **Source:** resource `x46e-abhz` on datos.gov.co, queried through its SODA API.
- **Download:** `python src/download_data.py` fetches records with `fecha_hecho < '2023-01-01'`, ordered by `:id`, in pages of 20,000 rows, and checks the row count against the server. It writes `data/delitos_bucaramanga.csv` (ignored by git).
- **Why 2023 is excluded:** in the current version of the dataset, 93.4 % of `movil_victima` and 83.4 % of `edad` are "NO DISPONIBLE" in 2023 (aggregate API query, see the notebook). The cleaning step removes those rows, so 2023 nearly disappears.
- **Size:** 88,627 rows and 26 columns downloaded, 76,385 after cleaning (86.2 %).
- **SHA-256** of the downloaded file: `74aacd380a3c66ddd02d9df7e3754dca772f43496265ba456f1440d588e2d424`. It is valid for the resource as it was on 2026-09-19; the portal may change it.

## Methodology

- **Cleaning** (`src/prep.py`): the same filters, in the same order, as the original notebook (duplicates, rare weapons, districts outside Bucaramanga, "NO DISPONIBLE" values, invalid ages). The largest drops are unavailable sex (8,350 rows) and neighborhoods outside Bucaramanga (2,550 rows).
- **Inputs (9):** `num_comuna` (as a category), `mes_num`, `sexo`, `movil_victima`, `dia_nombre`, `rango_edad`, `rango_horario`, `curso_vida` and `clase_sitio`. The 224 values of `clase_sitio` are grouped into the 19 most frequent plus an "infrequent" bucket, learned on the training set only. `arma_empleada` and `movil_agresor` are not used.
- **Pipeline:** `ColumnTransformer` with `OneHotEncoder` inside a scikit-learn `Pipeline`, fitted on the training set only. Everything is categorical, so there is no scaling.
- **Temporal split:** train 2016-2021 (62,483 rows), test 2022 (13,902 rows).
- **Models:** `Dummy(prior)`, `LogisticRegression(max_iter=1000)` and `HistGradientBoostingClassifier`. Seed `random_state=0`, default hyperparameters, no tuning. The saved model is the HGB trained on 2016-2021 only.
- **Versions:** Python 3.13.3, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0.

## Results (test set: 2022)

| Model | Accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|
| Dummy (prior) | 0.56 | 0.14 | 1.35 |
| Logistic Regression | 0.61 | 0.41 | 1.07 |
| HistGradientBoosting | 0.62 | 0.42 | 1.07 |

Compared with the baseline, HGB gains about 5.5 accuracy points and lowers log loss by 21 % (1.3525 → 1.0705, computed from `reports/metrics.json`). The improvement is real but modest.

HistGradientBoosting by class:

| Class | Precision | Recall | Test rows |
|---|---:|---:|---:|
| Patrimonio económico | 0.67 | 0.90 | 7,826 |
| Vida e integridad personal | 0.39 | 0.25 | 2,501 |
| Familia | 0.51 | 0.45 | 1,498 |
| Seguridad pública | 0.14 | 0.00 (0.0006) | 1,694 |
| Libertad e integridad sexual | 0.52 | 0.56 | 383 |

Class distribution, train → test: Patrimonio 56.45 → 56.29 %, Vida 20.45 → 17.99 %, Familia 16.70 → 10.78 %, Sexuales 4.20 → 2.75 %, Seguridad pública 2.20 → 12.19 %.

## Limitations

- **A single temporal split.** No cross-validation and no confidence intervals in this repository.
- **The model was chosen by looking at the 2022 test set**, so that test is not independent. `SELECTED_MODEL` is fixed in `src/train.py`.
- **`Seguridad pública` is almost never predicted** (recall 0.0006 with HGB, 0.00 with LR), even though it is 12.19 % of the test set.
- **Label drift across years.** `Seguridad pública` does not exist before 2021 and is almost only threats (3,365 of 3,369 reports). In 2021, culpable injuries and culpable homicide in traffic accidents also appear. The weight of `Patrimonio` moves between 50.3 % and 65.8 % depending on the year. A model trained today cannot be trusted going forward.
- **The most influential variables are attributes of the report itself.** In a preliminary evaluation (not versioned here), `clase_sitio` and age carried the most weight. They are not known before the event, so this is classifying a report, not predicting a crime.
- **Selection bias and under-reporting.** Rows with unavailable sex or age are dropped (about 7-13 % per year), and only reported crimes exist in the data.
- **Demographic attributes of victims** (sex, age) are model inputs. Use it as an experiment, not for decisions about people.
- **The dataset changed.** The original 2024 notebook used 100,993 rows; the resource now has 130,202 and 2023 is encoded differently, so those figures are not comparable.
- No `LICENSE`, no CI, and the dev container has not been built or run (see below).

## Project structure

```
.
├─ app/streamlit_app.py
├─ notebooks/01_exploracion.ipynb
├─ models/pipeline.joblib
├─ reports/metrics.json
├─ src/{download_data,prep,train}.py
├─ tests/test_pipeline.py
├─ .devcontainer/devcontainer.json
├─ requirements.txt · requirements-dev.txt · .gitignore
└─ data/                      (generated, ignored by git)
```

## Quick start

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python src/download_data.py        # creates data/delitos_bucaramanga.csv
python src/train.py                # overwrites models/pipeline.joblib and reports/metrics.json
pytest
streamlit run app/streamlit_app.py
```

The tests only need `models/pipeline.joblib`, which is already committed; they do not need `data/`.

> **Security:** `models/pipeline.joblib` is a joblib pickle. Loading a pickle from an unknown source can execute arbitrary code. Only load models you trust, with the versions pinned in `requirements.txt`.

## Exploration notebook

`notebooks/01_exploracion.ipynb` is descriptive and shows only aggregates (no victim rows):

- Size and cleaning funnel (88,627 → 76,385).
- Report types by year: `Patrimonio` is the majority every year; `Familia` falls from 20.4 % (2016) to 10.8 % (2022).
- Districts: five (Oriental, San Francisco, Centro, Cabecera del Llano and Norte) account for 53.5 % of reports.
- Weekday and hour: Sundays and nights shift the mix toward `Vida e integridad personal` (31.5 % and 27.2 %).
- Missing data per year, including the 2023 problem (needs internet for the aggregate API query).
- Label drift between train and test.

## Dev Container

The image is `mcr.microsoft.com/devcontainers/python:1-3.13-bookworm`. It installs `requirements-dev.txt` on creation and launches the app on port 8501. CORS and XSRF protections are left at their defaults.

**Not tested:** the container was never built or run. What was verified: the image tag exists, all 53 packages resolve to `cp313` wheels for Linux (x86_64 and aarch64) with the pinned versions, and a clean virtual environment passes the tests. A possible issue is `streamlit`, installed with `--user`, not being on the container's `PATH`.

## Project history

The first version (September 2024) was an academic project. During a September 2026 review, the original notebook was found to have data leakage in its RandomForest evaluation, and the saved model (a `LinearRegression`) did not match the app (which called `predict_proba`). The repository was rebuilt with a temporal split, honest baselines and a reproducible pipeline. The old files remain in the git history.

## Author

- Diego Castro — [@DiegoACx](https://github.com/DiegoACx)

The 2026 refactor was developed with assistance from Claude (Anthropic).
