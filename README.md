<div align="center">

# Titanic Survival Prediction

**A production-grade machine learning system — not a Kaggle notebook.**

Leakage-safe preprocessing, a statistically-validated feature schema, a controlled four-model comparison, an explainable production model, a versioned deployment pipeline, and a tested, CI-gated codebase.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-76%20passing-2E7D32)
![Coverage](https://img.shields.io/badge/Core%20Module%20Coverage-100%25-2E7D32)
![License](https://img.shields.io/badge/License-MIT-1F4E79)

</div>

---

## Project Overview

This project predicts Titanic passenger survival — but the dataset is the
vehicle, not the point. It exists to demonstrate a full ML engineering
lifecycle: a leakage-safe preprocessing pipeline, a production feature
schema chosen by VIF and mutual-information evidence rather than
intuition, a controlled comparison across four models, and a deployment
plan that upgrades the model without breaking the live API. The system
ships as a versioned FastAPI service, a 6-page Streamlit dashboard, and
a Docker/EC2 deployment path, backed by 76 automated tests and a CI
pipeline that rebuilds and validates every artifact on every push.

## Key Highlights

- [x] End-to-end ML pipeline — raw data to deployed API
- [x] Production feature engineering — leakage-safe, fitted once, versioned
- [x] Statistical feature validation — VIF, mutual information, chi-square
- [x] Controlled model comparison — 4 models, one identical protocol
- [x] Explainable predictions — coefficient decomposition, no black box
- [x] FastAPI service — validated request/response contracts
- [x] Dockerized deployment — versioned images, no assumed downtime
- [x] AWS EC2 hosting — documented rollback runbook
- [x] 6-page Streamlit dashboard — architecture, live prediction, decision support
- [x] Versioned artifacts — model, preprocessing, and schema paired and checked
- [x] CI/CD — lint, rebuild artifacts, verify versions, test, on every push

## System Architecture

<div align="center">
<img src="docs/architecture-diagram.png" width="640" alt="System architecture — prediction request flow">
</div>

Both the preprocessing artifact and the model load once at process
startup — never refit, never reloaded per request. Full request-to-response
trace: [`reports/stage6_deployment_plan.md`](reports/stage6_deployment_plan.md).

## Dashboard Preview

<table>
<tr>
<td width="50%"><img src="docs/screenshots/home.png" alt="Home"><br><sub align="center">Home</sub></td>
<td width="50%"><img src="docs/screenshots/live-prediction.png" alt="Live Prediction"><br><sub>Live Prediction</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/model-performance.png" alt="Model Performance"><br><sub>Model Performance</sub></td>
<td width="50%"><img src="docs/screenshots/architecture-page.png" alt="Project Architecture"><br><sub>Project Architecture</sub></td>
</tr>
</table>

Full 6-page tour (Data Overview, Data Analysis, Project Architecture)
available by running the dashboard locally — see
[Running the Project](#running-the-project).

## Machine Learning Pipeline

<div align="center">
<img src="docs/pipeline-diagram.png" width="560" alt="ML pipeline — dataset to deployment">
</div>

Each stage is a standalone, reproducible artifact — not just a notebook cell:

| Stage | Output |
|---|---|
| Cleaning | `src/preprocessing.py` — leakage-safe imputation, fit on train split only |
| Feature Engineering | `artifacts/preprocessing.pkl` — versioned, fitted-once transformer |
| Feature Selection | [`reports/stage3_feature_selection.md`](reports/stage3_feature_selection.md) — 11 features cut to 8, evidence-based |
| Training | `analysis/stage4_model_comparison.py` — 4 models, identical split/CV — visualized in [`notebooks/Stage4_Model_Comparison.ipynb`](notebooks/Stage4_Model_Comparison.ipynb) |
| Evaluation | [`notebooks/Stage5_Model_Evaluation.ipynb`](notebooks/Stage5_Model_Evaluation.ipynb) — calibration, errors, explainability |
| Deployment | [`reports/stage6_deployment_plan.md`](reports/stage6_deployment_plan.md) — versioning, rollback, zero-downtime plan |

**Feature Selection, in detail** — how Stages 1-3 connect:

<div align="center">
<img src="docs/feature-selection-journey.png" width="560" alt="Feature selection journey — 11 features to 8, Stages 1-3">
</div>

## Model Performance

| Model | Accuracy | F1 | ROC-AUC | Features |
|---|---|---|---|---|
| **Logistic Regression (production)** | **0.832** | **0.792** | **0.865** | **8** |
| Random Forest (previous production) | 0.804 | 0.759 | 0.853 | 11 |
| Gradient Boosting | 0.804 | 0.729 | 0.841 | 8 |
| Random Forest (untuned, new schema) | 0.788 | 0.729 | 0.839 | 8 |

Logistic Regression beat the previously-deployed, tuned Random Forest —
**without any hyperparameter search of its own** — while training 17x
faster and showing no overfitting. Full controlled-comparison methodology:
[`reports/stage4_model_comparison.md`](reports/stage4_model_comparison.md).

## Engineering Highlights

- Preprocessing is fit once on the training split and persisted as a versioned artifact — never refit at inference.
- The production feature schema removed three perfectly-collinear features (VIF = infinite), found by direct measurement, not guesswork.
- Four models were compared under one identical split, cross-validation strategy, and evaluation protocol — not tuned unevenly and cherry-picked.
- A real cross-artifact version-drift bug was found, fixed, and turned into a permanent regression test enforced in CI.
- The API's client contract (`/predict`) stayed backward-compatible through a full model and schema change — `/v2/predict` is additive, not breaking.
- 76 automated tests cover preprocessing, validation, model integration, the API, and the dashboard's data layer, with 100% coverage on the core preprocessing module.

## Project Structure

```
Titanic-End-to-End/
├── analysis/          Reproducible stage scripts (data audit, model comparison)
├── api/               FastAPI service — schemas.py, main.py
├── artifacts/         Versioned preprocessing + feature schema
├── dashboard/         6-page Streamlit application
├── docs/              Diagrams and dashboard screenshots (this README's assets)
├── data/              Raw Titanic dataset
├── model/             model_v1.pkl, model_v2.pkl, train.py, train_v2.py
├── notebooks/         Stage 2 (EDA), Stage 4 (comparison), Stage 5 (evaluation) notebooks
├── reports/           Stage-by-stage engineering write-ups
├── src/               TitanicPreprocessor — shared feature engineering
├── tests/             unit/, integration/, e2e/ — 76 tests
├── .github/workflows/ CI pipeline
└── Dockerfile
```

## Installation

```bash
git clone <repository-url>
cd Titanic-End-to-End
pip install -r requirements-dev.txt
```

## Running the Project

```bash
# API
uvicorn api.main:app --reload

# Dashboard
streamlit run dashboard/Home.py

# Docker
docker build -t titanic-api .
docker run -d -p 8000:8000 --name titanic-api-container titanic-api
```

<div align="center">
<img src="docs/deployment-diagram.png" width="480" alt="Deployment pipeline — Docker and AWS EC2">
</div>

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "pclass": 1, "sex": "female", "age": 29,
        "sibsp": 0, "parch": 0, "fare": 100,
        "embarked": "C", "has_cabin": true, "title": "Mrs"
      }'
```

```json
{"survived": true, "survival_probability": 0.9867}
```

Interactive docs at `/docs` once the API is running.

## Dashboard

6 pages: Home, Data Overview, Data Analysis, Model Performance, Live
Prediction, Project Architecture. Every chart and metric is computed
live from the versioned artifacts — nothing is hardcoded.

Every major chart carries a collapsed "What This Means" panel: what the
chart shows, why it matters, and what engineering decision it drove.
Live Prediction goes further, decomposing a single passenger's prediction
into signed feature contributions, counterfactuals, and similar historical
cases, all computed from the model's own coefficients — no SHAP dependency.

## Testing & CI

```bash
pytest tests/ --cov=src --cov-report=term-missing
ruff check src/ api/ analysis/ model/ tests/ dashboard/
```

76 tests (47 unit, 25 integration, 4 end-to-end) · 100% coverage on
`src/preprocessing.py` · CI (`.github/workflows/ci.yml`) rebuilds every
artifact from source and verifies cross-artifact version compatibility on
every push. Full report: [`reports/stage8_engineering_quality.md`](reports/stage8_engineering_quality.md).

## Future Improvements

- Cut over `/predict` to `model_v2.pkl` and retire the version gap documented in Stage 6/8
- Automate the EC2 deployment step currently run by hand
- Add prediction monitoring and basic model-drift detection in production
- Revisit the schema with a larger, non-Kaggle passenger dataset if one becomes available

## License

[MIT](LICENSE)

---

## Why This Project Stands Out

- **A real leakage bug was found and fixed, not assumed away** — the original pipeline fit imputation on the full dataset before splitting; `TitanicPreprocessor` fits on the training split only, with a passing regression test proving it.
- **Feature selection is evidence-based, with the receipts kept** — VIF proved `SibSp`+`Parch`+`FamilySize` are exactly collinear (VIF = infinite), not just "probably correlated."
- **The winning model beat a tuned baseline while untuned** — Logistic Regression outperforms the production Random Forest without a hyperparameter search of its own, the least-confounded way that result could land.
- **A production incident was caught before it shipped** — Stage 7 introduced a cross-artifact version mismatch; it was found by inspection, fixed, and converted into a CI-enforced regression test so it can't recur silently.
- **The API never broke, through a full model swap** — `/predict`'s request and response contract is unchanged across an 11-feature-to-8-feature schema change and a Random-Forest-to-Logistic-Regression model swap.
- **Every prediction explains itself** — coefficients decompose into signed, per-feature contributions for any single passenger, with no SHAP dependency required.
- **The test suite documents a known gap instead of hiding it** — the end-to-end tests explicitly assert that the API (v1) and dashboard (v2) are allowed to disagree today, because that's the true state of the migration, not a bug to paper over.
- **Artifacts are rebuilt in CI, not trusted as committed binaries** — `model_v2.pkl` and the preprocessing artifact are regenerated from `data/train.csv` on every push, so CI catches reproducibility breaks, not just code style.
- **Rollback is a config change, not a redeploy** — model version selection is designed around an environment variable and two frozen `.pkl` files, so reverting a bad model doesn't require a code revert.
- **Nine stages of engineering decisions are each independently documented** — every claim in this README is backed by a stage-specific report or notebook, not a single sprawling changelog.
