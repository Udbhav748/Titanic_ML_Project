# Titanic Survival Prediction — End-to-End ML Project

An end-to-end machine learning project: statistical analysis, an interactive dashboard, a tuned classifier, a live API, and a cloud deployment — all built around one dataset, one connected pipeline.

**Live Demo:**
- API Docs: `http://13.51.85.67:8000/docs`
- Dashboard: `http://13.51.85.67:8501`

---

## Phase 1 — Statistical Analysis (EDA)

- Loaded and inspected 891 passengers, 12 columns; found missing data in `Age` (~20%), `Cabin` (~77%), `Embarked` (2 rows)
- **Univariate analysis:** histograms + skewness per column
  - `Fare` highly skewed → log-transformed
  - `Age` roughly symmetric → median imputation
- **Outlier detection:** boxplots + IQR rule
  - Retained all outliers — real passengers (e.g. genuine 1st-class fares), not errors
- **Bivariate analysis:** correlation heatmap + survival rate by Sex, Class, Embarked, Age Group
  - Caught a **confounding variable**: Embarked's apparent effect was really just Pclass composition at that port
- **Hypothesis testing:** two-sample t-test (Welch's, after checking variance with Levene's test) — survivors paid significantly more fare (**p < 0.001**)
- Summarized into plain-English business insights

---

## Phase 2 — Interactive Dashboard (Streamlit)

- Converted top insights into a live, filterable dashboard
- **KPI row:** total passengers, survival rate, avg fare by outcome
- **1 filter:** Passenger Class (selectbox) — chosen since it had the clearest effect on survival
- **3 Plotly charts:** survival by Sex, by Age Group, and a Fare comparison — each with a plain-English insight caption
- Later expanded into **3 pages**: Overview, full live Analysis, and a Prediction page (loads the same `model.pkl` as the API)

---

## Phase 3 — Machine Learning Model

**Feature engineering:**
- `Title` extracted from `Name` (Mr/Mrs/Miss/Master) → also improved Age imputation (median per title group)
- `FamilySize`, `IsAlone`
- `HasCabin` binary flag from the sparse `Cabin` column

**Modeling approach:**
- Single `sklearn.Pipeline` — preprocessing + model bundled as one artifact (prevents train/serve mismatch)
- Stratified 80/20 train/test split
- Compared **3 models** via 5-fold CV: Logistic Regression, Random Forest, Gradient Boosting → **Random Forest selected**
- Tuned with **RandomizedSearchCV** instead of guessed hyperparameters
- Diagnosed a precision/recall imbalance → fixed with **`class_weight="balanced"`**

**Final results (test set):**

| Metric | Score |
|---|---|
| Accuracy | 80.5% |
| Precision | 0.72 |
| Recall | 0.80 |
| F1-score | 0.7586 |
| ROC-AUC | 0.8526 |

Confusion matrix:

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| Actual: No | 89 | 21 |
| Actual: Yes | 14 | 55 |

Saved as a single `model.pkl` via `joblib`.

---

## Phase 4 — FastAPI Service

- Built a `PassengerInput` **Pydantic schema** — typed, constrained fields; invalid input auto-rejected (422)
- **Endpoints:** `/health` (service + model status) and `/predict` (returns prediction + probability)
- Model loaded once at startup, not per-request
- Global exception handling — no raw errors leaked to callers
- Verified through auto-generated **Swagger UI** (`/docs`)

---

## Phase 5 — Docker

- `python:3.11-slim` base image
- Dependencies installed **before** code copied → faster rebuilds via layer caching
- Slimmer `requirements-api.txt` (API-only deps) instead of full dev requirements → smaller image (712MB)
- Runs as a **non-root user** for security
- Bound to `0.0.0.0:8000` (not `127.0.0.1`) so it's reachable externally
- Tested via `curl` and Python `requests`

---

## Phase 6 — GitHub

- Full project pushed with proper structure, README, and `requirements.txt`
- Used **branches + pull requests** for larger changes (not direct edits to `main`)
- Fixed a real bug: a **stale git remote** pointing to an old repo — caught via `git remote -v`

---

## Phase 7 — AWS EC2 Deployment (Stretch Goal)

- Launched a free-tier-eligible **t3.micro** Ubuntu instance (eu-north-1)
- Key pair (`.pem`) for SSH — no passwords
- **Security group** opened for ports 22 (SSH), 8000 (API), 8501 (dashboard)
- Installed Docker on the instance, cloned the repo from GitHub
- **API** runs containerized (Docker); **Dashboard** runs directly in a Python venv via `nohup` — a deliberate architecture choice (always-on service vs. on-demand tool)
- Verified both live from an external browser, not just internally on the server

---

## Key Techniques Used

`Missing value imputation` · `Log transformation` · `IQR outlier detection` · `Hypothesis testing (t-test, Levene's)` · `Correlation analysis` · `Feature engineering` · `One-hot encoding` · `sklearn Pipeline` · `Stratified split` · `Cross-validation` · `RandomizedSearchCV` · `Class imbalance handling` · `REST API design` · `Data validation (Pydantic)` · `Containerization (Docker)` · `Cloud deployment (EC2)` · `Version control (Git/GitHub)`

---

## Tech Stack

Python · pandas · scikit-learn · SciPy · Streamlit · Plotly · FastAPI · Pydantic · Docker · AWS EC2

---

## Project Structure

```
Titanic-End-to-End/
├── data/train.csv
├── notebooks/Udbhav_Statistical_Analysis.ipynb
├── dashboard/
│   ├── Overview.py
│   └── pages/1_Analysis.py, 2_Prediction.py
├── model/train.py, model.pkl
├── api/main.py, schemas.py
├── Dockerfile
├── requirements.txt, requirements-api.txt
└── README.md
```

---

## How to Run

```bash
# Local — API
uvicorn api.main:app --reload

# Local — Dashboard
streamlit run dashboard/Overview.py

# Docker
docker build -t titanic-api .
docker run -d -p 8000:8000 --name titanic-api-container titanic-api
```

---

## Future Improvements

- Try XGBoost / LightGBM
- Automated tests for API and pipeline
- CI/CD pipeline
- Elastic IP + systemd/restart policies for full reboot persistence
