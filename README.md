# Titanic Survival Prediction

End-to-end machine learning project on the classic [Titanic dataset](https://www.kaggle.com/c/titanic/data) — from raw data to a deployed prediction service. It covers exploratory data analysis with formal hypothesis testing, a tuned Random Forest classifier, an interactive Streamlit dashboard, and a FastAPI service containerized with Docker. Built to demonstrate a complete, production-style ML workflow rather than just a notebook.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Results](#results)
- [Screenshots](#screenshots)
- [Getting Started](#getting-started)
- [Docker](#docker)
- [Possible Improvements](#possible-improvements)
- [License](#license)

## Features

- **EDA notebook** with formal hypothesis testing (Welch's t-test on fare vs. survival, ANOVA, chi-square, Cramer's V)
- **Interactive dashboard** (Streamlit, 3 pages):
  - **Overview** — KPIs and headline charts
  - **Analysis** — full statistical write-up, computed live: outlier boxplots, correlation heatmap, bivariate breakdowns
  - **Prediction** — single-passenger prediction form
- **Random Forest classifier**, tuned with `RandomizedSearchCV`, outperforming Logistic Regression and Gradient Boosting in 5-fold cross-validation, wrapped in a single scikit-learn pipeline
- **Feature engineering**: title extraction, family size, "has cabin" flag
- **FastAPI service** exposing `/predict` and `/health` (root path redirects to `/docs`)
- **Dockerized API** — slim image built from API-only dependencies (~712MB)

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Modeling | Python, pandas, NumPy, scikit-learn, SciPy |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Pydantic, Uvicorn |
| Deployment | Docker |

## Project Structure

```
Titanic_ML_Project/
├── data/train.csv
├── notebooks/Udbhav_Statistical_Analysis.ipynb
├── dashboard/
│   ├── Overview.py
│   └── pages/
│       ├── 1_Analysis.py   # notebook write-up
│       └── 2_Prediction.py # single-passenger prediction
├── model/
│   ├── train.py
│   ├── evaluate.py
│   └── model.pkl
├── api/
│   ├── main.py
│   └── schemas.py
├── images/
├── Dockerfile
├── requirements.txt      # full dev environment (dashboard + API + notebook)
└── requirements-api.txt  # slim deps for the Docker image (API only)
```

## Results

Random Forest (`class_weight="balanced"`, tuned via `RandomizedSearchCV`) was selected over Logistic Regression and Gradient Boosting — mean F1 0.7711 vs. 0.7591 vs. 0.7493 in cross-validation.

**Test set performance** (179 held-out passengers):

| Metric | Value |
|---|---|
| Accuracy | 80.5% |
| Precision | 0.72 |
| Recall | 0.80 |
| F1-score | 0.7586 |
| ROC-AUC | 0.8526 |

**Confusion matrix:**

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| Actual: No | 89 | 21 |
| Actual: Yes | 14 | 55 |

Full statistical write-up is in `notebooks/Udbhav_Statistical_Analysis.ipynb`, also reproduced live on the dashboard's **Analysis** page.

## Screenshots

| Dashboard | Analysis |
|---|---|
| ![Dashboard](images/dashboard.png) | ![Analysis](images/analysis.png) |

| Prediction | FastAPI Docs |
|---|---|
| ![Prediction](images/prediction.png) | ![FastAPI Docs](images/fastapi_docs.png) |

## Getting Started

**1. Clone and install:**

```bash
git clone https://github.com/Udbhav748/Titanic_ML_Project.git
cd Titanic_ML_Project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the dashboard** (Overview, Analysis, and Prediction pages are in the sidebar):

```bash
streamlit run dashboard/Overview.py
```

**3. Train the model** (optional — `model.pkl` is already included):

```bash
python model/train.py
```

**4. Run the API:**

```bash
uvicorn api.main:app --reload
```

Docs available at `http://127.0.0.1:8000/docs`.

## Docker

Build and run the API as a container:

```bash
docker build -t titanic-api .
docker run -d -p 8000:8000 --name titanic-api-container titanic-api
```

Test it:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 100.0,
    "embarked": "C",
    "has_cabin": true,
    "title": "Mrs"
  }'
```

## Possible Improvements

- Try XGBoost/LightGBM (currently comparing against scikit-learn's `GradientBoostingClassifier`)
- Add tests for the API and preprocessing pipeline
- Set up CI/CD and a cloud-hosted deployment

## License

Built for academic evaluation.
