# Titanic Survival Prediction

End-to-end ML project on the Titanic dataset: EDA, a trained model, a Streamlit dashboard, and a FastAPI service you can run in Docker.

## Project structure

```
Titanic-End-to-End/
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

## What's here

- EDA notebook with hypothesis testing (Welch's t-test on fare vs. survival)
- Streamlit dashboard with 3 pages: Overview (KPIs/charts), Analysis (notebook write-up), Prediction (single-passenger form)
- Random Forest classifier (tuned with RandomizedSearchCV, beat Logistic Regression and Gradient Boosting in 5-fold CV) wrapped in a single scikit-learn pipeline
- Feature engineering: title, family size, cabin flag
- FastAPI service exposing `/predict` and `/health` (root path redirects to `/docs`)
- Dockerfile for running the API as a slim container (API-only deps, ~712MB)

## Results

Random Forest (`class_weight="balanced"`, tuned via RandomizedSearchCV) selected over Logistic Regression and Gradient Boosting, mean F1 0.7711 vs 0.7591 vs 0.7493 in cross-validation.

Test set (179 held-out passengers):

| Metric | Value |
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

Full analysis is in `notebooks/Udbhav_Statistical_Analysis.ipynb`.

## Screenshots

![Dashboard](images/dashboard.png)
![Analysis](images/analysis.png)
![Prediction](images/prediction.png)
![FastAPI Docs](images/fastapi_docs.png)

## Running it

Clone and install:

```bash
git clone <your-repo-url>
cd Titanic-End-to-End
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dashboard (Overview, Analysis, and Prediction pages are in the sidebar):

```bash
streamlit run dashboard/Overview.py
```

Train the model (optional, `model.pkl` is already included):

```bash
python model/train.py
```

API:

```bash
uvicorn api.main:app --reload
```

Docs at `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker build -t titanic-api .
docker run -d -p 8000:8000 --name titanic-api-container titanic-api
```

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

## Tech stack

Python, pandas, scikit-learn, SciPy, Streamlit, Plotly, FastAPI, Pydantic, Docker

## Possible improvements

- Try XGBoost/LightGBM (currently comparing against sklearn's GradientBoostingClassifier)
- Tests for the API and preprocessing pipeline
- CI/CD and a cloud-hosted deployment

## License

Built for academic evaluation.
