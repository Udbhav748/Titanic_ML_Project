"""Predict survival for a single passenger using the trained pipeline."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

MODEL_LABELS = {
    "RandomForestClassifier": "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "GradientBoostingClassifier": "Gradient Boosting",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def test_metrics():
    """Recreate the same held-out split as model/train.py to report real performance."""
    data = pd.read_csv(BASE_DIR / "data" / "train.csv")
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")
    rare_titles = data["Title"].value_counts()[lambda x: x < 10].index.tolist()
    data["Title"] = data["Title"].replace(rare_titles, "Rare")
    data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    data["Age"] = data["Age"].fillna(data.groupby("Title")["Age"].transform("median"))

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "HasCabin", "FamilySize", "IsAlone", "Title",
    ]
    X, y = data[feature_cols], data["Survived"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = load_model()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


model = load_model()
classifier = model.named_steps["classifier"]
model_label = MODEL_LABELS.get(type(classifier).__name__, type(classifier).__name__)
metrics = test_metrics()

st.title("Predict Survival")
st.markdown(f"Fill in passenger details and get a survival prediction from the trained **{model_label}** model.")

m1, m2, m3 = st.columns(3)
m1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Test F1", f"{metrics['f1']:.3f}")
m3.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
st.caption("Performance on the same 20% held-out test split used in model/train.py.")

st.markdown("---")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
        sex = st.selectbox("Sex", ["female", "male"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
        fare = st.number_input("Fare ($)", min_value=0.0, value=32.0)

    with col3:
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        has_cabin = st.checkbox("Has Cabin Record")
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    family_size = sibsp + parch + 1

    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "HasCabin": int(has_cabin),
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "Title": title,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    result_col, prob_col = st.columns([2, 1])

    with result_col:
        if prediction == 1:
            st.success("Predicted: Survived")
        else:
            st.error("Predicted: Did Not Survive")
        st.progress(probability)

    with prob_col:
        st.metric("Survival Probability", f"{probability:.1%}")
