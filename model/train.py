from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data and engineer features (matches notebooks/Udbhav_Statistical_Analysis.ipynb)."""
    data = pd.read_csv(DATA_PATH)

    data["HasCabin"] = data["Cabin"].notna().astype(int)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")
    rare_titles = data["Title"].value_counts()[lambda x: x < 10].index.tolist()
    data["Title"] = data["Title"].replace(rare_titles, "Rare")
    data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Title-based median beats a single global median (Master ~3.5 vs Mr ~30)
    data["Age"] = data["Age"].fillna(data.groupby("Title")["Age"].transform("median"))

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "HasCabin", "FamilySize", "IsAlone", "Title",
    ]
    X = data[feature_cols]
    y = data["Survived"]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing step of the pipeline.

    Numeric columns: median-imputed then scaled. Categorical columns:
    mode-imputed then one-hot encoded. HasCabin/IsAlone are already
    binary and pass through unchanged.
    """
    numeric_features = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
    categorical_features = ["Sex", "Embarked", "Pclass", "Title"]
    passthrough_features = ["HasCabin", "IsAlone"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("pass", "passthrough", passthrough_features),
    ])

    return preprocessor


RF_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__max_depth": [4, 6, 8, 10, None],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__min_samples_split": [2, 5, 10],
}


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        preprocessor: ColumnTransformer) -> RandomForestClassifier:
    """Search RandomForest hyperparameters via 5-fold CV, return the best estimator."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
    ])

    search = RandomizedSearchCV(
        pipeline, RF_PARAM_DISTRIBUTIONS, n_iter=20, scoring="f1",
        cv=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"Best RF params: {search.best_params_}")
    print(f"Best RF CV F1:  {search.best_score_:.4f}")

    return search.best_estimator_.named_steps["classifier"]


LR_PARAM_DISTRIBUTIONS = {
    "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "classifier__l1_ratio": [0.0, 1.0],  # 0 = l2 penalty, 1 = l1 penalty
}


def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                              preprocessor: ColumnTransformer) -> LogisticRegression:
    """Search LogisticRegression hyperparameters via 5-fold CV, return the best estimator."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, solver="liblinear"
        )),
    ])

    search = RandomizedSearchCV(
        pipeline, LR_PARAM_DISTRIBUTIONS, n_iter=12, scoring="f1",
        cv=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"Best LR params: {search.best_params_}")
    print(f"Best LR CV F1:  {search.best_score_:.4f}")

    return search.best_estimator_.named_steps["classifier"]


def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                    preprocessor: ColumnTransformer,
                    tuned_rf: RandomForestClassifier,
                    tuned_lr: LogisticRegression) -> tuple[str, dict]:
    """
    Compare candidate models via 5-fold cross-validation on the training
    set only. Returns the best model's name and the candidates dict.
    """
    candidates = {
        "logistic_regression": tuned_lr,
        "random_forest": tuned_rf,
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    print("\nModel comparison (5-fold cross-validation, training set only):")
    print("-" * 60)

    cv_results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        cv_results[name] = scores.mean()
        print(f"{name:22s} mean F1 = {scores.mean():.4f}  (+/- {scores.std():.4f})")

    best_model_name = max(cv_results, key=cv_results.get)
    print(f"\nSelected model: {best_model_name}")
    return best_model_name, candidates


def evaluate_on_test_set(pipeline: Pipeline, X_test: pd.DataFrame,
                          y_test: pd.Series) -> None:
    """Evaluate the final trained pipeline on the untouched test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))


def main() -> None:
    X, y = load_and_prepare_data()

    # Stratified split preserves the ~38% survival rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()

    tuned_rf = tune_random_forest(X_train, y_train, preprocessor)
    tuned_lr = tune_logistic_regression(X_train, y_train, preprocessor)
    best_model_name, candidates = compare_models(X_train, y_train, preprocessor, tuned_rf, tuned_lr)

    final_classifier = candidates[best_model_name]

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", final_classifier),
    ])

    final_pipeline.fit(X_train, y_train)

    evaluate_on_test_set(final_pipeline, X_test, y_test)

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nModel pipeline saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
    