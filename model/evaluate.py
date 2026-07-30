import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Reload data the same way train.py does
data = pd.read_csv("data/train.csv")

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
X = data[feature_cols]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("model/model.pkl")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))
