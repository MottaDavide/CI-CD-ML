#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import matplotlib.pyplot as plt
import skops.io as sio

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- DagsHub + MLflow ---
import dagshub
import mlflow
import mlflow.sklearn

RANDOM_STATE = 42
DTYPE = {
    "Age": int, "Sex": object, "BP": object, "Cholesterol": object, "Na_to_K": float, "Drug": object
}

# ---- DagsHub init ----
OWNER = os.getenv("DAGSHUB_USERNAME")
REPO  = os.getenv("DAGSHUB_REPO")
TOKEN = os.getenv("DAGSHUB_TOKEN")

if OWNER and REPO and TOKEN:
    # setta esplicitamente l’autenticazione via token
    os.environ["MLFLOW_TRACKING_USERNAME"] = OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{OWNER}/{REPO}.mlflow"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Drug_Classification"))
    print(f"✅ MLflow configurato su DagsHub: {OWNER}/{REPO}")
else:
    mlflow.set_tracking_uri("file://mlruns")
    print("⚠️ DagsHub env non trovate → uso tracking locale (mlruns/)")

os.environ["MLFLOW_TRACKING_USERNAME"] = OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN


mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Drug_Classification"))

# ---- IO dirs ----
Path("results").mkdir(parents=True, exist_ok=True)
Path("model").mkdir(parents=True, exist_ok=True)

# ---- Data ----
df = pd.read_csv("data/drug200.csv", dtype=DTYPE)
X = df.drop("Drug", axis=1)
y = df["Drug"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

num_t = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_t = Pipeline([("oe", OrdinalEncoder())])
pre = ColumnTransformer([("num", num_t, num_cols), ("cat", cat_t, cat_cols)])

#rf  = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=5, class_weight="balanced", n_jobs=-1, max_depth=10)
rf  = DecisionTreeClassifier(random_state=RANDOM_STATE)
clf = Pipeline([("preprocessor", pre), ("classifier", rf)])

with mlflow.start_run() as run:
    # ---- Train ----
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    # ---- Log params/metrics ----
    params = clf.named_steps['classifier'].get_params()
    mlflow.log_params(params)
    #mlflow.log_param("n_estimators", 100)
    #mlflow.log_param("class_weight", "balanced")
    #mlflow.log_param("max_depth", 1)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    # ---- Artifacts locali ----
    (Path("results") / "metrics.txt").write_text(f"Accuracy: {acc}\nF1: {f1}\n", encoding="utf-8")
    (Path("results") / "metrics.json").write_text(json.dumps({"accuracy": acc, "f1": f1}, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    fig = disp.plot().figure_
    fig.savefig("results/model_results.png", dpi=120)
    plt.close(fig)

    # ---- Log artifacts su MLflow (DagsHub) ----
    mlflow.log_artifact("results/metrics.txt")
    mlflow.log_artifact("results/metrics.json")
    mlflow.log_artifact("results/model_results.png")

    # ---- Modello: salva locale + registra su registry DagsHub ----
    sio.dump(clf, "model/drug_pipeline.skops")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sio.dump(clf, f"model/drug_pipeline_{ts}.skops")

    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example = X_test.iloc[:5],
        registered_model_name=os.getenv("REGISTERED_MODEL_NAME", "DrugPipeline")
    )

    print(f"Run ID: {run.info.run_id}  |  acc={acc:.3f}  f1={f1:.3f}")