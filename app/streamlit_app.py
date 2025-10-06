#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st
import skops.io as sio
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
#

# =========================
# Config
# =========================
USE_MLFLOW = os.getenv("USE_MLFLOW", "0") == "1"
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "DrugPipeline")
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "drug_pipeline.skops"

st.set_page_config(page_title="Drug Classification", page_icon="💊", layout="centered")
st.title("Drug Classification 💊")


# =========================
# MLflow HEADLESS CONFIG (no OAuth)
# =========================
def configure_mlflow() -> str:
    """Configura MLflow in modalità headless verso DagsHub."""
    owner = os.getenv("DAGSHUB_USERNAME")
    repo = os.getenv("DAGSHUB_REPO")
    token = os.getenv("DAGSHUB_TOKEN")

    if not (owner and repo and token):
        raise RuntimeError("DAGSHUB_* env mancanti (USERNAME, REPO, TOKEN)")

    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Drug_Classification")
    try:
        mlflow.set_experiment(exp_name)
    except Exception:
        pass

    return tracking_uri


@st.cache_resource(show_spinner=True)
def load_pipeline_from_mlflow(model_name: str):
    """Carica la versione @production dal registry MLflow."""
    tracking_uri = configure_mlflow()
    import mlflow.sklearn
    model_uri = f"models:/{model_name}@production"
    pipe = mlflow.sklearn.load_model(model_uri)
    return pipe, f"MLflow · {model_name} @production ({tracking_uri})"


@st.cache_resource(show_spinner=True)
def load_pipeline_from_skops(path: Path):
    """Carica pipeline locale .skops con gestione trusted types."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Modello locale non trovato: {path}")
    try:
        pipe = sio.load(path)
    except sio.exceptions.UntrustedTypesFoundException:
        untrusted = sio.get_untrusted_types(file=path)
        pipe = sio.load(path, trusted=untrusted)
    return pipe, f"Local skops · {path.name}"

if st.sidebar.button("🔄 Force Reload Model"):
    st.cache_resource.clear()
    st.rerun()
    
st.write("### Debug Info")
st.write(f"USE_MLFLOW: {os.getenv('USE_MLFLOW', '0')}")
st.write(f"DAGSHUB_USERNAME: {os.getenv('DAGSHUB_USERNAME', 'NOT SET')}")
st.write(f"DAGSHUB_REPO: {os.getenv('DAGSHUB_REPO', 'NOT SET')}")
st.write(f"DAGSHUB_TOKEN: {'SET' if os.getenv('DAGSHUB_TOKEN') else 'NOT SET'}")

def load_pipeline_prod() -> Tuple[object, str]:
    """Tenta MLflow se USE_MLFLOW=1, altrimenti fallback locale."""
    use_mlflow = os.getenv("USE_MLFLOW", "0") == "1"
    st.write(f"🔍 USE_MLFLOW={use_mlflow}")  # Debug
    
    if use_mlflow:
        st.write("🚀 Tentativo caricamento da MLflow...")  # Debug
        try:
            result = load_pipeline_from_mlflow(REGISTERED_MODEL_NAME)
            st.write("✅ Modello caricato da MLflow!")  # Debug
            return result
        except Exception as e:
            st.warning(f"Caricamento da MLflow fallito: {e}\n→ Fallback locale.")
    else:
        st.write("📁 Caricamento modello locale (USE_MLFLOW non attivo)")  # Debug
    
    return load_pipeline_from_skops(MODEL_PATH)


# =========================
# Carica modello
# =========================
try:
    pipe, source_label = load_pipeline_prod()
    st.caption(f"Model source: {source_label}")
except Exception as e:
    st.error(f"Errore nel caricamento del modello: {e}")
    st.stop()


# =========================
# Expected columns detection
# =========================
def get_expected_columns(pipeline) -> List[str]:
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            if hasattr(step, "transformers_") and hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return ["Age", "Sex", "Blood Pressure", "Cholesterol", "Na_to_K"]


EXPECTED_COLS = get_expected_columns(pipe)
st.caption(f"Colonne attese dal modello: {EXPECTED_COLS}")


# =========================
# Input helpers
# =========================
def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "").replace(" ", "")


UI_TO_VALUE_KEYS = {
    "age": "age",
    "sex": "sex",
    "bloodpressure": "blood_pressure",
    "cholesterol": "cholesterol",
    "natok": "na_to_k_ratio",
}


def build_input_dataframe(
    age: int,
    sex: str,
    blood_pressure: str,
    cholesterol: str,
    na_to_k_ratio: float,
    expected_cols: List[str],
) -> pd.DataFrame:
    """Costruisce un DataFrame 1-riga con le colonne attese."""
    ui_values: Dict[str, object] = {
        "age": int(age),
        "sex": str(sex),
        "blood_pressure": str(blood_pressure),
        "cholesterol": str(cholesterol),
        "na_to_k_ratio": float(na_to_k_ratio),
    }

    row: Dict[str, object] = {}
    for col in expected_cols:
        key_norm = _norm(col)
        if key_norm in UI_TO_VALUE_KEYS and UI_TO_VALUE_KEYS[key_norm] in ui_values:
            row[col] = ui_values[UI_TO_VALUE_KEYS[key_norm]]
        elif key_norm in ("natokratio", "natok", "sodiumpotassium", "sodiumpotassiumratio"):
            row[col] = ui_values["na_to_k_ratio"]
        elif key_norm in ("bp", "bloodpres", "bloodpressurelevel"):
            row[col] = ui_values["blood_pressure"]
        else:
            row[col] = None

    df = pd.DataFrame([row], columns=expected_cols)
    for c in df.columns:
        cn = _norm(c)
        if cn == "age":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        if cn in ("natok", "natokratio", "sodiumpotassium", "sodiumpotassiumratio"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Prediction
# =========================
def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    X = build_input_dataframe(age, sex, blood_pressure, cholesterol, na_to_k_ratio, EXPECTED_COLS)
    pred = pipe.predict(X)[0]

    proba_df = None
    if hasattr(pipe, "predict_proba"):
        try:
            probs = pipe.predict_proba(X)[0]
            classes = getattr(pipe, "classes_", None)
            if classes is None and hasattr(pipe, "named_steps"):
                for step in reversed(list(pipe.named_steps.values())):
                    if hasattr(step, "classes_"):
                        classes = step.classes_
                        break
            if classes is not None:
                proba_df = (
                    pd.DataFrame({"class": classes, "probability": probs})
                    .sort_values("probability", ascending=False)
                    .reset_index(drop=True)
                )
        except Exception:
            pass
    return str(pred), proba_df


# =========================
# UI
# =========================
with st.form("input-form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 15, 74, 30, 1)
        sex = st.radio("Sex", ["M", "F"], horizontal=True, index=0)
        na_to_k_ratio = st.slider("Na_to_K", 6.2, 38.2, 15.4, 0.1)
    with c2:
        blood_pressure = st.radio("Blood Pressure", ["HIGH", "LOW", "NORMAL"], index=0)
        cholesterol = st.radio("Cholesterol", ["HIGH", "NORMAL"], index=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    label, proba = predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio)
    st.success(f"Predicted Drug: **{label}**")
    if proba is not None:
        st.subheader("Class probabilities")
        st.dataframe(proba)
        st.bar_chart(proba.set_index("class"))