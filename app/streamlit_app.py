#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import skops.io as sio

# =========================
# Config
# =========================
USE_MLFLOW = os.getenv("USE_MLFLOW", "0") == "1"
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "DrugPipeline")

# Percorso fallback locale (.skops)
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "drug_pipeline.skops"

st.set_page_config(page_title="Drug Classification", page_icon="💊", layout="centered")
st.title("Drug Classification 💊")


# =========================
# Loading helpers
# =========================
def _init_dagshub_mlflow() -> None:
    """
    Inizializza l'endpoint MLflow su DagsHub usando le env:
      DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN
    """
    owner = os.getenv("DAGSHUB_USERNAME")
    repo = os.getenv("DAGSHUB_REPO")
    token = os.getenv("DAGSHUB_TOKEN")

    if not (owner and repo and token):
        raise RuntimeError("DAGSHUB_* env non configurate (USERNAME, REPO, TOKEN)")

    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # Import qui per non imporre dipendenze quando si usa solo il modello locale
    import dagshub  # type: ignore
    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)


@st.cache_resource(show_spinner=True)
def load_pipeline_from_mlflow(model_name: str):
    """
    Carica SEMPRE la versione 'Production' dal registry MLflow.
    Cache Streamlit: ricarica solo se cambia model_name o env.
    """
    _init_dagshub_mlflow()
    import mlflow  # type: ignore
    import mlflow.sklearn  # type: ignore

    uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(uri)
    return model


@st.cache_resource(show_spinner=True)
def load_pipeline_from_skops(path: Path):
    """
    Carica il pipeline da file .skops locale con gestione tipi 'trusted'.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Modello locale non trovato: {path}")
    try:
        return sio.load(path)
    except sio.exceptions.UntrustedTypesFoundException:
        untrusted = sio.get_untrusted_types(file=path)
        return sio.load(path, trusted=untrusted)


def load_pipeline_prod() -> Tuple[object, str]:
    """
    Logica di scelta sorgente modello:
      - se USE_MLFLOW=1 -> tenta MLflow Production, altrimenti fallback locale
      - se USE_MLFLOW=0 -> usa solo locale
    Ritorna: (pipeline, source_str)
    """
    if USE_MLFLOW:
        try:
            pipe = load_pipeline_from_mlflow(REGISTERED_MODEL_NAME)
            return pipe, f"MLflow/DagsHub · {REGISTERED_MODEL_NAME} @ Production"
        except Exception as e:
            st.warning(
                f"Caricamento da MLflow fallito: {e}\n"
                f"Passo al fallback locale: {MODEL_PATH}"
            )
    # Fallback locale
    pipe = load_pipeline_from_skops(MODEL_PATH)
    return pipe, f"Local skops · {MODEL_PATH.name}"


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
    """
    Prova a recuperare i nomi colonne attesi dal pipeline/trasformatori.
    Ritorna una lista di fallback se non presenti.
    """
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            if hasattr(step, "transformers_"):  # ColumnTransformer
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)

    # Fallback robusto: adatta ai tuoi label UI se servono
    return ["Age", "Sex", "Blood Pressure", "Cholesterol", "Na_to_K"]


EXPECTED_COLS = get_expected_columns(pipe)
st.caption(f"Colonne attese dal modello: {EXPECTED_COLS}")


# =========================
# Input mapping helpers
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
    """
    Costruisce un DataFrame 1-riga con esattamente le colonne attese dal pipeline.
    Tenta il match per normalizzazione del nome.
    """
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
        else:
            if key_norm in ("natokratio", "natok", "sodiumpotassium", "sodiumpotassiumratio"):
                row[col] = ui_values["na_to_k_ratio"]
            elif key_norm in ("bp", "bloodpres", "bloodpressurelevel"):
                row[col] = ui_values["blood_pressure"]
            else:
                row[col] = None  # estremo fallback, il preprocessor potrebbe imputare

    df = pd.DataFrame([row], columns=expected_cols)

    # Tipi utili
    for c in df.columns:
        cn = _norm(c)
        if cn == "age":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        if cn in ("natok", "natokratio", "sodiumpotassium", "sodiumpotassiumratio"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Predict wrapper
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
            proba_df = None
    return str(pred), proba_df


# =========================
# UI
# =========================
with st.form("input-form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", min_value=15, max_value=74, step=1, value=30)
        sex = st.radio("Sex", ["M", "F"], horizontal=True, index=0)
        na_to_k_ratio = st.slider("Na_to_K", min_value=6.2, max_value=38.2, step=0.1, value=15.4)
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