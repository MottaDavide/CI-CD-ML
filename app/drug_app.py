import streamlit as st
import skops.io as sio
import pandas as pd
from typing import List, Dict

MODEL_PATH = "./model/drug_pipeline.skops"

st.set_page_config(page_title="Drug Classification", page_icon="💊", layout="centered")
st.title("Drug Classification 💊")

# --------- LOAD PIPELINE (skops >= 0.10) ----------
@st.cache_resource
def load_pipeline(path: str):
    try:
        return sio.load(path)  # primo tentativo
    except sio.exceptions.UntrustedTypesFoundException:
        # da skops 0.10 si chiama SENZA argomenti
        untrusted = sio.get_untrusted_types(file = path)
        # Se vuoi, mostra a video per audit
        st.info(f"Carico il modello autorizzando questi tipi: {untrusted}")
        return sio.load(path, trusted=untrusted)

pipe = load_pipeline(MODEL_PATH)

# --------- EXPECTED COLUMNS DETECTION ----------
def get_expected_columns(pipeline) -> List[str]:
    """
    Prova a recuperare i nomi colonne attesi dal pipeline/trasformatori.
    Ritorna una lista di fallback se non presenti.
    """
    # 1) Caso: il pipeline stesso ha feature_names_in_
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    # 2) Caso: Pipeline con named_steps -> cerca step con feature_names_in_
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            # se è un ColumnTransformer, può non avere feature_names_in_
            # ma i suoi trasformatori spesso sì
            if hasattr(step, "transformers_"):  # ColumnTransformer
                # Se il ColumnTransformer è stato fit su DF,
                # di solito lui stesso ha feature_names_in_
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)

    # 3) Fallback robusto: usa i label della tua UI (adattali se servono)
    return ["Age", "Sex", "Blood Pressure", "Cholesterol", "Na_to_K"]

EXPECTED_COLS = get_expected_columns(pipe)
st.caption(f"Colonne attese dal modello: {EXPECTED_COLS}")

# Mappa "normalizzata" per collegare i nomi UI ai nomi attesi (case-insensitive, underscores e spazi ignorati)
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
    ui_values = {
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
            # Se non troviamo match, prova a gestire alias noti
            if key_norm in ("natokratio", "natok", "sodiumpotassium", "sodiumpotassiumratio"):
                row[col] = ui_values["na_to_k_ratio"]
            elif key_norm in ("bp", "bloodpres", "bloodpressurelevel"):
                row[col] = ui_values["blood_pressure"]
            else:
                # estrema difesa: se non so cosa mettere, prova None (il preprocessor potrebbe gestirlo)
                row[col] = None

    df = pd.DataFrame([row], columns=expected_cols)

    # Tipi utili (non obbligatorio, ma aiuta):
    # Int & float
    for c in df.columns:
        cn = _norm(c)
        if cn == "age":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        if cn in ("natok", "natokratio", "sodiumpotassium", "sodiumpotassiumratio"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --------- PREDICT WRAPPER ----------
def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    X = build_input_dataframe(age, sex, blood_pressure, cholesterol, na_to_k_ratio, EXPECTED_COLS)
    pred = pipe.predict(X)[0]

    proba_df = None
    if hasattr(pipe, "predict_proba"):
        try:
            probs = pipe.predict_proba(X)[0]
            # classi: prova stimatore finale, poi pipeline
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


# --------- UI ---------
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