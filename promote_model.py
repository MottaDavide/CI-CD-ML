#!/usr/bin/env python
# coding: utf-8
import os
from mlflow.tracking import MlflowClient
import dagshub
import mlflow

OWNER = os.getenv("DAGSHUB_USERNAME")
REPO  = os.getenv("DAGSHUB_REPO")
TOKEN = os.getenv("DAGSHUB_TOKEN")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "DrugPipeline")

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
dagshub.init(repo_owner=OWNER, repo_name=REPO, mlflow=True)

client = MlflowClient()


def get_metrics(run_id):
    try:
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception:
        return {}


def main():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("Nessuna versione trovata nel registry.")
        return

    # Ricava f1 per ogni versione
    scored = []
    for v in versions:
        m = get_metrics(v.run_id)
        f1 = m.get("f1", 0.0)
        scored.append((int(v.version), f1))
    scored.sort(key=lambda x: x[0])  # ordina per versione
    print(f"Tutte le versioni: {scored}")

    # Trova la migliore per f1
    best_version, best_f1 = max(scored, key=lambda x: x[1])
    latest_version, latest_f1 = scored[-1]

    print(f"🏆 Migliore: v{best_version} (f1={best_f1:.4f})")
    print(f"🆕 Ultima:   v{latest_version} (f1={latest_f1:.4f})")

    # Versione attuale production (se esiste)
    try:
        current_prod = client.get_model_version_by_alias(MODEL_NAME, "production")
        current_prod_v = int(current_prod.version)
        current_prod_f1 = get_metrics(current_prod.run_id).get("f1", 0.0)
        print(f"🔸 Production attuale: v{current_prod_v} (f1={current_prod_f1:.4f})")
    except Exception:
        current_prod_v = None
        current_prod_f1 = -1
        print("Nessun alias @production ancora assegnato.")

    # Se la migliore è diversa dalla Production, aggiorna alias
    if best_version != current_prod_v:
        client.set_registered_model_alias(MODEL_NAME, "production", best_version)
        print(f"🚀 Promosso v{best_version} come @production (f1={best_f1:.4f})")
    else:
        print("✅ Nessuna promozione necessaria (Production è già la migliore).")

    # Se l’ultima versione non è la migliore → staging
    if latest_version != best_version:
        client.set_registered_model_alias(MODEL_NAME, "staging", latest_version)
        print(f"🧪 Assegnato v{latest_version} come @staging (f1={latest_f1:.4f})")

if __name__ == "__main__":
    main()