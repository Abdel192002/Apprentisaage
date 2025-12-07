
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Explicabilité SIPAKMED",
    page_icon="🔍",
    layout="wide"
)

# Titre
st.title("🔍 Dashboard d'Explicabilité - SIPAKMED")

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
# Charger vos résultats
RESULTS_PATH = Path("reports/explainability/simple_report_0.json")

if RESULTS_PATH.exists():
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)
    
    st.success("✅ Données chargées avec succès")
    
    # Extraire les données
    classes = data["metadata"]["classes"]
    true_label = data["image_info"]["true_label"]
    model_details = data["model_details"]
    
    # ========================================================================
    # AFFICHAGE DES RÉSULTATS
    # ========================================================================
    # Métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        correct = sum(1 for d in model_details.values() if d["is_correct"])
        total = len(model_details)
        accuracy = (correct / total * 100) if total > 0 else 0
        st.metric("Exactitude", f"{accuracy:.1f}%")
    
    with col2:
        st.metric("Vérité terrain", true_label)
    
    with col3:
        st.metric("Modèles", total)
    
    # Tableau
    st.subheader("📋 Performances des modèles")
    
    table_data = []
    for model_name, details in model_details.items():
        table_data.append({
            "Modèle": model_name,
            "Prédiction": details["predicted_class"],
            "Confiance": f"{details['confidence']:.1%}",
            "Statut": "✅" if details["is_correct"] else "❌"
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    
    # Graphique
    st.subheader("📈 Comparaison des confiances")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    models = list(model_details.keys())
    confidences = [model_details[m]["confidence"] for m in models]
    colors = ['green' if model_details[m]["is_correct"] else 'red' for m in models]
    
    bars = ax.bar(models, confidences, color=colors)
    ax.set_ylabel("Confiance")
    ax.set_ylim([0, 1])
    ax.set_title("Niveaux de confiance")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Analyse
    st.subheader("🔍 Analyse")
    
    correct_models = [m for m, d in model_details.items() if d["is_correct"]]
    incorrect_models = [m for m, d in model_details.items() if not d["is_correct"]]
    
    if correct_models:
        st.success(f"**✅ Modèles performants :** {', '.join(correct_models)}")
    
    if incorrect_models:
        st.error(f"**❌ Modèles à améliorer :** {', '.join(incorrect_models)}")
    
else:
    st.error("❌ Aucun résultat trouvé")
    st.info("Exécutez d'abord le notebook d'analyse")

# Footer
st.markdown("---")
st.caption("Dashboard Streamlit | Projet MLOps | Explicabilité SIPAKMED")
