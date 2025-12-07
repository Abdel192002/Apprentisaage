
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Configuration
st.set_page_config(
    page_title="Dashboard Explicabilité",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre
st.title("🔍 Dashboard d'Explicabilité")
st.markdown("### Visualisation interactive des résultats")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Chercher les rapports
def find_reports():
    reports = []
    for i in range(5):
        path = f"reports/explainability/simple_report_{i}.json"
        if os.path.exists(path):
            reports.append((i, path))
    return reports

# Charger un rapport
@st.cache_data
def load_report(idx):
    path = f"reports/explainability/simple_report_{idx}.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# Interface principale
def main():
    # Chercher les rapports disponibles
    reports = find_reports()
    
    if not reports:
        st.error("❌ Aucun rapport trouvé")
        st.info("Exécutez d'abord les cellules d'analyse ci-dessus")
        return
    
    # Sélection du rapport
    report_options = [f"Rapport {idx+1}" for idx, _ in reports]
    selected = st.sidebar.selectbox("Choisir un rapport", report_options)
    report_idx = report_options.index(selected)
    
    # Charger le rapport
    report = load_report(report_idx)
    
    if report is None:
        st.error("Erreur de chargement")
        return
    
    st.sidebar.success(f"✅ Rapport {report_idx+1} chargé")
    
    # ==================== MÉTRIQUES ====================
    st.header("📊 Vue d'ensemble")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = report.get("results_summary", {}).get("accuracy_percentage", 0)
        st.metric("Exactitude", f"{accuracy:.1f}%")
    
    with col2:
        true_label = report.get("image_info", {}).get("true_label", "Inconnu")
        st.metric("Vérité terrain", true_label)
    
    with col3:
        total = report.get("results_summary", {}).get("total_models", 0)
        correct = report.get("results_summary", {}).get("correct_predictions", 0)
        st.metric("Correctes", f"{correct}/{total}")
    
    # ==================== MODÈLES ====================
    st.markdown("---")
    st.header("🤖 Analyse par modèle")
    
    model_details = report.get("model_details", {})
    
    if model_details:
        # Créer des onglets pour chaque modèle
        tabs = st.tabs(list(model_details.keys()))
        
        for tab, (model_name, details) in zip(tabs, model_details.items()):
            with tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Statut
                    if details.get("is_correct", False):
                        st.success("✅ **Correct**")
                    else:
                        st.error("❌ **Incorrect**")
                    
                    # Prédiction
                    st.metric(
                        "Classe prédite", 
                        details.get("predicted_class", "N/A")
                    )
                    
                    # Confiance
                    confidence = details.get("confidence", 0)
                    st.metric("Confiance", f"{confidence:.1%}")
                
                with col2:
                    # Graphique des probabilités
                    st.subheader("Probabilités")
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    classes = report.get("metadata", {}).get("classes", [])
                    probs = details.get("all_probabilities", [])
                    
                    if len(classes) == len(probs):
                        bars = ax.bar(classes, probs, color='skyblue')
                        
                        # Colorer la barre prédite
                        pred_idx = details.get("predicted_class_idx", 0)
                        if pred_idx < len(bars):
                            bars[pred_idx].set_color('green')
                        
                        ax.set_ylabel("Probabilité")
                        ax.set_ylim([0, 1])
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
    
    # ==================== VISUALISATIONS ====================
    st.markdown("---")
    st.header("🖼️ Visualisations")
    
    # Chercher l'image de comparaison
    img_idx = report.get("metadata", {}).get("image_index", 0)
    comparison_path = f"reports/explainability/comparison_image_{img_idx+1}.png"
    
    if os.path.exists(comparison_path):
        st.image(comparison_path, use_column_width=True)
    else:
        st.info("Visualisation non disponible")
    
    # ==================== COMPARAISON ====================
    st.markdown("---")
    st.header("📈 Comparaison")
    
    if model_details:
        # Tableau
        st.subheader("Tableau comparatif")
        
        data = []
        for name, details in model_details.items():
            data.append({
                "Modèle": name,
                "Prédiction": details.get("predicted_class", "N/A"),
                "Confiance": f"{details.get('confidence', 0):.1%}",
                "Statut": "✅" if details.get("is_correct") else "❌"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Graphique des confiances
        st.subheader("Confiances par modèle")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        models = list(model_details.keys())
        confidences = [model_details[m].get("confidence", 0) for m in models]
        
        colors = ['green' if model_details[m].get("is_correct") else 'red' for m in models]
        bars = ax.bar(models, confidences, color=colors, edgecolor='black')
        
        ax.set_ylabel("Confiance")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Ajouter les valeurs
        for bar, conf in zip(bars, confidences):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{conf:.1%}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # ==================== GUIDE ====================
    st.markdown("---")
    st.header("📚 Guide")
    
    with st.expander("Comment interpréter les résultats"):
        st.markdown("""
        - **Grad-CAM** : Montre les régions importantes (rouge = très important)
        - **Confiance** : Fiabilité de la prédiction
        - **Statut** : ✅ correct ou ❌ incorrect
        
        **Conseils** :
        1. Comparez les modèles entre eux
        2. Vérifiez la cohérence des visualisations
        3. Analysez les erreurs avec haute confiance
        """)
    
    # Footer
    st.markdown("---")
    st.caption("Dashboard généré automatiquement - Projet MLOps")

if __name__ == "__main__":
    main()
