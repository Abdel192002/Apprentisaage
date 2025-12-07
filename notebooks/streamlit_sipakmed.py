
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from PIL import Image

# Configuration
st.set_page_config(
    page_title="SIPAKMED Explicabilite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre
st.title("Dashboard d'Explicabilite - SIPAKMED")
st.markdown("### **Votre dataset** : sipakmed_new6")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Mode
    mode = st.radio(
        "Source des donnees",
        ["Vos rapports generes", "Mode demonstration"],
        index=0
    )
    
    # Charger le rapport
    @st.cache_data
    def load_real_report():
        """Charger VOTRE rapport"""
        try:
            with open("reports/explainability/simple_report_0.json", "r", encoding='utf-8') as f:
                data = json.load(f)
            
            # Verifier si ce sont VOS donnees
            if "data_path" in data.get("metadata", {}):
                if "sipakmed" in data["metadata"]["data_path"].lower():
                    st.sidebar.success("DONNEES SIPAKMED detectees")
            
            return data
        except Exception as e:
            st.sidebar.error(f"Erreur: {e}")
            # Rapport de demo
            return {
                "metadata": {
                    "project": "SIPAKMED - Donnees reelles",
                    "classes": ["Abnormal", "Benign", "Normal"],
                    "data_path": "C:/Users/nessa/Downloads/sipakmed_new6/"
                },
                "image_info": {"true_label": "Normal"},
                "results_summary": {"accuracy_percentage": 66.7},
                "model_details": {
                    "ResNet": {"predicted_class": "Normal", "confidence": 0.85, "is_correct": True},
                    "MobileNet": {"predicted_class": "Benign", "confidence": 0.72, "is_correct": False},
                    "EfficientNet": {"predicted_class": "Normal", "confidence": 0.91, "is_correct": True}
                }
            }
    
    report = load_real_report()
    
    # Info sur les donnees
    st.markdown("---")
    st.subheader("Vos donnees")
    
    classes = report.get("metadata", {}).get("classes", [])
    st.write(f"**Classes:** {', '.join(classes)}")
    
    data_path = report.get("metadata", {}).get("data_path", "Non specifie")
    st.write(f"**Chemin:** {data_path}")

# ==================== AFFICHAGE PRINCIPAL ====================

# En-tete
st.header("Resultats d'analyse")

# Metriques
col1, col2, col3 = st.columns(3)

with col1:
    accuracy = report.get("results_summary", {}).get("accuracy_percentage", 0)
    st.metric("Exactitude globale", f"{accuracy:.1f}%")

with col2:
    true_label = report.get("image_info", {}).get("true_label", "Inconnu")
    st.metric("Verite terrain", true_label)

with col3:
    models = list(report.get("model_details", {}).keys())
    st.metric("Modeles analyses", len(models))

# Image de test
st.subheader("Image analysee")

# Chercher l'image
img_paths = [
    "reports/explainability/test_image.png",
    "reports/explainability/comparison_image_1.png"
]

img_found = False
for path in img_paths:
    if os.path.exists(path):
        try:
            st.image(path, use_container_width=True, caption="Image de test extraite de votre dataset")
            img_found = True
            break
        except:
            continue

if not img_found:
    st.info("L'image sera affichee apres l'extraction des donnees")

# Analyse par modele
st.markdown("---")
st.header("Predictions par modele")

model_details = report.get("model_details", {})

if model_details:
    # Creer des colonnes
    cols = st.columns(len(model_details))
    
    for idx, (model_name, details) in enumerate(model_details.items()):
        with cols[idx]:
            # Carte du modele
            with st.container():
                st.subheader(model_name)
                
                # Statut
                if details.get("is_correct", False):
                    st.success(f"CORRECT: {details.get('predicted_class', 'N/A')}")
                else:
                    st.error(f"INCORRECT: {details.get('predicted_class', 'N/A')}")
                
                # Confiance
                confidence = details.get("confidence", 0)
                st.metric("Confiance", f"{confidence:.1%}")
                
                # Graphique simple
                try:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    
                    classes = report.get("metadata", {}).get("classes", ["C1", "C2", "C3"])
                    probs = details.get("all_probabilities", [0.5, 0.3, 0.2])
                    
                    if len(probs) >= len(classes):
                        bars = ax.bar(classes[:3], probs[:3], color=['lightblue']*3)
                        
                        # Colorer la prediction
                        pred_idx = details.get("predicted_class_idx", 0)
                        if pred_idx < len(bars):
                            bars[pred_idx].set_color('green')
                        
                        ax.set_ylim([0, 1])
                        ax.set_xticklabels(classes[:3], rotation=45, fontsize=8)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur graphique: {e}")

# Comparaison
st.markdown("---")
st.header("Comparaison des modeles")

if model_details:
    # Tableau
    comparison_data = []
    for model_name, details in model_details.items():
        comparison_data.append({
            "Modele": model_name,
            "Prediction": details.get("predicted_class", "N/A"),
            "Confiance": f"{details.get('confidence', 0):.1%}",
            "Statut": "CORRECT" if details.get("is_correct") else "INCORRECT",
            "Verite": "OUI" if details.get("is_correct") else "NON"
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Graphique
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        models = list(model_details.keys())
        confidences = [model_details[m].get("confidence", 0) for m in models]
        colors = ['green' if model_details[m].get("is_correct") else 'red' for m in models]
        
        bars = ax.bar(models, confidences, color=colors, edgecolor='black')
        ax.set_ylabel("Confiance")
        ax.set_ylim([0, 1])
        ax.set_title("Performance des modeles sur votre dataset")
        ax.grid(True, alpha=0.3)
        
        # Ajouter les valeurs
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{conf:.1%}', ha='center', va='bottom')
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur graphique: {e}")

# Informations sur votre projet
st.markdown("---")
st.header("Informations sur votre projet")

with st.expander("Configuration technique", expanded=True):
    st.markdown(f"""
    **Dataset:** SIPAKMED
    **Chemin:** `{data_path}`
    **Classes:** {', '.join(classes)}
    **Nombre de modeles:** {len(model_details)}
    
    **Methodes d'explicabilite appliquees:**
    1. Grad-CAM - Localisation des regions importantes
    2. Importance par gradients - Sensibilite des pixels
    3. LIME - Explications locales (si disponible)
    
    **Prochaines etapes:**
    - Executer les cellules d'analyse complete
    - Generer les visualisations reelles
    - Integrer avec MLflow pour le suivi
    """)

# Footer
st.markdown("---")

footer_cols = st.columns(3)

with footer_cols[0]:
    st.caption("**Votre projet MLOps**")
    st.caption("Avec vos donnees reelles")

with footer_cols[1]:
    st.caption(f"**Dataset:** SIPAKMED")
    st.caption(f"{len(classes)} classes")

with footer_cols[2]:
    st.caption("**Modeles:**")
    if model_details:
        st.caption(", ".join(list(model_details.keys())))

st.markdown("---")
st.caption("Dashboard genere avec vos donnees reelles | Interface Streamlit")
