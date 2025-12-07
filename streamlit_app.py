import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# Ajoute le dossier src au path pour importer tes modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configuration de la page
st.set_page_config(
    page_title="MLOps Classification App",
    layout="wide"
)

# Titre principal
st.title("🚀 MLOps Pipeline - Classification Interface")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Choisir une page:",
    ["🏠 Accueil", "🔍 Prédiction Unitaires", "📊 Prédiction par Lot", "📈 Performance du Modèle", "🤖 Explicabilité"]
)

# Charger le modèle (à adapter)
@st.cache_resource
def load_model():
    """
    Charge le modèle entraîné.
    Adapte cette fonction selon comment tu sauvegardes ton modèle.
    """
    try:
        # Exemple 1: Modèle pickle
        # model = pickle.load(open("models/best_model.pkl", "rb"))
        
        # Exemple 2: Modèle MLflow
        # import mlflow.pyfunc
        # model = mlflow.pyfunc.load_model("models:/best_model/Production")
        
        # Pour l'instant, on retourne None
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Charger le modèle une fois
model = load_model()

# Charger les données d'exemple pour les features
@st.cache_data
def load_feature_info():
    """
    Charge les informations sur les features (noms, types, ranges).
    Tu peux créer un fichier JSON ou CSV avec ces infos.
    """
    # Exemple pour un dataset avec 20+ features
    features = {
        "feature1": {"type": "float", "min": 0, "max": 100, "default": 50},
        "feature2": {"type": "int", "min": 1, "max": 10, "default": 5},
        # ... ajoute toutes tes features
    }
    return features

# Page d'accueil
if page == "🏠 Accueil":
    st.header("Bienvenue dans l'interface MLOps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 À propos du projet")
        st.markdown("""
        Cette application fait partie du pipeline MLOps développé pour le projet de classification.
        
        **Fonctionnalités :**
        - 🔍 Prédictions unitaires
        - 📊 Prédictions par lot (fichier CSV)
        - 📈 Visualisation des performances
        - 🤖 Explicabilité du modèle (SHAP)
        """)
    
    with col2:
        st.markdown("### 🛠️ Technologies utilisées")
        st.markdown("""
        - **Streamlit** : Interface web
        - **MLflow** : Tracking d'expériences
        - **DVC** : Versionnement des données
        - **SHAP** : Explicabilité
        - **Scikit-learn** / **TensorFlow** : Modèles
        """)
    
    st.markdown("---")
    st.markdown("### 👥 Équipe")
    st.markdown("""
    - Membre 1
    - Membre 2
    - Membre 3
    """)

# Page de prédiction unitaire
elif page == "🔍 Prédiction Unitaires":
    st.header("Prédiction Unitaires")
    
    if model is None:
        st.warning("⚠️ Modèle non chargé. Assure-toi d'avoir un modèle entraîné.")
    
    # Section pour saisir les features
    st.subheader("📝 Saisie des caractéristiques")
    
    # Charger les infos sur les features
    features_info = load_feature_info()
    
    # Créer des colonnes pour organiser les inputs
    num_columns = 3  # Nombre de colonnes pour l'affichage
    features_list = list(features_info.keys())
    
    # Dictionnaire pour stocker les valeurs
    input_values = {}
    
    # Créer les inputs dynamiquement
    cols = st.columns(num_columns)
    for i, feature_name in enumerate(features_list):
        info = features_info[feature_name]
        with cols[i % num_columns]:
            if info["type"] == "float":
                value = st.number_input(
                    f"{feature_name}",
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(info["default"]),
                    step=0.1
                )
            elif info["type"] == "int":
                value = st.number_input(
                    f"{feature_name}",
                    min_value=int(info["min"]),
                    max_value=int(info["max"]),
                    value=int(info["default"]),
                    step=1
                )
            else:
                value = st.text_input(f"{feature_name}", value="")
            
            input_values[feature_name] = value
    
    # Bouton de prédiction
    if st.button("🎯 Lancer la prédiction", type="primary"):
        if model is not None:
            try:
                # Préparer les données pour le modèle
                input_df = pd.DataFrame([input_values])
                
                # Faire la prédiction
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)
                
                # Afficher les résultats
                st.success("✅ Prédiction terminée !")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🎯 Prédiction", f"Classe {prediction[0]}")
                
                with col2:
                    proba_max = np.max(probability[0]) * 100
                    st.metric("📊 Confiance", f"{proba_max:.2f}%")
                
                # Afficher les probabilités par classe
                st.subheader("📈 Probabilités par classe")
                proba_df = pd.DataFrame({
                    'Classe': range(len(probability[0])),
                    'Probabilité': probability[0]
                })
                st.bar_chart(proba_df.set_index('Classe'))
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
        else:
            st.error("❌ Modèle non disponible pour la prédiction")

# Page de prédiction par lot
elif page == "📊 Prédiction par Lot":
    st.header("Prédiction par Lot")
    
    st.markdown("""
    Téléchargez un fichier CSV contenant les données à prédire.
    Le fichier doit contenir les mêmes colonnes que le dataset d'entraînement.
    """)
    
    # Upload du fichier
    uploaded_file = st.file_uploader(
        "📁 Choisir un fichier CSV",
        type=['csv'],
        help="Le fichier doit contenir un en-tête avec les noms des features"
    )
    
    if uploaded_file is not None:
        try:
            # Lire le fichier CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Fichier chargé : {uploaded_file.name}")
            st.write(f"📊 **{len(df)}** lignes et **{len(df.columns)}** colonnes détectées")
            
            # Aperçu des données
            with st.expander("👁️ Aperçu des données"):
                st.dataframe(df.head())
            
            # Vérifier que le modèle est disponible
            if model is not None:
                if st.button("🎯 Lancer les prédictions par lot", type="primary"):
                    with st.spinner("Prédiction en cours..."):
                        # Faire les prédictions
                        predictions = model.predict(df)
                        probabilities = model.predict_proba(df)
                        
                        # Ajouter les prédictions au DataFrame
                        df_result = df.copy()
                        df_result['prediction'] = predictions
                        df_result['confidence'] = np.max(probabilities, axis=1)
                        
                        # Afficher un échantillon des résultats
                        st.subheader("📋 Résultats des prédictions")
                        st.dataframe(df_result.head())
                        
                        # Statistiques
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📈 Nombre total", len(df_result))
                        with col2:
                            st.metric("🎯 Prédiction la plus fréquente", df_result['prediction'].mode()[0])
                        with col3:
                            avg_conf = df_result['confidence'].mean() * 100
                            st.metric("📊 Confiance moyenne", f"{avg_conf:.2f}%")
                        
                        # Téléchargement des résultats
                        st.subheader("💾 Télécharger les résultats")
                        csv = df_result.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            type="primary"
                        )
            else:
                st.warning("⚠️ Modèle non chargé. Impossible de faire des prédictions.")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier: {e}")

# Page de performance du modèle
elif page == "📈 Performance du Modèle":
    st.header("Performance du Modèle")
    
    st.markdown("""
    Cette page affiche les métriques de performance du meilleur modèle sélectionné.
    """)
    
    # Section pour afficher les métriques
    st.subheader("📊 Métriques principales")
    
    # Créer des métriques fictives (à remplacer par tes vraies métriques)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "0.92", "+0.02")
    with col2:
        st.metric("Precision", "0.89", "-0.01")
    with col3:
        st.metric("Recall", "0.94", "+0.03")
    with col4:
        st.metric("F1-Score", "0.91", "+0.01")
    
    # Section pour les graphiques
    st.subheader("📈 Visualisations")
    
    # Exemple de graphiques (à remplacer par tes vrais graphiques)
    tab1, tab2, tab3 = st.tabs(["Matrice de confusion", "Courbe ROC", "Learning Curve"])
    
    with tab1:
        st.markdown("**Matrice de confusion**")
        # Code pour afficher une image ou générer un graphique
        # st.image("path/to/confusion_matrix.png")
        st.info("La matrice de confusion sera chargée depuis MLflow")
    
    with tab2:
        st.markdown("**Courbe ROC**")
        # Générer un exemple de courbe ROC
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.plot([0, 0.2, 0.8, 1], [0, 0.6, 0.9, 1], label='Modèle')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Courbe ROC')
        ax.legend()
        st.pyplot(fig)
    
    with tab3:
        st.markdown("**Learning Curve**")
        # Placeholder pour la courbe d'apprentissage
        st.info("Courbe d'apprentissage à générer depuis MLflow")
    
    # Lien vers MLflow UI
    st.markdown("---")
    st.markdown("### 🔗 Accès à MLflow")
    st.markdown("""
    Pour plus de détails sur les expériences, accédez à l'interface MLflow :
    """)
    
    if st.button("🌐 Ouvrir MLflow UI"):
        st.info("Lancez `mlflow ui` dans votre terminal et accédez à http://localhost:5000")

# Page d'explicabilité
elif page == "🤖 Explicabilité":
    st.header("Explicabilité du Modèle avec SHAP")
    
    st.markdown("""
    Cette page utilise SHAP pour expliquer les prédictions du modèle.
    """)
    
    if model is None:
        st.warning("⚠️ Modèle non chargé. Impossible de générer des explications.")
    else:
        # Option pour l'explication
        explanation_type = st.radio(
            "Type d'explication:",
            ["📊 Explication globale", "🔍 Explication locale"]
        )
        
        if explanation_type == "📊 Explication globale":
            st.subheader("Importance globale des features")
            
            # Générer des valeurs SHAP fictives (à remplacer)
            try:
                # Exemple avec des données fictives
                np.random.seed(42)
                feature_names = list(load_feature_info().keys())[:10]  # 10 premières features
                shap_values = np.random.randn(100, len(feature_names))
                
                # Créer le summary plot
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, feature_names=feature_names, show=False)
                st.pyplot(fig)
                
                # Importance moyenne
                st.subheader("📈 Importance moyenne (SHAP values)")
                mean_shap = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': mean_shap
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df)
                
            except Exception as e:
                st.error(f"Erreur lors de la génération des explications: {e}")
        
        else:  # Explication locale
            st.subheader("Explication pour une prédiction spécifique")
            
            # Sélectionner un échantillon
            sample_idx = st.slider(
                "Sélectionner un échantillon:",
                min_value=0,
                max_value=99,
                value=0
            )
            
            # Générer un waterfall plot fictif
            try:
                np.random.seed(42)
                feature_names = list(load_feature_info().keys())[:7]  # 7 features pour l'exemple
                shap_values_single = np.random.randn(len(feature_names))
                
                # Créer le waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap.Explanation(values=shap_values_single, 
                                                      feature_names=feature_names), 
                                    max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Explication textuelle
                st.markdown("### 📝 Interprétation")
                st.markdown("""
                - **Features en bleu** : Contribuent positivement à la prédiction
                - **Features en rouge** : Contribuent négativement à la prédiction
                - La valeur de base (base value) est la prédiction moyenne
                """)
                
            except Exception as e:
                st.error(f"Erreur lors de la génération de l'explication locale: {e}")

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "Projet MLOps - M2 SID 2025-2026<br>"
    "</div>",
    unsafe_allow_html=True
)