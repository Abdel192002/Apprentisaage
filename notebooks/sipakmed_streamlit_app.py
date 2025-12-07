
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import tempfile
import io
import base64
from pathlib import Path
import zipfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import subprocess

# Configuration de la page
st.set_page_config(
    page_title="SipakMed Batch Prediction",
    page_icon="🔬",
    layout="wide"
)

# Titre
st.title("🔬 SipakMed Batch Prediction System")
st.markdown("""
Cette application permet de réaliser des prédictions par lot sur des images médicales SipakMed.
Téléchargez un fichier CSV contenant les chemins d'images ou un ZIP avec des images.
""")

# ============================================================================
# CHARGEMENT DU MODÈLE (CACHÉ)
# ============================================================================

@st.cache_resource
def load_predictor():
    """Charger le prédicteur"""
    import sys
    import os
    
    class SipakMedBatchPredictor:
        def __init__(self):
            self.model = None
            self.class_names = ['Abnormal', 'Benign', 'Normal']
            
        def load_model(self):
            try:
                # Chemin dans l'environnement temporaire Streamlit
                possible_paths = [
                    "resnet_model.h5",
                    "/tmp/resnet_model.h5",
                    "./models/resnet_model.h5"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.model = load_model(path)
                        break
                        
                if self.model is None:
                    from tensorflow.keras.applications import ResNet50
                    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
                    from tensorflow.keras.models import Model
                    
                    base = ResNet50(weights='imagenet', include_top=False, 
                                   input_shape=(224, 224, 3))
                    x = base.output
                    x = GlobalAveragePooling2D()(x)
                    outputs = Dense(len(self.class_names), activation='softmax')(x)
                    self.model = Model(inputs=base.input, outputs=outputs)
                    
            except Exception as e:
                st.error(f"Erreur chargement modèle: {e}")
                return None
                
            return self
    
    predictor = SipakMedBatchPredictor()
    predictor.load_model()
    return predictor

# Initialiser
predictor = load_predictor()

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def preprocess_image(img, target_size=(224, 224)):
    """Prétraiter une image"""
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    """Prédire une image"""
    predictions = predictor.model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = predictor.class_names[predicted_idx]
    confidence = float(predictions[0][predicted_idx])
    
    confidences = {
        class_name: float(predictions[0][i])
        for i, class_name in enumerate(predictor.class_names)
    }
    
    return predicted_class, confidence, confidences

def process_uploaded_file(uploaded_file, temp_dir):
    """Traiter un fichier uploadé"""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def extract_zip(zip_path, extract_dir):
    """Extraire un fichier ZIP"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Lister toutes les images extraites
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Paramètres")
    
    batch_size = st.slider("Taille du batch", 1, 50, 10)
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.7)
    
    st.header("📊 Classes")
    for i, class_name in enumerate(predictor.class_names):
        st.write(f"{i+1}. {class_name}")
    
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload CSV ou ZIP
    2. Configurer les paramètres
    3. Lancer les prédictions
    4. Télécharger les résultats
    """)

# ============================================================================
# ONGLETS PRINCIPAUX
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "🔍 Preview", "📊 Results", "📈 Analytics"])

with tab1:
    st.header("1. Upload des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        upload_option = st.radio(
            "Type de fichier",
            ["CSV avec chemins", "ZIP avec images", "Images individuelles"]
        )
    
    with col2:
        if upload_option == "CSV avec chemins":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=['csv'],
                help="CSV avec colonne 'image_path'"
            )
            
        elif upload_option == "ZIP avec images":
            uploaded_file = st.file_uploader(
                "Upload ZIP",
                type=['zip'],
                help="ZIP contenant des images"
            )
            
        else:  # Images individuelles
            uploaded_files = st.file_uploader(
                "Upload images",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Sélectionnez une ou plusieurs images"
            )
    
    # Traitement des fichiers uploadés
    if uploaded_file and upload_option != "Images individuelles":
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = process_uploaded_file(uploaded_file, temp_dir)
            
            if upload_option == "CSV avec chemins":
                df = pd.read_csv(file_path)
                st.session_state['image_paths'] = df['image_path'].tolist() if 'image_path' in df.columns else []
                st.session_state['file_type'] = 'csv'
                
            elif upload_option == "ZIP avec images":
                image_paths = extract_zip(file_path, temp_dir)
                st.session_state['image_paths'] = image_paths
                st.session_state['file_type'] = 'zip'
    
    elif upload_option == "Images individuelles" and uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            for uploaded_file in uploaded_files:
                file_path = process_uploaded_file(uploaded_file, temp_dir)
                image_paths.append(file_path)
            
            st.session_state['image_paths'] = image_paths
            st.session_state['file_type'] = 'images'
    
    # Bouton de prédiction
    if 'image_paths' in st.session_state and st.session_state['image_paths']:
        if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
            with st.spinner("Traitement en cours..."):
                # Stocker les résultats dans session_state
                st.session_state['results'] = []
                st.session_state['errors'] = []
                
                total = len(st.session_state['image_paths'])
                progress_bar = st.progress(0)
                
                for i, image_path in enumerate(st.session_state['image_paths']):
                    try:
                        # Charger et prédire
                        img = Image.open(image_path)
                        img_array = preprocess_image(img)
                        pred_class, confidence, confidences = predict_image(img_array)
                        
                        st.session_state['results'].append({
                            'image_path': image_path,
                            'filename': os.path.basename(image_path),
                            'predicted_class': pred_class,
                            'confidence': confidence,
                            'confidences': confidences
                        })
                        
                    except Exception as e:
                        st.session_state['errors'].append({
                            'image_path': image_path,
                            'error': str(e)
                        })
                    
                    # Mettre à jour la barre
                    progress_bar.progress((i + 1) / total)
                
                st.success(f"✅ {len(st.session_state['results'])} prédictions terminées!")
                if st.session_state['errors']:
                    st.warning(f"⚠️ {len(st.session_state['errors'])} erreurs")

with tab2:
    st.header("2. Aperçu des données")
    
    if 'image_paths' in st.session_state:
        st.info(f"📁 {len(st.session_state['image_paths'])} images chargées")
        
        # Afficher quelques images
        cols = st.columns(4)
        for idx, image_path in enumerate(st.session_state['image_paths'][:8]):
            try:
                img = Image.open(image_path)
                with cols[idx % 4]:
                    st.image(img, caption=os.path.basename(image_path), use_column_width=True)
            except:
                with cols[idx % 4]:
                    st.error(f"Erreur: {os.path.basename(image_path)}")
    else:
        st.info("ℹ️ Aucune donnée chargée. Upload des fichiers dans l'onglet 1.")

with tab3:
    st.header("3. Résultats des prédictions")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Convertir en DataFrame
        df_results = pd.DataFrame(results)
        
        # Afficher le tableau
        st.dataframe(df_results[['filename', 'predicted_class', 'confidence']], use_container_width=True)
        
        # Options de téléchargement
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV complet
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name="sipakmed_predictions.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            # JSON
            json_str = df_results.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="sipakmed_predictions.json",
                mime="application/json"
            )
        
        with col3:
            # Rapport Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Predictions')
                
                # Ajouter un résumé
                summary = pd.DataFrame({
                    'Statistique': ['Total', 'Abnormal', 'Benign', 'Normal', 'Confiance moyenne'],
                    'Valeur': [
                        len(df_results),
                        len(df_results[df_results['predicted_class'] == 'Abnormal']),
                        len(df_results[df_results['predicted_class'] == 'Benign']),
                        len(df_results[df_results['predicted_class'] == 'Normal']),
                        df_results['confidence'].mean()
                    ]
                })
                summary.to_excel(writer, index=False, sheet_name='Summary')
            
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_buffer.getvalue(),
                file_name="sipakmed_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Afficher les erreurs
        if 'errors' in st.session_state and st.session_state['errors']:
            with st.expander("⚠️ Erreurs", expanded=False):
                for error in st.session_state['errors']:
                    st.error(f"{error['image_path']}: {error['error']}")
    
    else:
        st.info("ℹ️ Aucun résultat disponible. Lancez les prédictions dans l'onglet 1.")

with tab4:
    st.header("4. Analytics")
    
    if 'results' in st.session_state:
        df_results = pd.DataFrame(st.session_state['results'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des classes
            st.subheader("Distribution des classes")
            class_dist = df_results['predicted_class'].value_counts()
            
            fig, ax = plt.subplots()
            ax.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%')
            ax.set_title("Distribution des prédictions")
            st.pyplot(fig)
        
        with col2:
            # Histogramme des confiances
            st.subheader("Distribution des confiances")
            
            fig, ax = plt.subplots()
            ax.hist(df_results['confidence'], bins=20, alpha=0.7, color='skyblue')
            ax.set_xlabel("Confiance")
            ax.set_ylabel("Nombre d'images")
            ax.set_title("Histogramme des confiances")
            st.pyplot(fig)
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total images", len(df_results))
        
        with col2:
            st.metric("Confiance moyenne", f"{df_results['confidence'].mean():.2%}")
        
        with col3:
            dominant_class = df_results['predicted_class'].mode()[0]
            st.metric("Classe dominante", dominant_class)
        
        # Top 5 des plus confiantes
        st.subheader("Top 5 des prédictions les plus confiantes")
        top5 = df_results.nlargest(5, 'confidence')[['filename', 'predicted_class', 'confidence']]
        st.dataframe(top5, use_container_width=True)
    
    else:
        st.info("ℹ️ Aucune donnée analytique disponible.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "🔬 SipakMed Batch Prediction v1.0 | "
    "Développé avec Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
