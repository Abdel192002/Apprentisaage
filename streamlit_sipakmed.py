import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys
import subprocess
from pathlib import Path
from PIL import Image
import yaml
import glob
import tensorflow as tf

# Configuration
st.set_page_config(
    page_title="SIPAKMED Explicabilité",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre
st.title("📊 Dashboard d'Explicabilité - SIPAKMED")
st.markdown("### **Dataset** : sipakmed_new6 - Données réelles avec DVC")

# ============================================================================
# FONCTIONS UTILITAIRES DVC
# ============================================================================

def check_dvc_status():
    """Vérifier l'état de DVC"""
    try:
        result = subprocess.run(['dvc', '--version'], 
                               capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def load_dvc_data_info():
    """Obtenir des informations sur les données DVC"""
    processed_dir = Path('data/processed')
    info = {
        "exists": processed_dir.exists(),
        "path": str(processed_dir),
        "files": [],
        "classes": []
    }
    
    if processed_dir.exists():
        # Compter les fichiers par type
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.npy')):
                    info["files"].append(os.path.join(root, file))
        
        # Trouver les classes (sous-répertoires)
        if os.path.exists(processed_dir):
            subdirs = [d for d in os.listdir(processed_dir) 
                      if os.path.isdir(processed_dir / d) and not d.startswith('.')]
            info["classes"] = sorted(subdirs)
    
    return info

# ============================================================================
# FONCTIONS POUR LES PRÉDICTIONS EN TEMPS RÉEL
# ============================================================================

@st.cache_resource
def load_models_from_dvc():
    """Charger les modèles depuis DVC"""
    models = {}
    models_dir = Path("src/models")
    
    # Configuration des modèles
    model_files = {
        "ResNet": ["resnet02.h5", "resnet02.keras", "resnet02_mlflow.h5"],
        "EfficientNet": ["efficientnet02.h5", "efficientnet02.keras", "eff02.h5"],
        "MobileNet": ["mobilenet01.h5", "mobilenet01.keras", "mobnet01.h5"]
    }
    
    for model_name, filenames in model_files.items():
        for filename in filenames:
            filepath = models_dir / filename
            if filepath.exists():
                try:
                    model = tf.keras.models.load_model(str(filepath))
                    models[model_name] = model
                    break
                except Exception as e:
                    continue
    
    return models

def preprocess_image_for_prediction(image, target_size=(224, 224)):
    """Prétraiter une image pour la prédiction"""
    if isinstance(image, Image.Image):
        # Convertir PIL en numpy
        image = np.array(image)
    
    # Redimensionner si nécessaire
    if image.shape[:2] != target_size:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pil_img = pil_img.resize(target_size)
        image = np.array(pil_img)
    
    # Normaliser
    if image.max() > 1.0:
        image = image / 255.0
    
    # S'assurer d'avoir 3 canaux
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    return image

def make_batch_predictions(images, models_dict, class_names):
    """Faire des prédictions batch sur plusieurs images"""
    batch_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, image in enumerate(images):
        status_text.text(f"📊 Traitement image {idx+1}/{len(images)}...")
        
        image_results = {}
        img_processed = preprocess_image_for_prediction(image)
        
        for model_name, model in models_dict.items():
            # Préparer l'entrée
            input_batch = np.expand_dims(img_processed, axis=0)
            
            # Prédiction
            predictions = model.predict(input_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            image_results[model_name] = {
                "predicted_class": class_names[predicted_idx],
                "predicted_class_index": int(predicted_idx),
                "confidence": confidence,
                "confidence_percentage": f"{confidence:.1%}",
                "all_probabilities": predictions[0].tolist()
            }
        
        batch_results[f"image_{idx+1}"] = {
            "image_shape": img_processed.shape,
            "predictions": image_results
        }
        
        progress_bar.progress((idx + 1) / len(images))
    
    progress_bar.empty()
    status_text.text("✅ Prédictions terminées !")
    
    return batch_results

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Vérifier DVC
    dvc_available = check_dvc_status()
    
    # Mode principal
    mode = st.radio(
        "Mode principal",
        ["📊 Rapports réels générés", 
         "📁 Données DVC (data/processed)", 
         "🚀 Prédictions en temps réel",
         "🔄 Mode démonstration"],
        index=0
    )
    
    # Section DVC
    if dvc_available:
        st.success("✅ DVC disponible")
        
        if mode == "📁 Données DVC (data/processed)":
            data_info = load_dvc_data_info()
            if data_info["exists"]:
                st.info(f"📂 {len(data_info['files'])} fichiers trouvés")
                if data_info["classes"]:
                    st.write("**Classes:**", ", ".join(data_info["classes"][:5]))
    
    st.markdown("---")
    
    # Sélection du rapport (seulement en mode rapports)
    if mode == "📊 Rapports réels générés":
        st.subheader("📄 Fichiers disponibles")
        
        # Chercher tous les rapports JSON
        report_patterns = [
            "reports/predictions/*.json",
            "results/*.json"
        ]
        
        json_files = []
        for pattern in report_patterns:
            json_files.extend(glob.glob(pattern))
        
        if json_files:
            selected_report = st.selectbox(
                "Choisir un rapport",
                json_files,
                index=0
            )
        else:
            selected_report = None
            st.warning("Aucun rapport JSON trouvé")
    
    # Paramètres pour les prédictions en temps réel
    if mode == "🚀 Prédictions en temps réel":
        st.subheader("⚡ Paramètres prédictions")
        
        batch_size = st.slider(
            "Nombre max d'images",
            min_value=1,
            max_value=10,
            value=3
        )
        
        show_predictions_details = st.checkbox("Afficher détails", value=True)
    
    # Information sur les données
    st.markdown("---")
    st.subheader("ℹ️ Informations")
    
    if mode == "📁 Données DVC (data/processed)":
        st.write("**Source:** Données versionnées DVC")
        st.write("**Chemin:** `data/processed/`")
    elif mode == "📊 Rapports réels générés":
        st.write("**Source:** Fichiers JSON générés")
    elif mode == "🚀 Prédictions en temps réel":
        st.write("**Source:** Images uploadées")
    else:
        st.write("**Source:** Données de démonstration")

# ============================================================================
# FONCTIONS DE CHARGEMENT DES RAPPORTS
# ============================================================================

@st.cache_data
def load_real_report(filepath):
    """Charger un rapport JSON avec gestion d'erreurs robuste"""
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier si le fichier est vide
        if not content.strip():
            st.error(f"Fichier vide: {filepath}")
            return None
        
        # Essayer de parser le JSON
        data = json.loads(content)
        
        # Vérifier la structure
        if not isinstance(data, dict):
            st.error(f"Structure JSON invalide dans {filepath}")
            return None
            
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Erreur JSON dans {filepath}: ligne {e.lineno}, colonne {e.colno}")
        st.error(f"Message: {e.msg}")
        
        # Essayer de voir ce qui cause le problème
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= e.lineno:
                    problematic_line = lines[e.lineno - 1]
                    st.error(f"Ligne problématique: {problematic_line.strip()}")
        except:
            pass
            
        return None
        
    except Exception as e:
        st.error(f"❌ Erreur chargement {filepath}: {str(e)}")
        return None

@st.cache_data
def load_demo_report():
    """Charger un rapport de démonstration"""
    return {
        "metadata": {
            "project": "SIPAKMED - Démonstration",
            "classes": ["Abnormal", "Benign", "Normal"],
            "data_path": "data/processed",
            "dvc_managed": True,
            "total_files": 1250
        },
        "image_info": {
            "true_label": "Normal",
            "image_shape": [224, 224, 3],
            "image_index": 0
        },
        "results_summary": {
            "accuracy_percentage": 66.7,
            "correct_predictions": 2,
            "total_models": 3,
            "timestamp": "2024-01-15 10:30:00"
        },
        "model_details": {
            "ResNet": {
                "predicted_class": "Normal",
                "predicted_class_index": 2,
                "confidence": 0.85,
                "confidence_percentage": "85.0%",
                "is_correct": True,
                "all_probabilities": [0.05, 0.10, 0.85]
            },
            "MobileNet": {
                "predicted_class": "Benign",
                "predicted_class_index": 1,
                "confidence": 0.72,
                "confidence_percentage": "72.0%",
                "is_correct": False,
                "all_probabilities": [0.15, 0.72, 0.13]
            },
            "EfficientNet": {
                "predicted_class": "Normal",
                "predicted_class_index": 2,
                "confidence": 0.91,
                "confidence_percentage": "91.0%",
                "is_correct": True,
                "all_probabilities": [0.03, 0.06, 0.91]
            }
        }
    }

@st.cache_data
def create_dvc_report():
    """Créer un rapport basé sur les données DVC"""
    data_info = load_dvc_data_info()
    
    # Classes par défaut si non trouvées
    classes = data_info.get("classes", ["Abnormal", "Benign", "Normal"])
    
    return {
        "metadata": {
            "project": "SIPAKMED - Données DVC",
            "classes": classes,
            "data_path": data_info["path"],
            "dvc_managed": True,
            "total_files": len(data_info["files"]),
            "classes_count": len(classes)
        },
        "image_info": {
            "true_label": classes[0] if classes else "Normal",
            "source": "DVC data/processed"
        },
        "results_summary": {
            "accuracy_percentage": 66.7,
            "data_source": "DVC Managed",
            "note": "Exécutez l'analyse pour générer des résultats réels"
        },
        "model_details": {
            "ResNet": {
                "predicted_class": classes[0] if classes else "Normal",
                "confidence": 0.85,
                "is_correct": True,
                "status": "✅ Correct"
            },
            "MobileNet": {
                "predicted_class": classes[1] if len(classes) > 1 else "Benign",
                "confidence": 0.72,
                "is_correct": False,
                "status": "❌ Incorrect"
            },
            "EfficientNet": {
                "predicted_class": classes[0] if classes else "Normal",
                "confidence": 0.91,
                "is_correct": True,
                "status": "✅ Correct"
            }
        }
    }

# ============================================================================
# SECTION : PRÉDICTIONS EN TEMPS RÉEL
# ============================================================================

if mode == "🚀 Prédictions en temps réel":
    
    st.header("🚀 Prédictions en Temps Réel")
    st.markdown("Analyse d'images uploadées avec les modèles DVC")
    
    # Charger les modèles
    with st.spinner("Chargement des modèles depuis DVC..."):
        models = load_models_from_dvc()
    
    if not models:
        st.error("❌ Aucun modèle chargé depuis DVC")
        st.info("💡 Pour télécharger les modèles :")
        st.code("""
# Dans un terminal :
dvc pull src/models.dvc

# Ou copiez manuellement dans models/ :
# - resnet02.h5/.keras
# - efficientnet02.h5/.keras  
# - mobilenet01.h5/.keras
        """)
    else:
        st.success(f"✅ {len(models)} modèles chargés: {', '.join(models.keys())}")
        
        # Upload d'images
        st.subheader("📁 Chargement des images")
        uploaded_files = st.file_uploader(
            "Sélectionnez des images médicales",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Sélectionnez jusqu'à 10 images"
        )
        
        if uploaded_files:
            # Limiter le nombre d'images
            files_to_process = uploaded_files[:batch_size]
            
            st.info(f"📸 {len(files_to_process)} images sélectionnées")
            
            # Afficher les images
            cols = st.columns(min(4, len(files_to_process)))
            images = []
            
            for idx, uploaded_file in enumerate(files_to_process):
                with cols[idx % len(cols)]:
                    image = Image.open(uploaded_file)
                    images.append(image)
                    st.image(image, caption=f"Image {idx+1}", use_container_width=True)
            
            # Bouton pour lancer les prédictions
            if st.button("🚀 Lancer l'analyse batch", type="primary"):
                with st.spinner("Exécution des prédictions sur tous les modèles..."):
                    # Classes
                    CLASS_NAMES = ["Abnormal", "Benign", "Normal"]
                    
                    # Faire les prédictions
                    results = make_batch_predictions(images, models, CLASS_NAMES)
                    
                    # Afficher les résultats
                    st.success(f"✅ {len(results)} images analysées")
                    
                    # Résumé global
                    st.subheader("📊 Résumé des prédictions")
                    
                    # Tableau de synthèse
                    summary_data = []
                    for img_key, img_data in results.items():
                        for model_name, pred_data in img_data["predictions"].items():
                            summary_data.append({
                                "Image": img_key,
                                "Modèle": model_name,
                                "Prédiction": pred_data["predicted_class"],
                                "Confiance": pred_data["confidence_percentage"]
                            })
                    
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)
                    
                    # Détails par image
                    if show_predictions_details:
                        st.subheader("🔍 Détails par image")
                        
                        for img_key, img_data in results.items():
                            with st.expander(f"{img_key} - {img_data['image_shape']}", expanded=False):
                                cols = st.columns(len(models))
                                
                                for idx, (model_name, pred_data) in enumerate(img_data["predictions"].items()):
                                    with cols[idx]:
                                        st.markdown(f"**{model_name}**")
                                        
                                        # Affichage avec couleur selon la confiance
                                        confidence = pred_data["confidence"]
                                        if confidence > 0.8:
                                            st.success(f"🔴 {pred_data['predicted_class']}")
                                        elif confidence > 0.5:
                                            st.info(f"🟡 {pred_data['predicted_class']}")
                                        else:
                                            st.warning(f"🔵 {pred_data['predicted_class']}")
                                        
                                        st.metric("Confiance", pred_data["confidence_percentage"])
                                        
                                        # Graphique des probabilités
                                        fig, ax = plt.subplots(figsize=(4, 3))
                                        probs = pred_data["all_probabilities"]
                                        bars = ax.bar(CLASS_NAMES, probs, color=['red', 'orange', 'green'])
                                        ax.set_ylim([0, 1])
                                        ax.set_ylabel("Probabilité")
                                        ax.set_title(f"{model_name}")
                                        plt.xticks(rotation=45)
                                        st.pyplot(fig)
                    
                    # Comparaison des modèles
                    st.subheader("📈 Comparaison des modèles")
                    
                    # Calculer la confiance moyenne par modèle
                    model_stats = {}
                    for img_data in results.values():
                        for model_name, pred_data in img_data["predictions"].items():
                            if model_name not in model_stats:
                                model_stats[model_name] = []
                            model_stats[model_name].append(pred_data["confidence"])
                    
                    # Graphique de comparaison
                    fig, ax = plt.subplots(figsize=(10, 5))
                    model_names = list(model_stats.keys())
                    avg_confidences = [np.mean(model_stats[m]) for m in model_names]
                    
                    bars = ax.bar(model_names, avg_confidences, alpha=0.7, 
                                 color=['blue', 'green', 'orange'])
                    ax.set_ylabel("Confiance moyenne")
                    ax.set_ylim([0, 1])
                    ax.set_title("Performance comparative des modèles")
                    ax.grid(True, alpha=0.3)
                    
                    # Ajouter les valeurs
                    for bar, avg in zip(bars, avg_confidences):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                               f'{avg:.1%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    # Bouton d'export
                    st.download_button(
                        label="📥 Exporter les résultats en JSON",
                        data=json.dumps(results, indent=2, ensure_ascii=False),
                        file_name=f"batch_predictions_{len(results)}_images.json",
                        mime="application/json"
                    )
        
        else:
            st.info("📤 Veuillez uploader des images pour commencer l'analyse")

# ============================================================================
# SECTIONS EXISTANTES (pour les autres modes)
# ============================================================================

else:
    # Charger le rapport selon le mode
    if mode == "📊 Rapports réels générés":
        if 'selected_report' in locals() and selected_report:
            report = load_real_report(selected_report)
            if report is None:
                st.warning("Utilisation du mode démonstration suite à l'erreur")
                report = load_demo_report()
        else:
            report = load_demo_report()
            
    elif mode == "📁 Données DVC (data/processed)":
        report = create_dvc_report()
    else:
        report = load_demo_report()
    
    # ============================================================================
    # AFFICHAGE DES MÉTRIQUES
    # ============================================================================
    
    st.header("📈 Résultats d'analyse")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = report.get("results_summary", {}).get("accuracy_percentage", 0)
        st.metric("Exactitude globale", f"{accuracy:.1f}%")
    
    with col2:
        true_label = report.get("image_info", {}).get("true_label", "Inconnu")
        st.metric("Vérité terrain", true_label)
    
    with col3:
        if "results_summary" in report:
            correct = report["results_summary"].get("correct_predictions", 0)
            total = report["results_summary"].get("total_models", 0)
            st.metric("Prédictions correctes", f"{correct}/{total}")
        else:
            st.metric("Prédictions correctes", "N/A")
    
    with col4:
        models = list(report.get("model_details", {}).keys())
        st.metric("Modèles analysés", len(models))
    
    # ============================================================================
    # IMAGE DE TEST
    # ============================================================================
    
    st.markdown("---")
    st.subheader("🖼️ Image analysée")
    
    # Chercher des images dans différents dossiers
    img_patterns = [
        "reports/explainability/*.png",
        "reports/predictions/*.png",
        "reports/*.png",
        "results/*.png",
        "comparison*.png",
        "gradcam*.png"
    ]
    
    image_files = []
    for pattern in img_patterns:
        image_files.extend(glob.glob(pattern))
    
    if image_files:
        # Afficher plusieurs images si disponibles
        cols = st.columns(min(3, len(image_files)))
        for idx, img_path in enumerate(image_files[:3]):
            try:
                with cols[idx]:
                    image = Image.open(img_path)
                    st.image(image, 
                            caption=os.path.basename(img_path),
                            use_container_width=True)
            except Exception as e:
                st.error(f"Erreur chargement {img_path}: {e}")
    else:
        # Si aucune image n'est trouvée, afficher un placeholder
        st.info("Exécutez l'analyse pour générer les visualisations d'images")
        
        # Créer une image de placeholder
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Image de test\n(à générer)", 
                ha='center', va='center', fontsize=20)
        ax.axis('off')
        st.pyplot(fig)
    
    # ============================================================================
    # ANALYSE PAR MODÈLE
    # ============================================================================
    
    st.markdown("---")
    st.header("🤖 Prédictions par modèle")
    
    model_details = report.get("model_details", {})
    
    if model_details:
        # Créer des colonnes pour chaque modèle
        cols = st.columns(len(model_details))
        
        for idx, (model_name, details) in enumerate(model_details.items()):
            with cols[idx]:
                # Carte du modèle
                with st.container():
                    st.subheader(model_name)
                    
                    # Statut
                    is_correct = details.get("is_correct", False)
                    predicted_class = details.get("predicted_class", "N/A")
                    
                    if is_correct:
                        st.success(f"✅ CORRECT: {predicted_class}")
                    else:
                        st.error(f"❌ INCORRECT: {predicted_class}")
                    
                    # Confiance
                    confidence = details.get("confidence", 0)
                    st.metric("Confiance", f"{confidence:.1%}")
                    
                    # Graphique des probabilités
                    try:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        
                        classes = report.get("metadata", {}).get("classes", ["C1", "C2", "C3"])
                        probs = details.get("all_probabilities", [])
                        
                        if not probs:
                            # Créer des probabilités fictives
                            probs = np.random.dirichlet(np.ones(3))
                        
                        bars = ax.bar(range(len(probs[:3])), probs[:3], 
                                     color=['lightblue']*min(3, len(probs)))
                        
                        # Colorer la prédiction
                        pred_idx = details.get("predicted_class_index", 0)
                        if pred_idx < len(bars):
                            bars[pred_idx].set_color('green')
                        
                        # Colorer la vraie classe si différente
                        true_idx = report.get("image_info", {}).get("true_class_index", -1)
                        if 0 <= true_idx < len(bars) and true_idx != pred_idx:
                            bars[true_idx].set_color('red')
                        
                        ax.set_xticks(range(min(3, len(classes))))
                        ax.set_xticklabels(classes[:3], rotation=45, fontsize=8)
                        ax.set_ylim([0, 1])
                        ax.set_ylabel("Probabilité")
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur graphique: {e}")
    
    # ============================================================================
    # TABLEAU COMPARATIF
    # ============================================================================
    
    st.markdown("---")
    st.header("📋 Comparaison des modèles")
    
    if model_details:
        # Créer le tableau de comparaison
        comparison_data = []
        for model_name, details in model_details.items():
            comparison_data.append({
                "Modèle": model_name,
                "Prédiction": details.get("predicted_class", "N/A"),
                "Confiance": f"{details.get('confidence', 0):.1%}",
                "Statut": "✅ Correct" if details.get("is_correct") else "❌ Incorrect",
                "Vérité": "✓" if details.get("is_correct") else "✗"
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Afficher avec style conditionnel
        def color_status(val):
            color = 'green' if "Correct" in val else 'red'
            return f'background-color: {color}; color: white'
        
        styled_df = df.style.applymap(color_status, subset=['Statut'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Graphique de comparaison
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            models = list(model_details.keys())
            confidences = [model_details[m].get("confidence", 0) for m in models]
            colors = ['green' if model_details[m].get("is_correct") else 'red' 
                     for m in models]
            
            bars = ax.bar(models, confidences, color=colors, edgecolor='black', alpha=0.8)
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Seuil 50%')
            ax.set_ylabel("Confiance", fontsize=12)
            ax.set_ylim([0, 1])
            ax.set_title("Comparaison des niveaux de confiance", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Ajouter les valeurs
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{conf:.1%}', ha='center', va='bottom', fontsize=10)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur création graphique: {e}")
    
    # ============================================================================
    # DONNÉES DÉTAILLÉES
    # ============================================================================
    
    st.markdown("---")
    st.header("🔍 Données détaillées")
    
    with st.expander("Afficher les données brutes", expanded=False):
        st.json(report)
    
    # ============================================================================
    # SECTION DVC SPÉCIFIQUE
    # ============================================================================
    
    if mode == "📁 Données DVC (data/processed)":
        st.markdown("---")
        st.header("📂 Informations DVC")
        
        data_info = load_dvc_data_info()
        
        if data_info["exists"]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fichiers", len(data_info["files"]))
            
            with col2:
                st.metric("Classes", len(data_info.get("classes", [])))
            
            with col3:
                st.metric("Chemin", data_info["path"].split("/")[-1])
            
            # Afficher les classes
            if data_info.get("classes"):
                st.subheader("Classes disponibles:")
                classes_text = ", ".join(data_info["classes"])
                st.write(classes_text)
                
                # Bouton pour explorer les données
                if st.button("🔄 Explorer les données DVC"):
                    st.info("Exécutez `dvc status` pour voir l'état actuel")
                    
                    try:
                        result = subprocess.run(['dvc', 'status'], 
                                               capture_output=True, text=True)
                        if result.stdout:
                            st.code(result.stdout)
                        else:
                            st.success("✅ Données DVC à jour")
                    except Exception as e:
                        st.error(f"Erreur DVC: {e}")
        else:
            st.warning("Le répertoire data/processed n'existe pas")
            
            if st.button("📥 Télécharger avec DVC"):
                st.info("Exécutez: `dvc pull data/processed.dvc`")

# ============================================================================
# GUIDE ET INFORMATIONS
# ============================================================================

if mode != "🚀 Prédictions en temps réel":
    st.markdown("---")
    st.header("📚 Guide d'utilisation")
    
    with st.expander("Comment générer vos propres rapports", expanded=False):
        st.markdown("""
        ### Étapes pour générer vos rapports :
        
        1. **Exécuter l'analyse complète** dans votre notebook
        2. **Les rapports seront générés** dans :
           - `reports/predictions/` - Prédictions détaillées
           - Visualisations PNG automatiquement créées
        
        3. **Format attendu des rapports JSON :**
        ```json
        {
            "metadata": {
                "project": "Votre projet",
                "classes": ["class1", "class2", ...],
                "data_path": "chemin/vers/données"
            },
            "image_info": {
                "true_label": "Nom de la classe",
                "image_shape": [224, 224, 3]
            },
            "results_summary": {
                "accuracy_percentage": 66.7,
                "correct_predictions": 2,
                "total_models": 3
            },
            "model_details": {
                "Model1": {
                    "predicted_class": "class1",
                    "confidence": 0.85,
                    "is_correct": true,
                    "all_probabilities": [0.1, 0.85, 0.05]
                }
            }
        }
        ```
        
        4. **Pour les prédictions en temps réel** :
           - Sélectionnez le mode "Prédictions en temps réel"
           - Uploadez vos images
           - Analysez avec les modèles DVC
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_cols = st.columns(4)

with footer_cols[0]:
    st.caption("**Projet MLOps**")
    st.caption("M2 SID 2025-2026")

with footer_cols[1]:
    st.caption(f"**Dataset:** SIPAKMED")
    classes = report.get("metadata", {}).get("classes", ["Abnormal", "Benign", "Normal"]) if 'report' in locals() else ["Abnormal", "Benign", "Normal"]
    st.caption(f"{len(classes)} classes")

with footer_cols[2]:
    if mode == "🚀 Prédictions en temps réel":
        if 'models' in locals():
            st.caption(f"**Modèles:** {len(models)}")
        else:
            st.caption("**Modèles:** Chargement...")
    else:
        st.caption("**Modèles analysés**")
        if 'model_details' in locals() and model_details:
            st.caption(f"{len(model_details)} modèles")

with footer_cols[3]:
    if mode == "📁 Données DVC (data/processed)":
        st.caption("**Versioning:** DVC")
    elif mode == "🚀 Prédictions en temps réel":
        st.caption("**Mode:** Temps réel")
    else:
        st.caption("**Source:** Rapports générés")

st.markdown("---")
st.caption("Dashboard d'explicabilité | Interface Streamlit | Projet MLOps")