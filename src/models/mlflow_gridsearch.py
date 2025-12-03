"""
Script qui importe TES modèles existants de ta structure
"""

import mlflow
import mlflow.keras
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# CONFIGURATION DES CHEMINS POUR TA STRUCTURE
# -------------------------------------------------------------------
# Le script est dans src/models/, donc parent est src, grand-parent est projet
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
parent_dir = os.path.dirname(current_dir)  # src/
project_dir = os.path.dirname(parent_dir)  # projet/ (APPRENTISAAGE-1)

src_path = parent_dir  # déjà src
data_path = os.path.join(project_dir, 'data')

sys.path.extend([src_path, data_path, project_dir])

print(f"📁 Dossier courant: {current_dir}")
print(f"📁 Dossier src: {src_path}")
print(f"📁 Dossier data: {data_path}")
print(f"📁 Dossier projet: {project_dir}")

# -------------------------------------------------------------------
# IMPORT DE TES MODÈLES EXISTANTS
# -------------------------------------------------------------------
print("\n🔍 Tentative d'import des modèles...")

try:
    # IMPORT CORRIGÉ : importer depuis le même dossier
    from resnet_model import ResNet50_Model
    from efficient_model import EfficientNetB0_Model
    from mobilenet_model import MobileNetV2_Model
    print("✅ Modèles importés avec succès")
    
    # Créer des fonctions wrapper pour tes classes
    def create_resnet50_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, **kwargs):
        model = ResNet50_Model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate
        )
        return model.get_model()
    
    def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, **kwargs):
        model = EfficientNetB0_Model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate
        )
        return model.get_model()
    
    def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, **kwargs):
        model = MobileNetV2_Model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate
        )
        return model.get_model()
        
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Contenu du dossier courant:")
    for f in os.listdir(current_dir):
        print(f"  - {f}")
    
    # Essayer une méthode alternative
    try:
        # Ajouter le chemin actuel
        sys.path.append(current_dir)
        
        # Importer directement
        import resnet_model
        import efficient_model
        import mobilenet_model
        
        # Chercher les classes
        ResNet50_Model = resnet_model.ResNet50_Model
        EfficientNetB0_Model = efficient_model.EfficientNetB0_Model
        MobileNetV2_Model = mobilenet_model.MobileNetV2_Model
        
        print("✅ Modèles importés avec méthode alternative")
        
        # Redéfinir les fonctions wrapper
        def create_resnet50_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, **kwargs):
            model = ResNet50_Model(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=learning_rate
            )
            return model.get_model()
        
        def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, **kwargs):
            model = EfficientNetB0_Model(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=learning_rate
            )
            return model.get_model()
        
        def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, **kwargs):
            model = MobileNetV2_Model(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=learning_rate
            )
            return model.get_model()
            
    except Exception as e2:
        print(f"❌ Échec méthode alternative: {e2}")
        sys.exit(1)

# -------------------------------------------------------------------
# IMPORT DE DATALOADERS
# -------------------------------------------------------------------
try:
    from Databoaders import DataLoader
    print("✅ DataLoader importé")
except ImportError as e:
    print(f"⚠️ DataLoader non trouvé ({e}), création d'un mock...")
    
    # Vérifier si le fichier existe
    databoaders_path = os.path.join(data_path, 'Databoaders.py')
    if os.path.exists(databoaders_path):
        print(f"ℹ️  Fichier trouvé: {databoaders_path}")
        # Ajouter au path et réessayer
        sys.path.append(data_path)
        try:
            from Databoaders import DataLoader
            print("✅ DataLoader importé après ajustement")
        except:
            pass
    else:
        print(f"⚠️  Fichier non trouvé: {databoaders_path}")
    
    # Mock pour tester
    class DataLoader:
        def __init__(self, data_dir, batch_size=32, img_size=(224, 224)):
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.img_size = img_size
        
        def create_generators(self, validation_split=0.2):
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            val_generator = val_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            return train_generator, val_generator, train_generator.class_indices

# -------------------------------------------------------------------
# 1. CONFIGURATION MLflow
# -------------------------------------------------------------------
def setup_mlflow(experiment_name="CNN_Models_GridSearch"):
    """Configure MLflow pour le tracking"""
    # MLflow local
    mlflow.set_tracking_uri("http://localhost:5000")  # À adapter si distant
    mlflow.set_experiment(experiment_name)
    
    # Tags
    mlflow.set_tag("project", "CNN_Classification")
    mlflow.set_tag("team", "Data_Science_Team")
    mlflow.set_tag("mlflow.note.content", "GridSearch sur modèles CNN")

# -------------------------------------------------------------------
# 2. FONCTION D'ENTRAÎNEMENT POUR CNN
# -------------------------------------------------------------------
def train_cnn_model(model_func, params, train_gen, val_gen, epochs=10):
    """Entraîne un modèle CNN"""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Créer le modèle
    model = model_func(**params)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ]
    
    # Entraînement
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# -------------------------------------------------------------------
# 3. GRIDSEARCH ADAPTÉ POUR CNN
# -------------------------------------------------------------------
def run_cnn_gridsearch(model_func, param_grid, data_loader, model_name):
    """
    GridSearch adapté pour les modèles CNN avec générateurs
    """
    run_name = f"{model_name}_GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n🚀 Début GridSearch pour {model_name}")
        
        # Préparer les résultats
        all_results = []
        
        # Parcourir toutes les combinaisons
        import itertools
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"🔍 Nombre de combinaisons à tester: {len(param_combinations)}")
        
        best_score = 0
        best_params = None
        best_model = None
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}] Test des paramètres: {params}")
            
            # Log des paramètres dans MLflow
            for key, value in params.items():
                mlflow.log_param(f"{key}_run{i}", value)
            
            try:
                # Créer les générateurs avec les paramètres actuels
                train_gen, val_gen, class_indices = data_loader.create_generators(
                    validation_split=0.2
                )
                
                # Mettre à jour les paramètres du modèle
                model_params = {
                    'input_shape': train_gen.image_shape,
                    'num_classes': len(class_indices),
                    'learning_rate': params.get('learning_rate', 0.001),
                    'dropout_rate': params.get('dropout_rate', 0.5)
                }
                
                # Entraîner le modèle
                model, history = train_cnn_model(
                    model_func=model_func,
                    params=model_params,
                    train_gen=train_gen,
                    val_gen=val_gen,
                    epochs=params.get('epochs', 10)
                )
                
                # Évaluation
                val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
                
                # Log des métriques
                mlflow.log_metric(f"val_accuracy_run{i}", val_accuracy)
                mlflow.log_metric(f"val_loss_run{i}", val_loss)
                
                # Sauvegarder les résultats
                result = {
                    'params': params,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'history': history.history
                }
                all_results.append(result)
                
                # Mettre à jour le meilleur modèle
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params
                    best_model = model
                    best_history = history
                    
                    print(f"🎯 Nouveau meilleur score: {val_accuracy:.4f}")
                    
            except Exception as e:
                print(f"❌ Erreur avec les paramètres {params}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log des meilleurs résultats
        if best_model:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_accuracy", best_score)
            
            # Sauvegarder le meilleur modèle
            model_path = f"models/{model_name}_best.h5"
            best_model.save(model_path)
            mlflow.log_artifact(model_path)
            
            # Sauvegarder l'historique
            history_path = f"results/{model_name}_history.csv"
            pd.DataFrame(best_history.history).to_csv(history_path, index=False)
            mlflow.log_artifact(history_path)
            
            # Sauvegarder les résultats complets
            results_path = f"results/{model_name}_all_results.json"
            with open(results_path, 'w') as f:
                # Convertir pour JSON
                json_results = []
                for r in all_results:
                    json_results.append({
                        'params': r['params'],
                        'val_accuracy': float(r['val_accuracy']),
                        'val_loss': float(r['val_loss'])
                    })
                json.dump(json_results, f, indent=4)
            mlflow.log_artifact(results_path)
        
        print(f"\n✅ GridSearch terminé pour {model_name}")
        print(f"🏆 Meilleur score: {best_score:.4f}")
        print(f"⚙️ Meilleurs paramètres: {best_params}")
        
        return best_model, best_score, best_params, all_results

# -------------------------------------------------------------------
# 4. DÉFINITION DES GRIDS POUR TES MODÈLES
# -------------------------------------------------------------------
def get_mobilenet_grid():
    """Grid pour MobileNetV2"""
    return {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32],
        'epochs': [10, 15],
        'dropout_rate': [0.3, 0.5]
    }

def get_resnet50_grid():
    """Grid pour ResNet50"""
    return {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [16, 32],
        'epochs': [10, 15],
        'dropout_rate': [0.3, 0.5]
    }

def get_efficientnet_grid():
    """Grid pour EfficientNet"""
    return {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [16, 32],
        'epochs': [10, 15],
        'dropout_rate': [0.3, 0.5]
    }

# -------------------------------------------------------------------
# 5. SCRIPT PRINCIPAL
# -------------------------------------------------------------------
def main():
    """Script principal"""
    
    # Créer les dossiers dans le répertoire du projet
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Dossier models: {models_dir}")
    print(f"📁 Dossier results: {results_dir}")
    
    # Configurer MLflow
    setup_mlflow()
    
    # Charger les données
    print("\n📂 Chargement des données...")
    
    # Adapter le chemin à tes données
    data_dir = os.path.join(data_path, "Data_project")
    if not os.path.exists(data_dir):
        print(f"❌ Dossier de données non trouvé: {data_dir}")
        print("Contenu du dossier data:")
        if os.path.exists(data_path):
            for item in os.listdir(data_path):
                print(f"  - {item}")
        print("Veuillez adapter le chemin dans le code")
        return
    
    print(f"✅ Dossier de données trouvé: {data_dir}")
    
    # Créer le DataLoader
    data_loader = DataLoader(
        data_dir=data_dir,
        batch_size=32,
        img_size=(224, 224)
    )
    
    # Définir les modèles à tester
    models_config = [
        ("MobileNetV2", create_mobilenet_model, get_mobilenet_grid()),
        ("ResNet50", create_resnet50_model, get_resnet50_grid()),
        ("EfficientNetB0", create_efficientnet_model, get_efficientnet_grid())
    ]
    
    # Exécuter GridSearch pour chaque modèle
    all_results = {}
    
    for model_name, model_func, param_grid in models_config:
        print(f"\n{'='*60}")
        print(f"🧠 TRAITEMENT DU MODÈLE: {model_name}")
        print(f"{'='*60}")
        
        best_model, best_score, best_params, results = run_cnn_gridsearch(
            model_func=model_func,
            param_grid=param_grid,
            data_loader=data_loader,
            model_name=model_name
        )
        
        all_results[model_name] = {
            'best_score': best_score,
            'best_params': best_params,
            'model': best_model
        }
        
        # Nettoyage mémoire
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
        except:
            pass
    
    # Afficher les résultats finaux
    print(f"\n{'='*60}")
    print("📊 RÉSULTATS FINAUX")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        print(f"{model_name:20} | Accuracy: {results['best_score']:.4f}")
        print(f"                  | Params: {results['best_params']}")
        print("-" * 60)
    
    # Sauvegarder le rapport final
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }
    
    report_path = os.path.join(results_dir, "final_gridsearch_report.json")
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=4, default=str)
    
    print(f"\n✅ Rapport final sauvegardé: {report_path}")
    print("🎉 GridSearch terminé avec succès !")

# -------------------------------------------------------------------
# 6. EXÉCUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Vérifications préalables
    print("🔧 Vérification de l'environnement...")
    
    # Vérifier MLflow
    try:
        import mlflow
        print("✅ MLflow est installé")
    except ImportError:
        print("❌ MLflow n'est pas installé. Installez-le avec: pip install mlflow")
        sys.exit(1)
    
    # Vérifier TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow est installé (version: {tf.__version__})")
    except ImportError:
        print("⚠️  TensorFlow n'est pas installé. Tentative d'installation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.10.0"])
        print("✅ TensorFlow installé")
    
    # Lancer le script principal
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Script interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()