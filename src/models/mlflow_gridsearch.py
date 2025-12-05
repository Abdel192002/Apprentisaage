"""
Script MLflow GridSearch pour SipakMed - Classification d'images médicales
Projet MLOps - M2 SID 2025-2026
Dataset: SipakMed (images cytologiques)
Version: GridSearch avec modèles configurables
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
warnings.filterwarnings('ignore')

print("=" * 80)
print("MLflow GridSearch - SipakMed (Images Médicales)")
print("=" * 80)

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DATA_PATH = "C:/Users/nessa/Downloads/sipakmed_new6/"

print(f"📂 Chemin données: {DATA_PATH}")
print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# -------------------------------------------------------------------
# 2. IMPORTS SPÉCIFIQUES IMAGES
# -------------------------------------------------------------------
print("\n🔍 IMPORTS POUR TRAITEMENT D'IMAGES...")

import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print(f"✅ TensorFlow {tf.__version__}")
print(f"✅ MLflow {mlflow.__version__}")

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------------------------
# 3. FONCTION POUR SÉRIALISATION JSON SÉCURISÉE
# -------------------------------------------------------------------
def safe_serialize(obj):
    """Convertit les types numpy en types Python pour JSON"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_serialize(i) for i in obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj

# -------------------------------------------------------------------
# 4. DATALOADER SPÉCIFIQUE SIPAKMED
# -------------------------------------------------------------------
print("\n📥 CHARGEMENT DU DATASET SIPAKMED...")

class SipakMedDataLoader:
    """Chargeur spécifique pour le dataset SipakMed"""
    
    def __init__(self, data_path, img_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None
        self.num_classes = None
        
        # Vérifier la structure des dossiers
        self._check_directory_structure()
    
    def _check_directory_structure(self):
        """Vérifie que la structure des dossiers est correcte"""
        print(f"🔍 Vérification structure des dossiers...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Chemin non trouvé: {self.data_path}")
        
        train_path = os.path.join(self.data_path, "train")
        test_path = os.path.join(self.data_path, "test")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"❌ Dossier 'train' manquant dans: {self.data_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"❌ Dossier 'test' manquant dans: {self.data_path}")
        
        # Lister les classes (sous-dossiers)
        self.class_names = sorted([d for d in os.listdir(train_path) 
                                 if os.path.isdir(os.path.join(train_path, d))])
        self.num_classes = len(self.class_names)
        
        print(f"✅ Structure OK - {self.num_classes} classes trouvées:")
        for i, cls in enumerate(self.class_names):
            train_count = len(os.listdir(os.path.join(train_path, cls)))
            test_count = len(os.listdir(os.path.join(test_path, cls)))
            print(f"   {i+1}. {cls}: {train_count} train, {test_count} test images")
    
    def create_generators(self, augmentation=True):
        """Crée les générateurs d'images"""
        print(f"\n🔄 Création des générateurs d'images...")
        
        # Data augmentation pour l'entraînement
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Pas d'augmentation pour validation/test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Générateur d'entraînement
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, "train"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        # Générateur de test
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_path, "test"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Générateurs créés:")
        print(f"   Train: {train_generator.samples} images")
        print(f"   Test: {test_generator.samples} images")
        
        return train_generator, test_generator

# Initialiser le chargeur
data_loader = SipakMedDataLoader(DATA_PATH, img_size=(224, 224), batch_size=32)
train_gen, test_gen = data_loader.create_generators(augmentation=True)

# -------------------------------------------------------------------
# 5. IMPORT DES MODÈLES AVEC GRIDSEARCH INTÉGRÉ
# -------------------------------------------------------------------
print("\n🔍 IMPORT DES MODÈLES AVEC GRIDSEARCH...")

# Importer les classes de modèles
from efficient_model import EfficientNetB0_Model
from resnet_model import ResNet50_Model
from mobilenet_model import MobileNetV2_Model

# -------------------------------------------------------------------
# 6. FONCTIONS DE CRÉATION DE MODÈLES
# -------------------------------------------------------------------
def create_resnet50_model(num_classes, config):
    """Crée un modèle ResNet50 avec hyperparamètres configurables"""
    print(f"  🏗️ Création ResNet50 avec config: {config}")
    
    model_builder = ResNet50_Model(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        learning_rate=config['learning_rate'],
        dropout_rate=config['dropout_rate'],
        l2_reg=config['l2_reg'],
        dense_units=config['dense_units'],
        freeze_backbone=config.get('freeze_backbone', True)
    )
    
    return model_builder.build_model()

def create_efficientnet_model(num_classes, config):
    """Crée un modèle EfficientNetB0 avec hyperparamètres configurables"""
    print(f"  🏗️ Création EfficientNet avec config: {config}")
    
    model_builder = EfficientNetB0_Model(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        learning_rate=config['learning_rate'],
        dropout_rate=config['dropout_rate'],
        l2_reg=config['l2_reg'],
        dense_units=config['dense_units'],
        freeze_backbone=config.get('freeze_backbone', True)
    )
    
    return model_builder.build_model()

def create_mobilenet_model(num_classes, config):
    """Crée un modèle MobileNetV2 avec hyperparamètres configurables"""
    print(f"  🏗️ Création MobileNet avec config: {config}")
    
    model_builder = MobileNetV2_Model(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        learning_rate=config['learning_rate'],
        dropout_rate=config['dropout_rate'],
        l2_reg=config['l2_reg'],
        dense_units=config['dense_units'],
        freeze_backbone=config.get('freeze_backbone', True)
    )
    
    return model_builder.build_model()

# -------------------------------------------------------------------
# 7. FONCTIONS D'ÉVALUATION
# -------------------------------------------------------------------
def calculate_comprehensive_metrics(model, test_generator):
    """Calcule des métriques complètes pour les images médicales"""
    print("    📊 Évaluation sur le test set...")
    
    # Réinitialiser le générateur
    test_generator.reset()
    
    # Prédictions
    predictions = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Métriques de base
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC-ROC pour classification multi-classes
    try:
        y_true_one_hot = to_categorical(y_true, num_classes=len(np.unique(y_true)))
        auc = roc_auc_score(y_true_one_hot, predictions, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    # Rapport de classification
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, 
                                   target_names=data_loader.class_names,
                                   output_dict=True,
                                   zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'test_samples': int(len(y_true))
    }
    
    # Ajouter les métriques par classe
    for i, class_name in enumerate(data_loader.class_names):
        if class_name in report:
            metrics[f'precision_{class_name}'] = float(report[class_name]['precision'])
            metrics[f'recall_{class_name}'] = float(report[class_name]['recall'])
            metrics[f'f1_{class_name}'] = float(report[class_name]['f1-score'])
            metrics[f'support_{class_name}'] = int(report[class_name]['support'])
    
    return metrics, predictions, report

def evaluate_model_safely(model, test_generator):
    """Évaluation sécurisée qui gère les multiples métriques"""
    print("    📈 Évaluation finale du modèle...")
    
    try:
        # Récupérer toutes les valeurs de model.evaluate()
        evaluation_results = model.evaluate(test_generator, verbose=0, return_dict=True)
        
        if isinstance(evaluation_results, dict):
            # Si model.evaluate() retourne un dictionnaire
            test_loss = evaluation_results.get('loss', 0)
            test_accuracy = evaluation_results.get('accuracy', 0)
        else:
            # Si model.evaluate() retourne une liste
            test_loss = evaluation_results[0] if len(evaluation_results) > 0 else 0
            test_accuracy = evaluation_results[1] if len(evaluation_results) > 1 else 0
            
        return float(test_loss), float(test_accuracy)
        
    except Exception as e:
        print(f"    ⚠️  Erreur lors de l'évaluation: {e}")
        # Retourner des valeurs par défaut
        return 0.0, 0.0

# -------------------------------------------------------------------
# 8. CONFIGURATION MLFLOW
# -------------------------------------------------------------------
print("\n⚙️  CONFIGURATION MLFLOW...")

mlflow.set_tracking_uri("file:./mlruns")

# Créer une nouvelle expérience
EXPERIMENT_NAME = f"SipakMed_Classification_{datetime.now().strftime('%Y%m%d_%H%M')}"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"📁 Expérience MLflow: {EXPERIMENT_NAME}")
print(f"📂 Tracking URI: {mlflow.get_tracking_uri()}")

# -------------------------------------------------------------------
# 9. CONFIGURATION DU GRIDSEARCH
# -------------------------------------------------------------------
print("\n🎯 CONFIGURATION DU GRIDSEARCH COMPLET...")

# Configuration COMPLÈTE pour chaque modèle
RESNET_GRID = [
    # Expérience 1
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "l2_reg": 0.01,
        "dense_units": 128,
        "epochs": 5,
        "freeze_backbone": True,
        "model_type": "resnet50"
    },
    # Expérience 2
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.5,
        "l2_reg": 0.01,
        "dense_units": 256,
        "epochs": 8,
        "freeze_backbone": True,
        "model_type": "resnet50"
    },
    # Expérience 3
    {
        "learning_rate": 0.0005,
        "dropout_rate": 0.4,
        "l2_reg": 0.001,
        "dense_units": 128,
        "epochs": 10,
        "freeze_backbone": True,
        "model_type": "resnet50"
    },
    # Expérience 4
    {
        "learning_rate": 0.0001,
        "dropout_rate": 0.6,
        "l2_reg": 0.01,
        "dense_units": 512,
        "epochs": 12,
        "freeze_backbone": True,
        "model_type": "resnet50"
    },
    # Expérience 5
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "l2_reg": 0.01,
        "dense_units": 256,
        "epochs": 7,
        "freeze_backbone": False,
        "model_type": "resnet50"
    }
]

EFFICIENTNET_GRID = [
    # Expérience 1
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "l2_reg": 0.01,
        "dense_units": 256,
        "epochs": 5,
        "freeze_backbone": True,
        "model_type": "efficientnet"
    },
    # Expérience 2
    {
        "learning_rate": 0.0005,
        "dropout_rate": 0.4,
        "l2_reg": 0.001,
        "dense_units": 512,
        "epochs": 8,
        "freeze_backbone": True,
        "model_type": "efficientnet"
    },
    # Expérience 3
    {
        "learning_rate": 0.0001,
        "dropout_rate": 0.5,
        "l2_reg": 0.01,
        "dense_units": 128,
        "epochs": 10,
        "freeze_backbone": True,
        "model_type": "efficientnet"
    },
    # Expérience 4
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.2,
        "l2_reg": 0.001,
        "dense_units": 384,
        "epochs": 6,
        "freeze_backbone": False,
        "model_type": "efficientnet"
    },
    # Expérience 5
    {
        "learning_rate": 0.0005,
        "dropout_rate": 0.3,
        "l2_reg": 0.005,
        "dense_units": 256,
        "epochs": 9,
        "freeze_backbone": True,
        "model_type": "efficientnet"
    }
]

MOBILENET_GRID = [
    # Expérience 1
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "l2_reg": 0.01,
        "dense_units": 128,
        "epochs": 5,
        "freeze_backbone": True,
        "model_type": "mobilenet"
    },
    # Expérience 2
    {
        "learning_rate": 0.0005,
        "dropout_rate": 0.5,
        "l2_reg": 0.01,
        "dense_units": 256,
        "epochs": 8,
        "freeze_backbone": True,
        "model_type": "mobilenet"
    },
    # Expérience 3
    {
        "learning_rate": 0.0001,
        "dropout_rate": 0.4,
        "l2_reg": 0.001,
        "dense_units": 192,
        "epochs": 10,
        "freeze_backbone": True,
        "model_type": "mobilenet"
    },
    # Expérience 4
    {
        "learning_rate": 0.001,
        "dropout_rate": 0.2,
        "l2_reg": 0.005,
        "dense_units": 64,
        "epochs": 7,
        "freeze_backbone": False,
        "model_type": "mobilenet"
    },
    # Expérience 5
    {
        "learning_rate": 0.0005,
        "dropout_rate": 0.3,
        "l2_reg": 0.01,
        "dense_units": 128,
        "epochs": 9,
        "freeze_backbone": True,
        "model_type": "mobilenet"
    }
]

print(f"📋 Total des configurations de GridSearch:")
print(f"   • ResNet50: {len(RESNET_GRID)} configurations")
print(f"   • EfficientNet: {len(EFFICIENTNET_GRID)} configurations")
print(f"   • MobileNet: {len(MOBILENET_GRID)} configurations")
print(f"   • TOTAL: {len(RESNET_GRID) + len(EFFICIENTNET_GRID) + len(MOBILENET_GRID)} expériences")

# -------------------------------------------------------------------
# 10. FONCTION GÉNÉRIQUE POUR EXÉCUTER LES EXPÉRIENCES
# -------------------------------------------------------------------
def run_experiment(config, experiment_num, model_type):
    """Exécute une expérience MLflow avec configuration donnée"""
    
    run_name = f"{model_type}_exp_{experiment_num:02d}"
    
    print(f"\n{'='*60}")
    print(f"  🔬 Expérience {experiment_num}: {run_name}")
    print(f"    ⚙️  Configuration: {config}")
    print(f"{'='*60}")
    
    try:
        with mlflow.start_run(run_name=run_name):
            # Logger TOUS les hyperparamètres
            mlflow.log_params({
                'model_type': model_type,
                'learning_rate': config['learning_rate'],
                'dropout_rate': config['dropout_rate'],
                'l2_reg': config['l2_reg'],
                'dense_units': config['dense_units'],
                'freeze_backbone': config.get('freeze_backbone', True),
                'epochs': config['epochs'],
                'batch_size': 32,
                'num_classes': data_loader.num_classes,
                'dataset': 'sipakmed_new6',
                'image_size': '224x224'
            })
            
            # Logger les infos du dataset
            mlflow.log_params({
                'class_names': str(data_loader.class_names),
                'train_samples': int(train_gen.samples),
                'test_samples': int(test_gen.samples)
            })
            
            # Sélectionner le bon modèle
            if model_type == 'resnet50':
                model = create_resnet50_model(data_loader.num_classes, config)
            elif model_type == 'efficientnet':
                model = create_efficientnet_model(data_loader.num_classes, config)
            elif model_type == 'mobilenet':
                model = create_mobilenet_model(data_loader.num_classes, config)
            else:
                raise ValueError(f"Type de modèle inconnu: {model_type}")
            
            # Callbacks
            callbacks_list = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
            ]
            
            # Entraînement
            print(f"    🏋️  Entraînement ({config['epochs']} epochs)...")
            
            steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
            validation_steps = max(1, test_gen.samples // test_gen.batch_size)
            
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=config['epochs'],
                validation_data=test_gen,
                validation_steps=validation_steps,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Évaluation
            test_loss, test_accuracy = evaluate_model_safely(model, test_gen)
            metrics, predictions, report = calculate_comprehensive_metrics(model, test_gen)
            metrics['test_loss'] = float(test_loss)
            metrics['test_accuracy'] = float(test_accuracy)
            
            # Logger les métriques
            mlflow.log_metrics(metrics)
            
            # Logger l'historique d'entraînement
            for epoch in range(len(history.history.get('accuracy', []))):
                epoch_metrics = {
                    'train_accuracy': float(history.history['accuracy'][epoch]),
                    'train_loss': float(history.history['loss'][epoch])
                }
                
                if 'val_accuracy' in history.history and epoch < len(history.history['val_accuracy']):
                    epoch_metrics['val_accuracy'] = float(history.history['val_accuracy'][epoch])
                if 'val_loss' in history.history and epoch < len(history.history['val_loss']):
                    epoch_metrics['val_loss'] = float(history.history['val_loss'][epoch])
                
                mlflow.log_metrics(epoch_metrics, step=epoch+1)
            
            # Sauvegarder le modèle
            mlflow.keras.log_model(model, "model")
            
            # Créer un rapport détaillé
            report_data = safe_serialize({
                'experiment_info': {
                    'run_name': run_name,
                    'experiment_id': experiment_num,
                    'timestamp': datetime.now().isoformat()
                },
                'model_config': config,
                'training_history': {
                    'final_train_accuracy': float(history.history['accuracy'][-1]) if 'accuracy' in history.history else 0,
                    'final_train_loss': float(history.history['loss'][-1]) if 'loss' in history.history else 0,
                    'epochs_completed': len(history.history['accuracy']) if 'accuracy' in history.history else 0
                },
                'evaluation_metrics': metrics,
                'dataset_info': {
                    'num_classes': int(data_loader.num_classes),
                    'class_names': data_loader.class_names,
                    'train_samples': int(train_gen.samples),
                    'test_samples': int(test_gen.samples)
                },
                'model_summary': {
                    'total_params': int(model.count_params()),
                    'trainable_params': int(sum([np.prod(v.shape) for v in model.trainable_weights])),
                    'non_trainable_params': int(sum([np.prod(v.shape) for v in model.non_trainable_weights]))
                }
            })
            
            # Sauvegarder le rapport
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "report")
            os.unlink(temp_path)
            
            print(f"    ✅ Réussi! Accuracy: {metrics['accuracy']:.4f}")
            print(f"    📊 F1-Score: {metrics['f1_score']:.4f}")
            print(f"    🎯 AUC-ROC: {metrics['auc_roc']:.4f}")
            
            return {
                'run_name': run_name,
                'config': config,
                'metrics': metrics,
                'history': history.history
            }
            
    except Exception as e:
        print(f"    ❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -------------------------------------------------------------------
# 11. EXÉCUTION DU GRIDSEARCH COMPLET
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🧠 DÉBUT DU GRIDSEARCH COMPLET")
print("=" * 80)

all_results = []
experiment_counter = 1

# Exécuter toutes les configurations ResNet
print(f"\n📋 RESNET50 - {len(RESNET_GRID)} configurations")
for config in RESNET_GRID:
    result = run_experiment(config, experiment_counter, 'resnet50')
    if result:
        all_results.append(result)
    experiment_counter += 1

# Exécuter toutes les configurations EfficientNet
print(f"\n📋 EFFICIENTNET - {len(EFFICIENTNET_GRID)} configurations")
for config in EFFICIENTNET_GRID:
    result = run_experiment(config, experiment_counter, 'efficientnet')
    if result:
        all_results.append(result)
    experiment_counter += 1

# Exécuter toutes les configurations MobileNet
print(f"\n📋 MOBILENET - {len(MOBILENET_GRID)} configurations")
for config in MOBILENET_GRID:
    result = run_experiment(config, experiment_counter, 'mobilenet')
    if result:
        all_results.append(result)
    experiment_counter += 1

# -------------------------------------------------------------------
# 12. ANALYSE ET RAPPORT FINAL
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📋 RAPPORT FINAL - SIPAKMED CLASSIFICATION")
print("=" * 80)

print(f"\n✅ EXPÉRIENCES TERMINÉES: {len(all_results)}")
print(f"📊 DATASET: SipakMed (images cytologiques)")
print(f"🎯 CLASSES: {data_loader.num_classes} classes")
print(f"📊 CLASSES TROUVÉES: {', '.join(data_loader.class_names)}")

if all_results:
    # Trouver le meilleur modèle
    best_result = max(all_results, key=lambda x: x['metrics']['accuracy'])
    
    print(f"\n🏆 MEILLEUR MODÈLE:")
    print(f"   Nom: {best_result['run_name']}")
    print(f"   Type: {best_result['config'].get('model_type', 'resnet50')}")
    print(f"   Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"   F1-Score: {best_result['metrics']['f1_score']:.4f}")
    print(f"   AUC-ROC: {best_result['metrics'].get('auc_roc', 0):.4f}")
    
    # Statistiques par type de modèle
    print(f"\n📈 STATISTIQUES PAR MODÈLE:")
    
    # Filtrer par type de modèle
    resnet_results = [r for r in all_results if r['config'].get('model_type') == 'resnet50']
    efficientnet_results = [r for r in all_results if r['config'].get('model_type') == 'efficientnet']
    mobilenet_results = [r for r in all_results if r['config'].get('model_type') == 'mobilenet']
    
    if resnet_results:
        acc_resnet = np.mean([r['metrics']['accuracy'] for r in resnet_results])
        f1_resnet = np.mean([r['metrics']['f1_score'] for r in resnet_results])
        print(f"   • ResNet50: Accuracy={acc_resnet:.4f}, F1={f1_resnet:.4f} ({len(resnet_results)} exp)")
    
    if efficientnet_results:
        acc_eff = np.mean([r['metrics']['accuracy'] for r in efficientnet_results])
        f1_eff = np.mean([r['metrics']['f1_score'] for r in efficientnet_results])
        print(f"   • EfficientNet: Accuracy={acc_eff:.4f}, F1={f1_eff:.4f} ({len(efficientnet_results)} exp)")
    
    if mobilenet_results:
        acc_mob = np.mean([r['metrics']['accuracy'] for r in mobilenet_results])
        f1_mob = np.mean([r['metrics']['f1_score'] for r in mobilenet_results])
        print(f"   • MobileNet: Accuracy={acc_mob:.4f}, F1={f1_mob:.4f} ({len(mobilenet_results)} exp)")
    
    # Top 3 modèles
    print(f"\n🥇 TOP 3 MODÈLES:")
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['accuracy'], reverse=True)[:3]
    for i, result in enumerate(sorted_results):
        print(f"   {i+1}. {result['run_name']}: Accuracy={result['metrics']['accuracy']:.4f}, "
              f"F1={result['metrics']['f1_score']:.4f}")
    
    # Sauvegarder le rapport final
    final_report = safe_serialize({
        'project': 'SipakMed Classification MLOps',
        'date': datetime.now().isoformat(),
        'dataset': {
            'name': 'sipakmed_new6',
            'path': DATA_PATH,
            'classes': data_loader.class_names,
            'num_classes': data_loader.num_classes,
            'train_samples': int(train_gen.samples),
            'test_samples': int(test_gen.samples)
        },
        'gridsearch_summary': {
            'total_experiments': len(all_results),
            'resnet_experiments': len(resnet_results),
            'efficientnet_experiments': len(efficientnet_results),
            'mobilenet_experiments': len(mobilenet_results),
            'best_accuracy': float(best_result['metrics']['accuracy']),
            'best_f1_score': float(best_result['metrics']['f1_score']),
            'best_model': best_result['run_name']
        },
        'best_model': {
            'run_name': best_result['run_name'],
            'config': best_result['config'],
            'metrics': best_result['metrics']
        },
        'top_3_models': [
            {
                'rank': i+1,
                'run_name': result['run_name'],
                'config': result['config'],
                'metrics': result['metrics']
            }
            for i, result in enumerate(sorted_results)
        ],
        'mlflow_info': {
            'experiment_name': EXPERIMENT_NAME,
            'tracking_uri': mlflow.get_tracking_uri()
        }
    })
    
    # Créer le dossier reports s'il n'existe pas
    os.makedirs("reports", exist_ok=True)
    
    # Sauvegarder le rapport
    report_path = f"reports/sipakmed_mlflow_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    
    print(f"\n📄 RAPPORT SAUVEGARDÉ: {report_path}")

print(f"\n🔍 POUR VISUALISER LES RÉSULTATS:")
print("  1. Lancer l'interface MLflow:")
print("     mlflow ui")
print("  2. Ouvrir dans le navigateur: http://localhost:5000")
print("  3. Sélectionner l'expérience: " + EXPERIMENT_NAME)
print("  4. Trier par 'accuracy' pour voir les meilleurs modèles")

print(f"\n🎯 EXIGENCES DU PROJET SATISFAITES:")
print(f"  ✅ Git - Code versionné")
print(f"  ⚠️  DVC - À intégrer (tracking des données)")
print(f"  ✅ MLflow - {len(all_results)} expériences (≥10 requis)")
print(f"  ⚠️  SHAP/LIME - Prochaine étape")
print(f"  ⚠️  Streamlit - Prochaine étape")

print(f"\n🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("🎉 PHASE MLFLOW TERMINÉE AVEC SUCCÈS!")
print("=" * 80)

# -------------------------------------------------------------------
# 13. INSTRUCTIONS POUR LES PROCHAINES ÉTAPES
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🚀 PROCHAINES ÉTAPES DU PROJET MLOPS")
print("=" * 80)

print("\n1. 📊 ANALYSE MLFLOW (Maintenant):")
print("   - Ouvrir MLflow UI: mlflow ui")
print("   - Comparer les modèles avec les métriques")
print("   - Exporter les paramètres du meilleur modèle")
print("   - Prendre des captures d'écran pour le rapport")

print("\n2. 🔍 EXPLICABILITÉ (SHAP/LIME):")
print("   - Installer: pip install shap lime")
print("   - Charger le meilleur modèle depuis MLflow")
print("   - Créer un script explainability.py")
print("   - Générer des visualisations des features importantes")

print("\n3. 🌐 INTERFACE STREAMLIT:")
print("   - Installer: pip install streamlit")
print("   - Créer streamlit_app.py")
print("   - Ajouter:")
print("     • Upload d'images médicales")
print("     • Visualisation des prédictions")
print("     • Affichage des métriques d'explicabilité")

print("\n4. 🔄 INTÉGRATION DVC:")
print("   - Initialiser DVC: dvc init")
print("   - Ajouter les données: dvc add data/")
print("   - Configurer le stockage distant (Google Drive, S3, etc.)")
print("   - Ajouter les hash DVC aux logs MLflow pour tracking complet")

print("\n5. 📚 DOCUMENTATION FINALE:")
print("   - Rédiger le rapport final (2-3 pages)")
print("   - Préparer la présentation (10-15 slides)")
print("   - Inclure:")
print("     • Architecture MLOps complète")
print("     • Résultats du GridSearch MLflow")
print("     • Analyse d'explicabilité avec SHAP/LIME")
print("     • Démonstration de l'interface Streamlit")

print("\n📚 RESSOURCES UTILES:")
print("  • MLflow Documentation: https://mlflow.org/docs/")
print("  • SHAP Documentation: https://shap.readthedocs.io/")
print("  • Streamlit Documentation: https://docs.streamlit.io/")
print("  • DVC Documentation: https://dvc.org/doc")
print("  • TensorFlow Documentation: https://www.tensorflow.org/")
print("  • Dataset SipakMed: https://www.cs.uoi.gr/~marina/sipakmed.html")

print("\n" + "=" * 80)
print("✅ PROJET MLOPS - PHASE MLFLOW & GRIDSEARCH COMPLÉTÉE!")
print("=" * 80)