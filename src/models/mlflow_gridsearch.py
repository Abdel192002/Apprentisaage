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
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "..", "data")
sys.path.append(data_path)

# Get the correct paths
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

print(project_root)

DATA_PATH = os.path.join(project_root, "data", "processed")  + "/"

# -------------------------------------------------------------------
# 2. IMPORTS SPÉCIFIQUES IMAGES
# -------------------------------------------------------------------
print("\nIMPORTS POUR TRAITEMENT D'IMAGES...")

import tensorflow as tf
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print(f"TensorFlow {tf.__version__}")
print(f"MLflow {mlflow.__version__}")

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

try:
    from data_loading import DataLoader
    print("DataLoader importé depuis data_loading.py")
except ImportError as e:
    print(f"Erreur d'import DataLoader: {e}")

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
# 4. CHARGEMENT DES DONNÉES AVEC DVC
# -------------------------------------------------------------------
print("\n" + "="*60)
print("CHARGEMENT DU DATASET SIPAKMED")
print("="*60)


loader = DataLoader(
        path=DATA_PATH,
        augment=True,
        batch_size=32,
        target_size=(224, 224)
    )
    
train_gen, test_gen = loader.get_generators()
    
summary = loader.get_summary()
    
print(f"\nDonnées chargées !")
print(f"Résumé:")
print(f"   Classes: {summary['classes']}")
print(f"   Nombre de classes: {summary['num_classes']}")
print(f"   Images d'entraînement: {summary['train_samples']}")
print(f"   Images de test: {summary['test_samples']}")
print(f"   Taille des batches: {summary['batch_size']}")
print(f"   Taille des images: {summary['target_size']}")
    
print(f"\nDétail par classe:")
counts = summary['counts']
for cls in summary['classes']:
    train_count = counts['train'].get(cls, 0)
    test_count = counts['test'].get(cls, 0)
    print(f"   {cls}: {train_count} train, {test_count} test")

# -------------------------------------------------------------------
# 5. IMPORT DES MODÈLES AVEC GRIDSEARCH INTÉGRÉ
# -------------------------------------------------------------------
print("\nIMPORT DES MODÈLES AVEC GRIDSEARCH...")

from efficient_model import EfficientNetB0_Model
from resnet_model import ResNet50_Model
from mobilenet_model import MobileNetV2_Model

# -------------------------------------------------------------------
# 6. FONCTIONS DE CRÉATION DE MODÈLES
# -------------------------------------------------------------------
def create_resnet50_model(num_classes, config):
    print(f"Création ResNet50 avec config: {config}")
    
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
    print(f"Création EfficientNet avec config: {config}")
    
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
    print(f"Création MobileNet avec config: {config}")
    
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
    print("Evaluation sur le test set...")
    
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
                                   target_names=loader.class_names,
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
    for i, class_name in enumerate(loader.class_names):
        if class_name in report:
            metrics[f'precision_{class_name}'] = float(report[class_name]['precision'])
            metrics[f'recall_{class_name}'] = float(report[class_name]['recall'])
            metrics[f'f1_{class_name}'] = float(report[class_name]['f1-score'])
            metrics[f'support_{class_name}'] = int(report[class_name]['support'])
    
    return metrics, predictions, report

def evaluate_model_safely(model, test_generator):
    """Évaluation sécurisée qui gère les multiples métriques"""
    print("Évaluation finale du modèle...")

    # Récupérer toutes les valeurs de model.evaluate()
    evaluation_results = model.evaluate(test_generator, verbose=0, return_dict=True)
        
    if isinstance(evaluation_results, dict):
        test_loss = evaluation_results.get('loss', 0)
        test_accuracy = evaluation_results.get('accuracy', 0)
    else:
        test_loss = evaluation_results[0] if len(evaluation_results) > 0 else 0
        test_accuracy = evaluation_results[1] if len(evaluation_results) > 1 else 0
            
    return float(test_loss), float(test_accuracy)


# -------------------------------------------------------------------
# 8. CONFIGURATION MLFLOW - TOUT DANS MLRUNS
# -------------------------------------------------------------------
print("\nCONFIGURATION MLFLOW...")

# SEULEMENT mlruns, PAS de reports
mlrun_dir = os.path.join(project_root, "mlruns")

mlflow.set_tracking_uri(f"file:{mlrun_dir}")

EXPERIMENT_NAME = f"SipakMed_Classification_{datetime.now().strftime('%Y%m%d_%H%M')}"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Expérience MLflow: {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Add this after mlflow setup
print(f"\nDEBUG - MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"DEBUG - MLflow artifact location: {mlflow.get_artifact_uri()}")

# -------------------------------------------------------------------
# 9. CONFIGURATION DU GRIDSEARCH
# -------------------------------------------------------------------
print("\nCONFIGURATION DU GRIDSEARCH COMPLET...")

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

print(f"Total des configurations de GridSearch:")
print(f"   - ResNet50: {len(RESNET_GRID)} configurations")
print(f"   - EfficientNet: {len(EFFICIENTNET_GRID)} configurations")
print(f"   - MobileNet: {len(MOBILENET_GRID)} configurations")
print(f"   - TOTAL: {len(RESNET_GRID) + len(EFFICIENTNET_GRID) + len(MOBILENET_GRID)} expériences")

# -------------------------------------------------------------------
# 10. FONCTION GÉNÉRIQUE POUR EXÉCUTER LES EXPÉRIENCES
# -------------------------------------------------------------------
def run_experiment(config, experiment_num, model_type):
    """Exécute une expérience MLflow avec configuration donnée"""
    
    run_name = f"{model_type}_exp_{experiment_num:02d}"
    
    print(f"\n{'='*60}")
    print(f"  Expérience {experiment_num}: {run_name}")
    print(f"     Configuration: {config}")
    print(f"{'='*60}")
    

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
                'num_classes': loader.num_classes,
                'dataset': 'sipakmed_new6',
                'image_size': '224x224'
            })
            
            # Logger les infos du dataset
        mlflow.log_params({
                'class_names': str(loader.class_names),
                'train_samples': int(train_gen.samples),
                'test_samples': int(test_gen.samples)
            })
            
            # Sélectionner le bon modèle
        if model_type == 'resnet50':
            model = create_resnet50_model(loader.num_classes, config)
        elif model_type == 'efficientnet':
            model = create_efficientnet_model(loader.num_classes, config)
        elif model_type == 'mobilenet':
            model = create_mobilenet_model(loader.num_classes, config)
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
            
            # Callbacks
        callbacks_list = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
            ]
            
        # Entraînement
        print(f"    Entraînement ({config['epochs']} epochs)...")
            
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
            
        # Evaluation
        test_loss, test_accuracy = evaluate_model_safely(model, test_gen)
        metrics, _, _ = calculate_comprehensive_metrics(model, test_gen)
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
                    'num_classes': int(loader.num_classes),
                    'class_names': loader.class_names,
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
            
        print(f" Réussi! Accuracy: {metrics['accuracy']:.4f}")
        print(f" F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            
        return {
                'run_name': run_name,
                'config': config,
                'metrics': metrics,
                'history': history.history
            }

# -------------------------------------------------------------------
# 11. EXÉCUTION DU GRIDSEARCH COMPLET
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("DÉBUT DU GRIDSEARCH COMPLET")
print("=" * 80)

# Cleanup any active runs before starting
if mlflow.active_run() is not None:
    print(f"Nettoyage d'une run active: {mlflow.active_run().info.run_id}")
    mlflow.end_run()

all_results = []
experiment_counter = 1

# Exécuter toutes les configurations ResNet
print(f"\nRESNET50 - {len(RESNET_GRID)} configurations")
for config in RESNET_GRID:
    result = run_experiment(config, experiment_counter, 'resnet50')
    if result:
        all_results.append(result)
    experiment_counter += 1

# Exécuter toutes les configurations EfficientNet
print(f"\nEFFICIENTNET - {len(EFFICIENTNET_GRID)} configurations")
for config in EFFICIENTNET_GRID:
    result = run_experiment(config, experiment_counter, 'efficientnet')
    if result:
        all_results.append(result)
    experiment_counter += 1

# Exécuter toutes les configurations MobileNet
print(f"\nMOBILENET - {len(MOBILENET_GRID)} configurations")
for config in MOBILENET_GRID:
    result = run_experiment(config, experiment_counter, 'mobilenet')
    if result:
        all_results.append(result)
    experiment_counter += 1

# -------------------------------------------------------------------
# 12. ANALYSE ET RAPPORT FINAL
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("RAPPORT FINAL - SIPAKMED CLASSIFICATION")
print("=" * 80)

print(f"\nEXPÉRIENCES TERMINÉES: {len(all_results)}")
print(f"DATASET: SipakMed (images cytologiques)")
print(f"CLASSES: {loader.num_classes} classes")
print(f"CLASSES TROUVÉES: {', '.join(loader.class_names)}")

if all_results:
    # Trouver le meilleur modèle
    best_result = max(all_results, key=lambda x: x['metrics']['accuracy'])
    
    print(f"\nMEILLEUR MODÈLE:")
    print(f"   Nom: {best_result['run_name']}")
    print(f"   Type: {best_result['config'].get('model_type', 'resnet50')}")
    print(f"   Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"   F1-Score: {best_result['metrics']['f1_score']:.4f}")
    print(f"   AUC-ROC: {best_result['metrics'].get('auc_roc', 0):.4f}")
    
    # Statistiques par type de modèle
    print(f"\nSTATISTIQUES PAR MODÈLE:")
    
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
    print(f"\nTOP 3 MODÈLES:")
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
            'classes': loader.class_names,
            'num_classes': loader.num_classes,
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
    
    # MODIFIEZ CETTE PARTIE - Assurez-vous que le chemin est correct
    mlflow_report_dir = os.path.join(mlrun_dir, "mlflow_reports")
    os.makedirs(mlflow_report_dir, exist_ok=True)  # Créer le dossier si nécessaire

    # SAUVEGARDER DANS MLRUNS, PAS REPORTS!
    report_filename = f"sipakmed_mlflow_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    report_path = os.path.join(mlrun_dir, report_filename)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\nRAPPORT SAUVEGARDÉ DANS MLRUNS: {report_path}")

reports_dir = os.path.join(project_root, "reports")
if os.path.exists(reports_dir) and os.path.isdir(reports_dir):
    print(f"\nATTENTION: Dossier 'reports' trouvé à la racine: {reports_dir}")
    print("   Ce dossier a été créé par MLflow ou une autre librairie.")
    print("   Voulez-vous le supprimer? (y/n)")

print("\n" + "="*80)
print("LANCEMENT DE MLflow UI")
print("="*80)

print("\nPour voir les expériences avec l'UI MLFlow, taper la commande \"./run_mlflow.bat\" dans un terminal puis cliquer sur \"http://127.0.0.1:5000\" ")
print("\nSinon, cliquer sur le lien suivant :")
print(f"\nURL: http://localhost:5000")
print(f"Expérience: {EXPERIMENT_NAME}")