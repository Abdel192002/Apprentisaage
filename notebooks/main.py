import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from DataLoaders import DataLoader

# Import des modèles
from Model import ResNet50_Model
from eficcient import EfficientNetB0_Model
from Mobile import MobileNetV2_Model

import tensorflow as tf

print(f"Keras version: {tf.keras.__version__}")

# =======================
# CHARGEMENT DES DONNÉES
# =======================

data_loader = DataLoader(path="C:/Users/nessa/Downloads/sipakmed_new6/")
datasets = data_loader.get_datasets()

train_with_aug = datasets["train_with_augmentation"]
test_set = datasets["test"]

# =======================
# CALLBACKS
# =======================

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
]

# =======================
# FONCTION D'ENTRAÎNEMENT
# =======================

def train_and_evaluate(model_class, model_name):

    print("\n==============================")
    print(f"  Entraînement du modèle : {model_name}")
    print("==============================\n")

    model_obj = model_class(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4)
    model = model_obj.get_model()

    # ---- PHASE 1 : Entraînement initial ----
    history = model.fit(
        train_with_aug,
        epochs=30,
        validation_data=test_set,
        callbacks=callbacks,
        verbose=2
    )

    # ---- FINE TUNING ----
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tuning_history = model.fit(
        train_with_aug,
        epochs=50,
        validation_data=test_set,
        callbacks=callbacks,
        verbose=2
    )

    # ==== EVALUATION ====
    print(f"\nÉvaluation du modèle {model_name} :")
    test_loss, test_accuracy = model.evaluate(test_set, verbose=2)
    print(f"> Accuracy : {test_accuracy * 100:.2f}%")

    # ==== PRÉDICTIONS ====
    y_true = np.concatenate([y for _, y in test_set], axis=0)
    y_pred = model.predict(test_set)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # ==== AUC ====
    auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    print(f"> AUC : {auc:.4f}")

    # ==== RAPPORT ====
    print("\nRapport de classification :")
    print(classification_report(y_true_classes, y_pred_classes, target_names=test_set.class_names))

    # ==== MATRICE ====
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_true_classes, y_pred_classes))

    # Retourner les résultats
    return {
        "model": model_name,
        "accuracy": test_accuracy,
        "auc": auc
    }

# ====================================
# ENTRAÎNEMENT DES 3 MODÈLES
# ====================================

results = []

results.append(train_and_evaluate(ResNet50_Model, "ResNet50"))
results.append(train_and_evaluate(EfficientNetB0_Model, "EfficientNetB0"))
results.append(train_and_evaluate(MobileNetV2_Model, "MobileNetV2"))

# ====================================
# TABLEAU FINAL DES PERFORMANCES
# ====================================

print("\n\n==============================")
print("     TABLEAU FINAL DES SCORES")
print("==============================\n")

for r in results:
    print(f"{r['model']}: Accuracy = {r['accuracy']:.4f} | AUC = {r['auc']:.4f}")
