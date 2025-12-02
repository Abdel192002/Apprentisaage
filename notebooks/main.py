import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from DataLoaders import DataLoader

from Model import ResNet50_Model
from eficcient import EfficientNetB0_Model
from Mobile import MobileNetV2_Model

import tensorflow as tf
import tensorflow.keras.backend as K
import gc

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

    model_obj = model_class(
        input_shape=(224, 224, 3),
        num_classes=3,
        learning_rate=1e-4
    )
    model = model_obj.get_model()

    # ---- PHASE 1 ----
    model.fit(
        train_with_aug,
        epochs=45,
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

    model.fit(
        train_with_aug,
        epochs=10,
        validation_data=test_set,
        callbacks=callbacks,
        verbose=2
    )

    # ================ ÉVALUATION ================
    print(f"\nÉvaluation du modèle {model_name} :")
    test_loss, test_accuracy = model.evaluate(test_set, verbose=2)
    print(f"> Accuracy : {test_accuracy * 100:.2f}%")

    # ==== EXTRACTION LABELS ====
    y_true = []
    for i in range(len(test_set)):
        _, label = test_set[i]   # (image, one-hot)
        y_true.append(label[0])

    y_true = np.array(y_true)
    y_true_classes = np.argmax(y_true, axis=1)

    # ==== PRÉDICTIONS ====
    y_pred = model.predict(test_set)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # ==== AUC ====
    auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    print(f"> AUC : {auc:.4f}")

    # ==== RAPPORT ====
    print("\nRapport de classification :")
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=data_loader.class_names
    ))

    # ==== MATRICE ====
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_true_classes, y_pred_classes))

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
K.clear_session()
gc.collect()

results.append(train_and_evaluate(EfficientNetB0_Model, "EfficientNetB0"))
K.clear_session()
gc.collect()

results.append(train_and_evaluate(MobileNetV2_Model, "MobileNetV2"))
K.clear_session()
gc.collect()

# ====================================
# TABLEAU FINAL
# ====================================

print("\n\n==============================")
print("     TABLEAU FINAL DES SCORES")
print("==============================\n")

for r in results:
    print(f"{r['model']}: Accuracy = {r['accuracy']:.4f} | AUC = {r['auc']:.4f}")
