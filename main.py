import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from Model import CNN_ViT
from DataLoaders import DataLoader  # Assurez-vous que ce module existe
import tensorflow as tf

# Vérifiez la version de Keras
print(f"Keras version: {tf.keras.__version__}")

# Initialiser le DataLoader avec le chemin des données
data_loader = DataLoader(path="C:/Users/nessa/Downloads/sipakmed_new6/")

# Charger les ensembles de données
datasets = data_loader.get_datasets()

# Accéder aux ensembles d'entraînement et de test
train_with_aug = datasets["train_with_augmentation"]
test_set = datasets["test"]

# Construire le modèle ResNet50
resnet_model = CNN_ViT(input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4)
model = resnet_model.get_model()

# Définir les callbacks pour ajuster l'apprentissage et enregistrer les meilleurs résultats
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Phase 1 : Entraînement initial (poids gelés)
history = model.fit(
    train_with_aug,
    epochs=300,  # Respect des paramètres de l'article
    validation_data=test_set,
    callbacks=callbacks,
    verbose=2
)

# Débloquer les couches pour fine-tuning
for layer in model.layers:
    layer.trainable = True

# Recompiler le modèle avec un taux d'apprentissage plus faible
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Phase 2 : Fine-tuning
fine_tuning_history = model.fit(
    train_with_aug,
    epochs=50,  # Phase de fine-tuning plus courte
    validation_data=test_set,
    callbacks=callbacks,
    verbose=2
)

# Évaluation sur l'ensemble de test
print("\nÉvaluation sur l'ensemble de test :")
test_loss, test_accuracy = model.evaluate(test_set, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Prédictions sur l'ensemble de test
print("\nPrédictions sur l'ensemble de test :")
y_true = np.concatenate([y for _, y in test_set], axis=0)
y_pred = model.predict(test_set)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_true_classes, y_pred_classes, target_names=test_set.class_names))

# Matrice de confusion
print("\nMatrice de confusion :")
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print(conf_matrix)
