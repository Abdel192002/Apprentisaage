"""
SHAP Explainer - Version pour structure explainability/
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import sys
from PIL import Image

# Ajouter le chemin pour pouvoir importer vos modèles
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
models_dir = os.path.join(src_dir, 'models')
sys.path.append(models_dir)

class CNN_SHAP_Explainer:
    """Classe SHAP Explainer"""
    
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or [f"Class_{i}" for i in range(model.output_shape[1])]
        self.explainer = None
    
    def build_explainer(self, background_images):
        """Construit l'explicateur SHAP"""
        print("🧠 Construction SHAP...")
        self.explainer = shap.DeepExplainer(self.model, background_images)
        return self.explainer
    
    def explain(self, image):
        """Explique une image"""
        if self.explainer is None:
            raise ValueError("Appelez build_explainer() d'abord")
        
        # Format batch
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, 0)
        else:
            image_batch = image
        
        # Calcul SHAP
        shap_values = self.explainer.shap_values(image_batch)
        
        # Prédiction
        preds = self.model.predict(image_batch, verbose=0)[0]
        pred_class = np.argmax(preds)
        
        return {
            'shap_values': shap_values,
            'predicted_class': pred_class,
            'class_name': self.class_names[pred_class],
            'confidence': preds[pred_class],
            'image': image
        }
    
    def visualize(self, explanation):
        """Visualise l'explication"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Image originale
        axes[0].imshow(explanation['image'])
        axes[0].set_title(f"Original\n{explanation['class_name']}")
        axes[0].axis('off')
        
        # Heatmap SHAP
        shap_vals = explanation['shap_values']
        if isinstance(shap_vals, list):
            heatmap = np.abs(shap_vals[explanation['predicted_class']][0]).mean(axis=-1)
        else:
            heatmap = np.abs(shap_vals[0]).mean(axis=-1)
        
        axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title("Importance SHAP")
        axes[1].axis('off')
        
        plt.suptitle(f"Confiance: {explanation['confidence']:.1%}")
        plt.tight_layout()
        plt.show()

# Test direct
if __name__ == "__main__":
    print("Test SHAP Explainer")
    
    # Créer un modèle de test
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    
    # Données de test
    test_images = np.random.randn(5, 224, 224, 3)
    test_images = (test_images - test_images.min()) / (test_images.max() - test_images.min())
    
    # Test
    explainer = CNN_SHAP_Explainer(model, ['A', 'B', 'C'])
    explainer.build_explainer(test_images[:3])
    result = explainer.explain(test_images[0])
    explainer.visualize(result)