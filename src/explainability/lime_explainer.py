"""
LIME Explainer pour vos modèles CNN - Version sans OpenCV
Emplacement: APPRENTISAAGE-2/src/models/lime_explainer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class CNN_LIME_Explainer:
    """Explique les prédictions avec LIME"""
    
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or [f"Classe_{i}" for i in range(model.output_shape[1])]
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_proba(self, images):
        """Fonction de prédiction pour LIME"""
        if len(images.shape) == 4:
            return self.model.predict(images, verbose=0)
        else:
            return self.model.predict(np.expand_dims(images, 0), verbose=0)[0]
    
    def explain(self, image, top_labels=3, num_samples=500):
        """Explique une image avec LIME"""
        # Normaliser si nécessaire
        if image.max() > 1.0:
            image = image / 255.0
        
        print("🍋 Explication LIME en cours...")
        
        explanation = self.explainer.explain_instance(
            image=image,
            classifier_fn=self.predict_proba,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Prédiction
        preds = self.predict_proba(image)
        pred_class = np.argmax(preds)
        
        return {
            'explanation': explanation,
            'predicted_class': pred_class,
            'predicted_class_name': self.class_names[pred_class],
            'confidence': preds[pred_class],
            'image': image
        }
    
    def visualize(self, explanation, save_path=None):
        """Visualise l'explication LIME"""
        exp_obj = explanation['explanation']
        image = explanation['image']
        pred_class = explanation['predicted_class']
        
        # Obtenir le masque
        temp, mask = exp_obj.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        # Créer la visualisation
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Image originale
        ax1.imshow(image)
        ax1.set_title(f"Image originale\n{explanation['predicted_class_name']}")
        ax1.axis('off')
        
        # 2. Masque LIME
        ax2.imshow(mark_boundaries(image, mask))
        ax2.set_title("Régions importantes")
        ax2.axis('off')
        
        # 3. Superposition
        from skimage.segmentation import slic
        try:
            segments = slic(image, n_segments=50, compactness=10, sigma=1)
            ax3.imshow(mark_boundaries(image, segments))
            ax3.set_title("Segmentation")
        except:
            ax3.imshow(image)
            ax3.set_title("Image (segmentation non disponible)")
        ax3.axis('off')
        
        plt.suptitle(f"Explication LIME - Confiance: {explanation['confidence']:.1%}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Sauvegardé: {save_path}")
        
        plt.show()
        return fig
    
    def generate_report(self, images, n_samples=5, save_dir="lime_reports"):
        """Génère un rapport LIME"""
        import os
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(save_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"📋 Génération du rapport LIME...")
        
        for i in range(min(n_samples, len(images))):
            print(f"  Traitement image {i+1}/{n_samples}...")
            
            # Expliquer
            explanation = self.explain(images[i])
            
            # Sauvegarder
            img_path = os.path.join(report_dir, f"lime_image_{i+1}.png")
            self.visualize(explanation, save_path=img_path)
            
            # Sauvegarder les prédictions
            preds = self.predict_proba(images[i])
            pred_path = os.path.join(report_dir, f"predictions_{i+1}.npy")
            np.save(pred_path, preds)
        
        # Rapport texte
        report_path = os.path.join(report_dir, "rapport_lime.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT D'EXPLICABILITÉ LIME\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Images analysées: {n_samples}\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n\n")
            
            f.write("Fichiers générés:\n")
            for i in range(min(n_samples, len(images))):
                f.write(f"  - lime_image_{i+1}.png (visualisation)\n")
                f.write(f"  - predictions_{i+1}.npy (probabilités)\n")
        
        print(f"✅ Rapport LIME généré dans: {report_dir}")
        return report_dir

# Fonctions utilitaires
def create_sample_images(n_samples=10, img_size=(224, 224, 3)):
    """Crée des images factices"""
    images = np.random.randn(n_samples, *img_size).astype(np.float32)
    images = (images - images.min()) / (images.max() - images.min())
    return images

def test_lime_with_your_model():
    """Teste LIME avec vos modèles"""
    print("=" * 60)
    print("TEST LIME AVEC VOS MODÈLES (SANS OPENCV)")
    print("=" * 60)
    
    # 1. Charger VOTRE modèle
    try:
        from resnet_model import ResNet50_Model
        
        model_obj = ResNet50_Model()
        model = model_obj.get_model()
        print(f"✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Création d'un modèle de démonstration...")
        
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base.output
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=outputs)
        print("✅ Modèle de démonstration créé")
    
    # 2. Créer des données
    test_images = create_sample_images(n_samples=8)
    print(f"✅ Données de test: {test_images.shape}")
    
    # 3. Créer l'explicateur
    explainer = CNN_LIME_Explainer(model, ['Animal', 'Véhicule', 'Personne'])
    
    # 4. Tester
    print("\n🔍 Analyse des images...")
    for i in range(3):
        explanation = explainer.explain(test_images[i])
        print(f"  Image {i+1}: {explanation['predicted_class_name']} ({explanation['confidence']:.1%})")
        
        if i == 0:
            explainer.visualize(explanation)
    
    # 5. Rapport
    print("\n📋 Génération du rapport...")
    report_dir = explainer.generate_report(test_images, n_samples=3)
    
    print(f"\n✅ Test LIME réussi!")
    print(f"📁 Rapport dans: {report_dir}")

if __name__ == "__main__":
    test_lime_with_your_model()