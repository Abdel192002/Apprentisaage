import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class GradCAM:
    """
    Implémentation Grad-CAM pour les modèles TensorFlow/Keras
    """
    
    def __init__(self, model, layer_name: Optional[str] = None):
        """
        Initialiser Grad-CAM avec un modèle
        
        Args:
            model: Modèle Keras
            layer_name: Nom de la couche convolutionnelle (si None, dernière couche conv)
        """
        self.model = model
        
        # Trouver la dernière couche convolutionnelle si non spécifiée
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        """Construire le modèle pour Grad-CAM"""
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        return grad_model
    
    def compute_heatmap(self, 
                       image: np.ndarray, 
                       class_idx: Optional[int] = None,
                       eps: float = 1e-8) -> np.ndarray:
        """
        Calculer la heatmap Grad-CAM
        
        Args:
            image: Image d'entrée (batch de 1)
            class_idx: Indice de classe (si None, classe prédite)
            eps: Petite valeur pour éviter la division par zéro
            
        Returns:
            Heatmap normalisée
        """
        # Vérifications
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Forward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        # Gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Moyenne globale des gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination of activation maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), 
            axis=-1
        )
        
        # Normalisation
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, 
                       image: np.ndarray, 
                       heatmap: np.ndarray, 
                       alpha: float = 0.5,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Superposer la heatmap sur l'image originale
        
        Args:
            image: Image originale (H, W, C)
            heatmap: Heatmap Grad-CAM (H, W)
            alpha: Transparence
            colormap: Colormap OpenCV
            
        Returns:
            Image avec heatmap superposée
        """
        # Redimensionner la heatmap
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Appliquer colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        # Convertir image si nécessaire
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Superposition
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def explain(self, 
               image_path: str, 
               target_size: Tuple[int, int] = (224, 224),
               save_path: Optional[str] = None) -> dict:
        """
        Explication complète pour une image
        
        Args:
            image_path: Chemin vers l'image
            target_size: Taille de redimensionnement
            save_path: Chemin pour sauvegarder les résultats
            
        Returns:
            Dictionnaire avec résultats
        """
        # Charger et prétraiter l'image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        
        # Normalisation pour le modèle
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # Prédiction
        predictions = self.model.predict(img_input, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Calcul Grad-CAM
        heatmap = self.compute_heatmap(img_input, predicted_class)
        
        # Superposition
        overlayed = self.overlay_heatmap(img_resized, heatmap)
        
        # Préparer les résultats
        results = {
            'original_image': img_resized,
            'heatmap': heatmap,
            'overlayed_image': overlayed,
            'predictions': predictions[0],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'layer_name': self.layer_name
        }
        
        # Sauvegarde si demandé
        if save_path:
            self._save_results(results, save_path)
        
        return results
    
    def _save_results(self, results: dict, save_path: str):
        """Sauvegarder les résultats de l'explication"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image originale
        axes[0].imshow(results['original_image'])
        axes[0].set_title(f"Original\nClasse prédite: {results['predicted_class']}\nConfiance: {results['confidence']:.3f}")
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(results['heatmap'], cmap='jet')
        axes[1].set_title("Heatmap Grad-CAM")
        axes[1].axis('off')
        
        # Superposition
        axes[2].imshow(results['overlayed_image'])
        axes[2].set_title("Superposition")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Résultats sauvegardés: {save_path}")


# Fonction utilitaire pour créer des explainers pour différents modèles
def create_gradcam_for_model(model, model_name: str):
    """
    Créer un explainer Grad-CAM adapté au modèle
    
    Args:
        model: Modèle Keras
        model_name: Nom du modèle
        
    Returns:
        Instance GradCAM
    """
    # Mapping des noms de couches par modèle
    layer_mapping = {
        'resnet': 'conv5_block3_out',
        'mobilenet': 'out_relu',
        'efficientnet': 'top_activation'
    }
    
    # Trouver la couche appropriée
    layer_name = None
    for key in layer_mapping:
        if key in model_name.lower():
            layer_name = layer_mapping[key]
            break
    
    # Parcourir les couches si non trouvé
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() or 'activation' in layer.name.lower():
                layer_name = layer.name
                break
    
    print(f"Utilisation de la couche pour Grad-CAM: {layer_name}")
    return GradCAM(model, layer_name)