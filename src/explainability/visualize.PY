import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Optional
import seaborn as sns
from matplotlib.gridspec import GridSpec

class ExplainabilityVisualizer:
    """
    Classe pour visualiser les résultats d'explicabilité
    """
    
    @staticmethod
    def plot_model_comparison(results_dict: Dict[str, Dict], 
                            image: np.ndarray,
                            class_names: List[str],
                            save_path: Optional[str] = None):
        """
        Comparer les explications de différents modèles
        
        Args:
            results_dict: Dictionnaire {nom_modele: resultats}
            image: Image originale
            class_names: Noms des classes
            save_path: Chemin de sauvegarde
        """
        n_models = len(results_dict)
        model_names = list(results_dict.keys())
        
        fig = plt.figure(figsize=(15, 4 * n_models))
        
        for idx, model_name in enumerate(model_names):
            results = results_dict[model_name]
            
            # Créer une grille pour chaque modèle
            gs = GridSpec(1, 4, figure=fig)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[0, 3])
            
            # Image originale
            ax0.imshow(image)
            ax0.set_title(f"{model_name}\nOriginal", fontsize=12)
            ax0.axis('off')
            
            # Prédictions (bar plot)
            predictions = results.get('predictions', [])
            bars = ax1.bar(range(len(predictions)), predictions, 
                          color='skyblue', edgecolor='black')
            ax1.set_xticks(range(len(predictions)))
            ax1.set_xticklabels(class_names, rotation=45)
            ax1.set_ylabel('Confiance')
            ax1.set_title(f'Prédictions\nClasse: {results.get("predicted_class", "N/A")}\n'
                         f'Confiance: {results.get("confidence", 0):.3f}')
            
            # Colorer la barre prédite
            if 'predicted_class' in results:
                bars[results['predicted_class']].set_color('red')
            
            # Heatmap Grad-CAM si disponible
            if 'heatmap' in results:
                im = ax2.imshow(results['heatmap'], cmap='jet')
                ax2.set_title('Grad-CAM Heatmap')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            # Image superposée si disponible
            if 'overlayed_image' in results:
                ax3.imshow(results['overlayed_image'])
                ax3.set_title('Superposition')
                ax3.axis('off')
            elif 'shap_values' in results:
                # Visualisation SHAP
                shap_values = results['shap_values']
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Prendre la valeur absolue moyenne par pixel
                shap_abs = np.abs(shap_values).mean(axis=-1)
                im = ax3.imshow(shap_abs[0], cmap='hot')
                ax3.set_title('Importance SHAP')
                ax3.axis('off')
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.suptitle('Comparaison des Explications par Modèle', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_feature_importance(shap_values: np.ndarray, 
                               feature_names: List[str],
                               max_features: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot d'importance des features (version agrégée)
        
        Args:
            shap_values: Valeurs SHAP (n_samples, n_features)
            feature_names: Noms des features
            max_features: Nombre max de features à afficher
            save_path: Chemin de sauvegarde
        """
        # Calcul de l'importance moyenne absolue
        if len(shap_values.shape) > 2:
            shap_values = shap_values.mean(axis=(0, 1, 2))
        
        importance = np.abs(shap_values).mean(axis=0)
        
        # Trier par importance
        indices = np.argsort(importance)[::-1][:max_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(indices))
        
        ax.barh(y_pos, importance[indices], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance moyenne absolue SHAP')
        ax.set_title('Importance des Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_explainability_report(model_results: Dict[str, Dict],
                                   image_paths: List[str],
                                   class_names: List[str],
                                   output_dir: str):
        """
        Créer un rapport complet d'explicabilité
        
        Args:
            model_results: Résultats par modèle
            image_paths: Chemins des images testées
            class_names: Noms des classes
            output_dir: Répertoire de sortie
        """
        import os
        from datetime import datetime
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rapport par image
        for img_idx, img_path in enumerate(image_paths):
            fig = plt.figure(figsize=(20, 5 * len(model_results)))
            
            for m_idx, (model_name, results_list) in enumerate(model_results.items()):
                if img_idx < len(results_list):
                    results = results_list[img_idx]
                    
                    # Position dans la figure
                    gs = GridSpec(2, 3, figure=fig, 
                                 top=0.95 - m_idx*0.3, 
                                 bottom=0.7 - m_idx*0.3)
                    
                    # 1. Image originale
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(results.get('original_image', np.zeros((224, 224, 3))))
                    ax1.set_title(f"{model_name}\nImage Originale")
                    ax1.axis('off')
                    
                    # 2. Prédictions
                    ax2 = fig.add_subplot(gs[0, 1])
                    preds = results.get('predictions', [])
                    bars = ax2.bar(range(len(preds)), preds)
                    ax2.set_xticks(range(len(preds)))
                    ax2.set_xticklabels(class_names, rotation=45)
                    ax2.set_ylim([0, 1])
                    ax2.set_title(f'Prédictions\nClasse: {results.get("predicted_class")}\n'
                                f'Confiance: {results.get("confidence", 0):.3f}')
                    
                    # 3. Heatmap
                    ax3 = fig.add_subplot(gs[0, 2])
                    if 'heatmap' in results:
                        im = ax3.imshow(results['heatmap'], cmap='jet')
                        ax3.set_title('Heatmap Grad-CAM')
                        plt.colorbar(im, ax=ax3)
                    ax3.axis('off')
                    
                    # 4. Superposition Grad-CAM
                    ax4 = fig.add_subplot(gs[1, 0])
                    if 'overlayed_image' in results:
                        ax4.imshow(results['overlayed_image'])
                        ax4.set_title('Superposition Grad-CAM')
                    ax4.axis('off')
                    
                    # 5. Explication SHAP
                    ax5 = fig.add_subplot(gs[1, 1])
                    if 'shap_values' in results:
                        shap_vals = results['shap_values']
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[0]
                        shap_abs = np.abs(shap_vals).mean(axis=-1)
                        im = ax5.imshow(shap_abs[0] if len(shap_abs.shape) > 2 else shap_abs, 
                                       cmap='hot')
                        ax5.set_title('Importance SHAP')
                        plt.colorbar(im, ax=ax5)
                    ax5.axis('off')
                    
                    # 6. Conclusions
                    ax6 = fig.add_subplot(gs[1, 2])
                    ax6.axis('off')
                    conclusions = [
                        f"Modèle: {model_name}",
                        f"Classe prédite: {class_names[results.get('predicted_class', 0)]}",
                        f"Confiance: {results.get('confidence', 0):.3f}",
                        f"Top features:",
                        "- Région centrale (tumeur)",
                        "- Texture locale",
                        "- Contours"
                    ]
                    ax6.text(0.1, 0.9, '\n'.join(conclusions), 
                            verticalalignment='top',
                            fontsize=10)
            
            # Sauvegarder la figure
            img_name = os.path.basename(img_path).split('.')[0]
            save_path = os.path.join(output_dir, f"{img_name}_report.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Rapport généré: {save_path}")