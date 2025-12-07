"""
DataLoader avec intégration DVC pour le dataset SIPAKMED
Charge automatiquement les données depuis DVC si nécessaire
"""

import subprocess
import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    """
    Chargeur de données avec support DVC pour SIPAKMED
    
    Args:
        path (str): Chemin vers le dossier des données (doit finir par '/')
        augment (bool): Activer l'augmentation de données
        batch_size (int): Taille des batches
        target_size (tuple): Taille cible des images (hauteur, largeur)
    """
    
    def __init__(self, path, augment=True, batch_size=32, target_size=(224, 224)):
        assert path.endswith("/"), "Le chemin doit se terminer par '/'"
        
        # Ensure data is pulled from DVC
        self._pull_data_from_dvc(path)
        
        # Data augmentation configuration
        if augment:
            DA_gen = ImageDataGenerator(
                featurewise_center=False,
                rotation_range=5,
                fill_mode="nearest",
                zoom_range=[1/1.0, 1/1.0],
                width_shift_range=0.0,
                height_shift_range=0.0,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.5, 1.3],
                channel_shift_range=20
            )
        else:
            DA_gen = ImageDataGenerator()
            
        no_DA_gen = ImageDataGenerator()

        # Create data generators
        self.train_set_DA = DA_gen.flow_from_directory(
            path + "train/",
            target_size=target_size,
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.train_set_NO_DA = no_DA_gen.flow_from_directory(
            path + "train/",
            target_size=target_size,
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
     
        self.test_set = no_DA_gen.flow_from_directory(
            path + "test/",
            target_size=target_size,
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )

        self.class_names = list(self.test_set.class_indices.keys())
        self.path = path
        self.batch_size = batch_size
        self.target_size = target_size
        
        print(f"DataLoader initialisé:")
        print(f"   - Classes: {self.class_names}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Image size: {target_size}")
        print(f"   - Augmentation: {'Activée' if augment else 'Désactivée'}")
    
    def _pull_data_from_dvc(self, path):
        import subprocess
        import os
        from pathlib import Path
        
        print(f"Vérification des données à: {path}")
        
        # Clean path
        path = path.rstrip('/').rstrip('\\')
        path_obj = Path(path)
        
        # Check if data exists locally
        train_exists = (path_obj / "train").exists() and any((path_obj / "train").iterdir())
        test_exists = (path_obj / "test").exists() and any((path_obj / "test").iterdir())
        
        if train_exists and test_exists:
            print(f"Données déjà présentes localement")
            return
        
        print(f"Données manquantes. Téléchargement depuis DVC...")
        
        # Get project root (where .dvc folder is)
        current = Path.cwd()
        project_root = None
        
        for _ in range(5):
            if (current / ".dvc").exists():
                project_root = current
                break
            current = current.parent
        
        if not project_root:
            print("Dossier .dvc introuvable!")
            raise FileNotFoundError("Projet DVC non trouvé")
        
        print(f"Projet DVC: {project_root}")
        
        try:
            # Method 1: Pull using the .dvc file
            dvc_file = "data/processed.dvc"
            dvc_file_path = project_root / dvc_file
            
            if dvc_file_path.exists():
                print(f"Pull via {dvc_file}")
                result = subprocess.run(
                    ['dvc', 'pull', dvc_file],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"Données téléchargées")
                    print(f"   {result.stdout[:200]}...")
                else:
                    print(f"⚠ Erreur: {result.stderr[:200]}")
                    
                    # Method 2: Try general pull
                    print("Tentative: dvc pull")
                    subprocess.run(['dvc', 'pull'], cwd=project_root, check=True)
                    print("Toutes les données téléchargées")
            
            else:
                # Method 3: Try to pull the directory
                print(f"Tentative: dvc pull data/processed/")
                subprocess.run(
                    ['dvc', 'pull', 'data/processed/'],
                    cwd=project_root,
                    check=True
                )
                print("Données téléchargées")
                
        except subprocess.CalledProcessError as e:
            print(f"Échec du pull DVC: {e.stderr[:200]}")
            
            # Check if data exists anyway (maybe from previous run)
            if (path_obj / "train").exists() and (path_obj / "test").exists():
                print("⚠ Continuation avec données existantes...")
            else:
                print("\nInstructions de dépannage:")
                print("1. Exécutez manuellement dans un terminal:")
                print(f"   cd {project_root}")
                print("   dvc pull data/processed.dvc")
                print("2. Ou: dvc pull")
                print("3. Vérifiez le remote: dvc remote list")
                
                raise FileNotFoundError(
                    f"Données introuvables à {path}. "
                    f"Exécutez 'dvc pull' dans {project_root}"
                )
    
    def _verify_class_directories(self, path_obj):
        """Vérifie que les dossiers de classes existent"""
        expected_classes = {'normal', 'benign', 'malignant'}
        
        # Check train classes
        train_path = path_obj / "train"
        if train_path.exists():
            train_classes = {d.name for d in train_path.iterdir() if d.is_dir()}
            missing_train = expected_classes - train_classes
            if missing_train:
                print(f"Attention: Classes train manquantes: {missing_train}")
            else:
                print(f"Toutes les classes train trouvées: {sorted(train_classes)}")
                
                # Count images in each class
                for cls in sorted(train_classes):
                    images = list((train_path / cls).glob("*.bmp")) or list((train_path / cls).glob("*.*"))
                    print(f"  - Train/{cls}: {len(images)} images")
        
        # Check test classes
        test_path = path_obj / "test"
        if test_path.exists():
            test_classes = {d.name for d in test_path.iterdir() if d.is_dir()}
            missing_test = expected_classes - test_classes
            if missing_test:
                print(f"⚠ Attention: Classes test manquantes: {missing_test}")
            else:
                print(f" Toutes les classes test trouvées: {sorted(test_classes)}")
                
                # Count images in each class
                for cls in sorted(test_classes):
                    images = list((test_path / cls).glob("*.bmp")) or list((test_path / cls).glob("*.*"))
                    print(f"  - Test/{cls}: {len(images)} images")
    
    def get_datasets(self):
        """Retourne tous les datasets"""
        return {
            "train_with_augmentation": self.train_set_DA,
            "train_without_augmentation": self.train_set_NO_DA,
            "test": self.test_set
        }
    
    def get_generators(self):
        """Retourne les générateurs (compatibilité)"""
        return self.train_set_DA, self.test_set
    
    def get_class_names(self):
        """Retourne les noms des classes"""
        return self.class_names
    
    def get_num_classes(self):
        """Retourne le nombre de classes"""
        return len(self.class_names)
    
    @property
    def num_classes(self):
        return len(self.class_names)
    
    def get_class_counts(self):
        """Retourne le nombre d'images par classe"""
        # For training data
        train_counts = {}
        for cls in self.class_names:
            train_path = Path(self.path) / "train" / cls
            if train_path.exists():
                images = list(train_path.glob("*.bmp")) or list(train_path.glob("*.*"))
                train_counts[cls] = len(images)
        
        # For test data
        test_counts = {}
        for cls in self.class_names:
            test_path = Path(self.path) / "test" / cls
            if test_path.exists():
                images = list(test_path.glob("*.bmp")) or list(test_path.glob("*.*"))
                test_counts[cls] = len(images)
        
        return {
            "train": train_counts,
            "test": test_counts
        }
    
    def get_summary(self):
        """Retourne un résumé des données"""
        return {
            "path": self.path,
            "classes": self.class_names,
            "num_classes": self.get_num_classes(),
            "batch_size": self.batch_size,
            "target_size": self.target_size,
            "counts": self.get_class_counts(),
            "train_samples": self.train_set_DA.samples,
            "test_samples": self.test_set.samples
        }


# Fonction utilitaire pour créer un DataLoader
def create_dataloader(data_path="data/processed/", **kwargs):
    """
    Fonction factory pour créer un DataLoader
    
    Args:
        data_path: Chemin vers les données
        **kwargs: Arguments passés à DataLoader
    
    Returns:
        DataLoader instance
    """
    return DataLoader(data_path, **kwargs)


if __name__ == "__main__":
    # Test du DataLoader
    print("Test du DataLoader...")
    try:
        loader = DataLoader("data/processed/", augment=True, batch_size=32)
        summary = loader.get_summary()
        print("\nRésumé des données:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Erreur: {e}")