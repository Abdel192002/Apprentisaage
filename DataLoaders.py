from tensorflow.keras.preprocessing.image import ImageDataGenerator
class DataLoader:

    def __init__(self,path, augment=True, batch_size=32, target_size=(224, 224)):
        assert path.endswith("/"), "Le chemin doit se terminer par '/'"
        
        # Initialiser les générateurs
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
        no_DA_gen = ImageDataGenerator()

        # Chargement des ensembles
        self.train_set_DA = DA_gen.flow_from_directory(
            path + "train/",
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        self.train_set_NO_DA = no_DA_gen.flow_from_directory(
            path + "train/",
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
     
        self.test_set = no_DA_gen.flow_from_directory(
            path + "test/",
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
    
    def get_datasets(self):
        """
        Retourne les ensembles d'entraînement, de validation et de test.

        Retourne:
        - train_set_DA: Ensemble d'entraînement avec augmentation
        - train_set_NO_DA: Ensemble d'entraînement sans augmentation
        - val_set: Ensemble de validation
        - test_set: Ensemble de test
        """
        return {
            "train_with_augmentation": self.train_set_DA,
            "train_without_augmentation": self.train_set_NO_DA,
            
            "test": self.test_set
        }