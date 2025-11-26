import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import regularizers


class ResNet50_Model:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
        # Input
        input_image = Input(shape=self.input_shape)

        # ===== Backbone ResNet50 (sans input_tensor !) =====
        backbone = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )

        # Appel du backbone sur le tensor d’entrée
        x = backbone(input_image)

        # ===== Head classification =====
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        # Model
        self.model = Model(inputs=input_image, outputs=output)

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self):
        if self.model is None:
            self.build_model()
        return self.model
