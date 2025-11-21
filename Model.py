import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Dropout, Reshape, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class CNN_ViT:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def transformer_block(self, x, num_heads, ff_dim, dropout=0.1):
        # Normalisation
        x_norm = LayerNormalization(epsilon=1e-6)(x)

        # Attention multi-tête
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x_norm, x_norm)

        # Skip connection
        x = Add()([x, attention_output])

        # Normalisation + MLP
        x_norm2 = LayerNormalization(epsilon=1e-6)(x)
        ff_output = Dense(ff_dim, activation='relu')(x_norm2)
        ff_output = Dense(x.shape[-1])(ff_output)
        x = Add()([x, ff_output])
        return x

    def build_model(self):
        input_image = Input(shape=self.input_shape)

        # Bloc CNN (extraction locale)
        x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(input_image)
        x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)

        # Préparation pour Transformer
        x = Reshape((-1, x.shape[-1]))(x)  # (batch, patches, channels)

        # Transformer blocks
        x = self.transformer_block(x, num_heads=4, ff_dim=256)
        x = self.transformer_block(x, num_heads=4, ff_dim=256)

        # Pooling global
        x = GlobalAveragePooling1D()(x)

        # Classification
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        # Compilation
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
