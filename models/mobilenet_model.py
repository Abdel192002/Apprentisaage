from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class MobileNetV2_Model:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4,
                 dropout_rate=0.5, l2_reg=0.01, dense_units=256, freeze_backbone=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.dense_units = dense_units
        self.freeze_backbone = freeze_backbone
        self.model = None

    def build_model(self):
        input_image = Input(shape=self.input_shape)

        backbone = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_image
        )
        
        if self.freeze_backbone:
            backbone.trainable = False

        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.dense_units, activation='relu',
                 kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=input_image, outputs=output)

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        if self.model is None:
            self.build_model()
        self.model.summary()