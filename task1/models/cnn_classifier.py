from task1.models.mnist_classifier_interface import MnistClassifierInterface
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class CNNClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

    def train(self, X_train: np.array, y_train: np.array):
        X_train_reshaped = X_train.reshape(-1, 28, 28, 1)

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(X_train_reshaped, y_train, epochs=5)

    def predict(self, data: np.array):
        data_reshaped = data.reshape(-1, 28, 28, 1)
        prediction = self.model.predict(data_reshaped)
        return np.argmax(prediction)
