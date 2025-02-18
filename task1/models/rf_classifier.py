from task1.models.mnist_classifier_interface import MnistClassifierInterface
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RFClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=50):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X_train: np.array, y_train: np.array):
        X_train_reshaped = X_train.reshape(X_train.shape[0], 28 * 28)
        self.model.fit(X_train_reshaped, y_train)

    def predict(self, data: np.array):
        data_reshaped = data.reshape(1, -1)
        return self.model.predict(data_reshaped)
