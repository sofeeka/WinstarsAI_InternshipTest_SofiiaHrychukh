from mnist_classifier_interface import MnistClassifierInterface
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RFClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)

    def train(self, X_train: np.array, y_train: np.array):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.array):
        return self.model.predict(X_test)
