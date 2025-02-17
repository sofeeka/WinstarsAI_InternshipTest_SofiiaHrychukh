from abc import ABC, abstractmethod
import numpy as np


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train: np.array, y_train: np.array):
        pass

    @abstractmethod
    def predict(self, X_test: np.array):
        pass
