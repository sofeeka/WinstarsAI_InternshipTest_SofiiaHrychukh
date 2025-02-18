from task1.models.rf_classifier import RFClassifier
from task1.models.cnn_classifier import CNNClassifier
from task1.models.fnn_classifier import FNNClassifier


class MnistClassifier:
    def __init__(self, algorithm):
        self.classifier = self.__create_classifier__(algorithm)

    @staticmethod
    def __create_classifier__(algorithm):
        if algorithm == 'rf':
            return RFClassifier()
        elif algorithm == 'cnn':
            return CNNClassifier()
        elif algorithm == 'nn':
            return FNNClassifier()
        else:
            raise ValueError(f"Invalid algorithm '{algorithm}'.")

    def train(self, X_train, y_train):
        return self.classifier.train(X_train, y_train)

    def predict(self, data):
        return self.classifier.predict(data)
