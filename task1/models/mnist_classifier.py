from task1.models.rf_classifier import RFClassifier
from task1.models.cnn_classifier import CNNClassifier
from task1.models.fnn_classifier import FNNClassifier


class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm not in ['cnn', 'rf', 'nn']:
            raise ValueError(f"Invalid algorithm '{algorithm}'.")

        self.algorithm = algorithm
        self.__initialise_model__()

    def __initialise_model__(self):
        if self.algorithm == 'rf':
            self.model = RFClassifier()
        elif self.algorithm == 'cnn':
            self.model = CNNClassifier()
        else:
            self.model = FNNClassifier()

    def train(self, x_train, y_train):
        self.model.train(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
