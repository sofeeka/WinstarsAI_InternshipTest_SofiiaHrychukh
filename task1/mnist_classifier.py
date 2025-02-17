class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm not in ['cnn', 'rf', 'nn']:
            raise ValueError(f"Invalid algorithm '{algorithm}'.")

        self.algorithm = algorithm

            