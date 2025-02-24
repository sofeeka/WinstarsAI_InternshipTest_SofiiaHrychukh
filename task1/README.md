# Task 1: Image Classification + OOP

---

## Meeting the requirements

In this task I used a publicly available MNIST dataset and built 3 classification models around it:

1. Random Forest (RF)
2. Feed-Forward Neural Network (FNN) or (NN)
3. Convolutional Neural Network (CNN)

Every classificator implements `MnistClassifierInterface`, that has two public methods `train()` and `predict()`.

Each model is implemented in separate classes `RFClassifier`, `FNNClassifier` and `CNNClassifier` respectively. All
these implementations are hidden behind `MnistClassifier`class. It provides predictions with exactly the same structure
for all the classifiers, regardless of the algorithm ("rf", "nn" or "fnn", "cnn") specified on creation
of `MnistClassifier` instance.

---

## Installation

### 1. Prerequisites

Ensure you have the following installed:

- Python (3.10-3.12)
- Git
- Virtual environment tools (venv or conda)

### 2. Clone the repository

```bash
git clone <https://github.com/sofeeka/WinstarsAI_InternshipTest_SofiiaHrychukh>
cd <WinstarsAI_InternshipTest_SofiiaHrychukh>
```

### 3. Set Up Virtual Environment

##### Linux

```bash
python -m venv venv
source venv/bin/activate
```

##### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Structure of the project

- `MnistClassifier`, `MnistClassifierInterface` as well as three algorithm implementations are located in
  directory `task1/models`
- directory `utils` has a file `data_loader.py` where a method `load_mnist_data()` can be found. It is used to fetch the
  data. 