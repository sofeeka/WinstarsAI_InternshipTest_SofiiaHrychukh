# Task 1: Image Classification + OOP
### Meeting the requirements
In this task I used a publicly available MNIST dataset and built 3 classification models around it:
1. Random Forest (RF)
2. Feed-Forward Neural Network (FNN) or (NN)
3. Convolutional Neural Network (CNN)

Every classificator implements `MnistClassifierInterface`, that has two public methods `train()` and `predict()`. This ensures that one could smoothly swap the implementations of classes if needed. 

Each model's use is implemented in separate classes `RFClassifier`, `FNNClassifier` and `CNNClassifier` respectfully. All preciously mentioned classificator implementations are hidden behind `MnistClassifier`class. It provides predictions with exactly the same structure for all the classifiers, no matter the algorithm ("rf", "nn" or "fnn", "cnn") chosen on creation of an instance.

### Installation
#### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- Git
- Virtual environment tools (venv or conda)

#### Clone the repository
```bash
git clone <https://github.com/sofeeka/WinstarsAI_InternshipTest_SofiiaHrychukh>
cd <WinstarsAI_InternshipTest_SofiiaHrychukh>
```

#### Set Up Virtual Environment
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

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Structure of the project
- `MnistClassifier`, `MnistClassifierInterface` as well as three algorithm implementations are located in directory `task1/models`
- directory `utils` has a file `data_loader.py` where a method `load_mnist_data()` can be found. It is used to fetch the data. 