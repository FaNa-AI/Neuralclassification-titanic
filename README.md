

# 🧠 Binary Classification with PyTorch

This repository contains a simple neural network implementation in **PyTorch** for performing **binary classification** on synthetic data generated using `scikit-learn`.

## 📌 Overview

The goal is to train a fully connected neural network to classify samples into two classes using synthetic data. This is a basic example that demonstrates how to:

* Generate classification data
* Define a feedforward neural network
* Train and evaluate a PyTorch model
* Visualize training loss

## 📊 Dataset

The dataset is generated using `make_classification` from `sklearn.datasets` with the following parameters:

* `n_samples=1000`
* `n_features=10`
* `n_classes=2`

Data is then split into training (70%) and test (30%) sets.

## 🧠 Model Architecture

The model is a simple feedforward neural network with:

* Input layer: 10 features
* Hidden layer: 64 neurons, ReLU activation
* Output layer: 2 classes (for binary classification)

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## ⚙️ Training

* Loss Function: `CrossEntropyLoss`
* Optimizer: `Adam` with learning rate `0.01`
* Epochs: 20

Training loss is recorded and printed at each epoch.

## ✅ Evaluation

The model is evaluated on the test set using `accuracy_score` from scikit-learn.

Example output:

```
Accuracy: 88.67%
```

## 📉 Training Loss Plot

A line plot is generated showing the training loss across epochs.

![Training Loss Plot](training_loss.png)

> *(Save the figure using `plt.savefig("training_loss.png")` if you want to include the image in GitHub)*

## 📂 File Structure

```
├── classification_nn.py       # Main training script
└── README.md                  # Project documentation
```

## ▶️ How to Run

Make sure you have the required packages:

```bash
pip install torch matplotlib scikit-learn
```

Then run the script:

```bash
python classification_nn.py
```

