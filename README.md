# MLFlow Advanced ��🚀

> A custom-built high-performance machine learning library featuring modular neural networks with advanced backpropagation, optimization strategies, adaptive learning, batch processing, model persistence, and more.

---

## 🔍 Overview

**MLFlow Advanced** is a lightweight, extensible machine learning library written in Python. It provides you with complete control over neural network architectures and training processes, making it ideal for researchers, developers, and learners who want to experiment with custom algorithms and optimization techniques.

---

## ⚙️ Features

* ✅ Modular neural network with customizable layers
* ✅ Custom backpropagation implementation
* ✅ Support for ReLU and Sigmoid activations (pluggable architecture)
* ✅ Adaptive learning rate and batch training
* ✅ Early stopping and validation support
* ✅ Model persistence (save/load model via `pickle`)
* ✅ Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Binary Cross-Entropy Loss
* ✅ Pluggable architecture for metrics, activations, and optimizers

---

## 💡 Use Cases

* 🔬 **Research**: Test and experiment with new training algorithms or activation functions.
* 🏫 **Education**: Understand how deep learning works under the hood.
* 🧪 **Prototyping**: Build and train quick models for tabular datasets without external libraries like TensorFlow or PyTorch.
* 🤖 **Local AI Agents**: Ideal for building lightweight models integrated in offline or embedded systems.

---

## 🛠️ Installation

Once published on PyPI:

```bash
pip install mlflow-advanced
```



---

## 🚀 Quick Start

```python
from mlflow_advanced.models.neural_network import CustomNeuralNetwork
import numpy as np

# Sample Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR pattern

# Build and Train Model
model = CustomNeuralNetwork(layers=[2, 4, 1], activations=['relu', 'sigmoid'], learning_rate=0.1)
model.fit(X, y, epochs=1000)

# Predict
predictions = model.predict(X)
print("Predictions:\n", predictions)
```

---

## 📁 Project Structure

```
mlflow-advanced/
│
├── mlflow/
│   ├── models/
│   │   └── neural_network.py        # Core implementation
│   ├── utils/
│   │   └── metrics.py               # Accuracy, Loss, Precision, Recall, F1
│
├── setup.py                         # Package configuration
├── README.md                        # Project README
├── requirements.txt                 # Dependencies
└── .gitignore                       # Ignored files/folders
```

---

## 🧪 Testing

To run a test training script:

```bash
python test_run.py
```

---

## 💾 Saving & Loading Models

```python
# Save model
model.save_model("model.pkl")

# Load model
from mlflow_advanced.models.neural_network import CustomNeuralNetwork
loaded_model = CustomNeuralNetwork.load_model("model.pkl")
```

---

## 🔍 Metrics Implemented

* **Accuracy**: Percentage of correctly predicted labels
* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1-Score**: Harmonic mean of Precision and Recall
* **Binary Cross-Entropy Loss**: For binary classification

---

## 📈 Customization

Want to plug in your own activation functions or metrics? No problem — use the abstract base class `ActivationFunction` in `neural_network.py` and extend it.

Example:

```python
class CustomActivation(ActivationFunction):
    def forward(self, x):
        return your_function(x)

    def backward(self, x):
        return your_derivative(x)
```

---

## 🙌 Contribution Guide

We welcome contributions! Feel free to:

* Report bugs
* Suggest features
* Submit PRs to improve core logic or usability

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💼 Author

Developed by **\[Jatin Hans]**
📢 GitHub: [jatiiiiiinnnnnn](https://github.com/jatiiiiiinnnnnn)
🔗 LinkedIn: \[[JatinHans](https://www.linkedin.com/in/jatin-hans-53892921b/)]

---

## ⭐️ If you find this helpful...

Please consider starring the repository 🌟 to support the project.
