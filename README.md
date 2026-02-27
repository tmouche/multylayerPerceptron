# 🧠 Multilayer Perceptron

A lightweight neural network framework implemented from scratch in Python.

---

## 📚 Table of Contents

1. 🚀 Introduction
2. 🏗️ Architecture  
3. ⚙️ Features  
4. 🛠 Tech Stack  
5. 🤖 AI Utilisation
6. 📖 Sources  

---

## 🚀 1. Introduction

This project is an implementation of a **Multilayer Perceptron (MLP)** neural network built entirely from scratch, without using any machine learning frameworks.
The goal of the network is to classify breast tumors as **Benign** or **Malignant**.  
Before training the model, an initial data analysis phase is required to identify which features are the most relevant to discriminate between the two classes, in order to maximize accuracy while keeping the network size minimal.

The main objective is to deeply understand and manually implement the core concepts behind artificial neural networks:

- 🔁 Feedforward propagation  
- 🔄 Backpropagation  
- 📉 Gradient-based optimization  

This project was developed as part of the **42 school curriculum**.  
The original subject and constraints are available in this repository.

The implemented network focuses on **binary classification**, prioritizing clarity, modularity, and explicit mathematical reasoning over abstraction or performance.

---

## 🏗️ 2. Architecture

The project is structured around a modular and layered design to clearly separate responsibilities within the training pipeline.

- Core
  - Model
  - Network
  - Layer
- Ml_tools
  - Activations
  - Evaluations
  - Fire
  - Initialisations
  - Losses
  - Optimizers
  - Utils
- Utils
  - Contant
  - Exception
  - History
  - Logger
  - Process Dataset
  - Types

#### 🪄 Model
The **Model** class is the main entry point of the framework.

it is responsible for: 
  - Managing the gloval training process (macro-level control)
  - Creating and handling batches
  - Logging training progress across epochs
  - Monitoring metrics throughout training

In short, **Model** orchestrates the full training lifecycle.

#### 🔗 Network
The **Network** class acts almost like a structured data container.

It stores:
  - weights and biases
  - Learning Rate
  - Batch size
  - The list of layers

It does not directly manage training logic but holds all parameters required for computation.

#### ➰ Layer
The **Layer** class is primarily declarative.

Its role is to define:
  - The layer's size (number of neurons)
  - The activation function
  - The parameter initialization strategy

Layers describe *what* the network looks like, not *how* it trains

#### 🔥 Fire - Micro Training Engine
The **Fire** class handle the core of the micro-level training logic.

It is responsible of:
  - Forward propagation
  - Back propaation
  - Gradient computation

Separating **Fire** and **Network** provides greater flexibility and cleaner design.
Most important the separation allows a dynamic usage of alternative weight and bias sets
This disign choice is especially useful for optimizers such as **Nesterov Accelerated Gradient**, where parameters are temporily projected before computing gradients.

#### 📊 Optimizers
The **Optmizer** class define the policy used to update weights and biases.

Each Optimizer:
  - Uses gradients computed by **Fire**
  - Applies it own upgrade strategy
  - Modifies the **Network** parameters accordingly

Different optimizers implement different update dynamics (momentum, adaptive learning rates, projection-base update, etc..)

#### 🧰 Utils
The project also includes several utility modules:
  - **Activations** -> Activation functions and derivatives
  - **Losses** -> Loss functions and associated gradients
  - **Evaluations** -> Accuracy, precision, recall, F1-score
  - **Initialisations** -> Weight initialization strategies
  - **History** -> Metric tracking and training history
  - **Logger** -> Structured logging system
  - **Process Dataset** -> Dataset preprocessing and batching
  - **Types** -> Custom types definitions
  - **Exception** -> Custom exception handling
  - **Constant** -> Shared constants

 #### 🎯 Design Philosophy
 - Clear separation between declarative structure and training logic
 - Explicit gradient computation
 - Optimizer flexibilty
 - Modularity and extensibility
 - Educational focus over abstraction
The architecture prioritizes transparency and mathematical clarity over automation or hidden mechanisms.

---

## ⚙️ 3. Features

### 📈 Optimization Algorithms

Several gradient-based optimization algorithms are implemented:

#### 🔹 Gradient Descent
The baseline optimization method.  
Weights are updated in the opposite direction of the loss gradient.

✔ Simple  
✔ Stable  
✖ Can converge slowly  

---

#### 🔹 RMSProp
Adaptive learning rate method using a moving average of squared gradients.

✔ Reduces oscillations  
✔ Handles unstable gradients  
✔ Faster convergence in many cases  

---

#### 🔹 Nesterov Accelerated Gradient
Momentum-based optimizer using a *lookahead* mechanism before computing gradients.

✔ Anticipates parameter updates  
✔ Improves convergence speed  
✔ Reduces overshooting  

---

#### 🔹 Adam
Combines momentum and adaptive learning rates.

✔ Per-parameter learning rate  
✔ Efficient and robust  
✔ Works well in most practical cases  

---

### 🎯 Binary Classification

The network is designed to solve **binary classification problems**.

- Outputs are interpreted as probabilities  
- A fixed threshold is applied for class decision  
- Predictions are compared to ground truth labels  

---

### 📊 Training Metrics

The following metrics are computed during training and evaluation:

#### ✅ Accuracy
Proportion of correct predictions over total samples.

#### 📉 Loss
Measures prediction error.  
This is the function minimized during training.

#### 🎯 Precision
Among predicted positives, how many are truly positive.

#### 🔍 Recall
Among actual positives, how many were correctly detected.

#### ⚖️ F1 Score
Harmonic mean of precision and recall.  
Useful when dealing with class imbalance.

---

## 🛠 4. Tech Stack

- 🔢 NumPy 
- 🐼 Pandas
- 📈 Plotly

---

## 🤖 5. AI Utilisation

**Transparency statement**

Using of OpenAI ChatGPT 5.2 Free Trial:
  - Explain with exemples some concepts
  - Generate DocStrings and the README template

---

## 📖 6. Sources
All sources I used to understand concepts:
  - https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  - https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
  - https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
  - https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
  - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
  - https://developer.nvidia.com/blog/a-data-scientists-guide-to-gradient-descent-and-backpropagation-algorithms/
  - https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/
  - https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
  - https://www.geeksforgeeks.org/machine-learning/what-is-sparse-categorical-crossentropy/
  - https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
  - https://www.geeksforgeeks.org/deep-learning/rmsprop-optimizer-in-deep-learning/
  - https://medium.com/@piyushkashyap045/understanding-rmsprop-a-simple-guide-to-one-of-deep-learnings-powerful-optimizers-403baeed9922
  - https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/

---

## 📌 Notes

- 🚫 No machine learning frameworks were used  
- 🧮 All forward passes and backpropagation steps are manually implemented  
- 🎓 The project prioritizes algorithmic understanding over performance  
