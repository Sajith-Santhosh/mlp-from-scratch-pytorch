# 🔢 MNIST Digit Classification with MLP in PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

> A **low-level** implementation of a Multilayer Perceptron (MLP) for handwritten digit 
> classification — built from scratch using PyTorch tensors and autograd, 
> **no torch.nn used**. 🚫📦

The full implementation is available as a Jupyter notebook, developed and tested 
on Google Colab with all dependencies pre-installed. Just open and run! ▶️

---

## 📌 Overview

This project was built as part of an **Introduction to Deep Learning** course. 
The goal was to understand what actually happens under the hood when training a 
neural network, by implementing everything manually — forward pass, loss function, 
and gradient updates.

---

## 🏗️ Model Architecture

```
Input Layer     →     Hidden Layer     →     Output Layer
  (784)               (256, ReLU)               (10)
28x28 pixels       learned features          digit scores
```

- 📥 **Input**: 28×28 MNIST images flattened to 784 pixels  
- 🔵 **Hidden layer**: 256 neurons with ReLU activation  
- 📤 **Output**: 10 class scores, one per digit (0–9)  
- 📉 **Loss**: Cross Entropy  

---

## ✅ What is Implemented

| Feature | Details |
|--------|---------|
| ⚙️ Weight Init | Uniform random distribution (-0.01, 0.01) |
| ➡️ Forward Pass | Manual matrix multiplications + ReLU |
| 📉 Loss Function | Log softmax + cross entropy from scratch |
| 🔁 Training Loop | Mini-batch gradient descent + manual backprop |
| ⚡ GPU Support | CUDA via PyTorch |
| 📊 Evaluation | Confusion matrix, precision, recall |
| 📈 Visualization | Learning curves, weight patterns, bias charts |
| 🔬 Case Study | Systematic batch size hyperparameter sweep |

---

## 📊 Results

| Model | Validation Accuracy |
|-------|-------------------|
| 📏 Linear Model (baseline) | ~92.4% |
| 🧠 MLP | ~97.2% |

The MLP significantly outperforms the linear model by learning **non-linear 
feature representations** in the hidden layer.

---

## 🔬 Batch Size Case Study

A systematic sweep over batch sizes from **2 to 2048** was conducted to study 
the effect on generalization performance.

| Batch Size | Val Accuracy | Notes |
|-----------|-------------|-------|
| 2 | 96.03% | too noisy |
| 4 | 97.77% | |
| **8** | **98.27%** | ✅ best |
| 16 | 98.10% | |
| 32 | 97.89% | |
| 64 | 97.79% | |
| 128 | 97.24% | |
| 256 | 95.85% | |
| 512 | 93.87% | generalization gap starts |
| 1024 | 91.94% | |
| 2048 | 90.24% | too large |

### 🔍 Key Findings

- 🏆 Sweet spot was around **batch_size=8**, giving 98.27% validation accuracy
- 📉 Very small batches (2, 4) introduce too much noise in gradient estimates
- 📦 Very large batches take fewer update steps per epoch and converge to sharper 
  minima that generalize worse — known as the **generalization gap**
- 📐 There is a clear trend: performance peaks around 8 and degrades consistently 
  in both directions

---

## 💭 Reflections on Low-Level PyTorch

Working without `torch.nn` made it clear why higher-level abstractions exist:

- 🗂️ **Parameter management** — manually maintaining a list of parameters is error 
  prone, forget to add a layer and it simply won't train
- 🔄 **Gradient zeroing** — calling `grad.zero_()` every batch should be automatic
- 📱 **Device handling** — moving model and inputs to device separately is a very 
  common source of bugs
- 🔁 **Training loop** — the forward, loss, backward, update cycle is identical 
  every time and should be abstracted away

> This was a great exercise in understanding what `torch.nn` actually wraps 
> and why those abstractions exist. 💡

---

## 📂 Dataset

**MNIST** — one of the most well known benchmarks in deep learning

- 🖼️ 60,000 training images + 10,000 test images
- ✍️ Handwritten digits 0–9
- 📐 Each image is 28×28 pixels in grayscale

---

## 🚀 Run it yourself

Open the notebook in **Google Colab** — no setup needed, all dependencies are 
pre-installed. Just make sure to enable GPU runtime for faster training.

`Runtime → Change runtime type → T4 GPU ✅`
