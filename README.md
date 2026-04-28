# 🧠 MLP Classification with PyTorch nn — MNIST & FashionMNIST

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

> Two notebooks — one built **from scratch** using raw PyTorch tensors, and one refactored using **torch.nn** — both classifying images with a simple MLP.

---

## 📌 Overview

- **Notebook 1** — MLP built from scratch, no torch.nn, everything manual
- **Notebook 2** — Same MLP refactored using torch.nn, tested on MNIST and FashionMNIST with hyperparameter benchmarking

---

## 🏗️ Model Architecture

    Input Layer  →  Hidden Layer  →  Output Layer
       (784)        (256, ReLU)          (10)
    28x28 pixels  learned features   digit scores

- 📥 **Input**: 28×28 images flattened to 784 pixels
- 🔵 **Hidden layer**: 256 neurons with ReLU activation
- 📤 **Output**: 10 class scores
- 📉 **Loss**: nn.CrossEntropyLoss
- 🔁 **Optimizer**: torch.optim.SGD

---

## 📓 Notebooks

### Notebook 1 — Low-Level MLP 
| Feature | Details |
|--------|---------|
| ⚙️ Weight Init | Uniform random (-0.01, 0.01) |
| ➡️ Forward Pass | Manual matrix multiplications + ReLU |
| 📉 Loss Function | Log softmax + cross entropy from scratch |
| 🔁 Training Loop | Mini-batch SGD + manual backprop |
| 🔬 Case Study | Batch size sweep (2 → 2048) |

### Notebook 2 — torch.nn MLP 
| Feature | Details |
|--------|---------|
| 🏗️ Model | nn.Sequential, nn.Linear, nn.ReLU |
| 📦 Data | torchvision DataLoader |
| 📉 Loss | nn.CrossEntropyLoss |
| 🔁 Optimizer | torch.optim.SGD |
| 📊 Evaluation | Confusion matrix, precision, recall |
| 🔬 Benchmarking | Batch size, learning rate, epochs on FashionMNIST |

---

## 📊 Results

### MNIST (Notebook 2)
| Metric | Value |
|--------|-------|
| Final Train Accuracy | 98.18% |
| Final Val Accuracy | 97.61% |
| Final Train Loss | 0.0678 |
| Final Val Loss | 0.0826 |

### FashionMNIST
| Metric | Value |
|--------|-------|
| Final Train Accuracy | 89.55% |
| Final Val Accuracy | 87.58% |

---

## 🔬 FashionMNIST Hyperparameter Benchmarking

### Epochs
| Epochs | Val Accuracy |
|--------|-------------|
| 5 | 85.09% |
| 10 | 87.24% |
| **15** | **87.58%** ✅ best |
| 20 | 87.36% |
| 30 | 86.27% |

Peak accuracy at 15 epochs — beyond this the model starts to overfit.

### Learning Rate
| Learning Rate | Val Accuracy |
|--------------|-------------|
| 0.001 | 67.88% |
| 0.01 | 81.91% |
| 0.1 | 86.26% |
| **0.5** | **87.31%** ✅ best |

Higher learning rates work better here. 0.001 is too slow and barely converges in 10 epochs.

### Batch Size
| Batch Size | Val Accuracy |
|-----------|-------------|
| 32 | 87.03% |
| **64** | **87.10%** ✅ best |
| 128 | 86.58% |
| 256 | 84.87% |
| 512 | 81.92% |

Smaller batches generalize better — more frequent gradient updates help the model on this harder dataset.

---

## 💭 What We Learned Moving to torch.nn

- 🗂️ **No more manual parameter lists** — nn.Module tracks everything automatically
- 🔄 **No more grad.zero_() loops** — optimizer.zero_grad() handles it
- 📦 **DataLoader** replaces all manual shuffling and batching
- 📱 **Cleaner device handling** — .to(DEVICE) on the model is enough
- 🧹 **Much shorter code** — same results with far less boilerplate

---

## 📂 Datasets

**MNIST** — 60,000 training + 10,000 test, handwritten digits 0–9, 28×28 grayscale

**FashionMNIST** — same structure as MNIST but clothing items, significantly harder (~87% vs ~97% accuracy)

---

## 🚀 Run it yourself

Open either notebook in **Google Colab** — no setup needed.

`Runtime → Change runtime type → T4 GPU ✅`
