# Synthetic Image Generation for Mitigating Overfitting in Deep Learning under Data-Scarce Conditions

This project investigates the effectiveness of **synthetic image generation** as a data-centric strategy for mitigating overfitting and improving generalization in deep learning models trained under **extreme data-scarce conditions**. The study is conducted using a controlled CIFAR-10 setup and a standardized ResNet-18 architecture.

The repository accompanies a research paper and emphasizes **reproducibility, controlled experimentation, and data-level analysis**, rather than architectural novelty.

---

## Dataset

- **Benchmark:** CIFAR-10  
- **Number of classes:** 10  
- **Image resolution:** 32 × 32 (RGB)

### Training Data (Low-Data Setting)
- Real images: **50 per class (500 total)**
- Synthetic images: **~5,000**, generated from real samples using:
  - Geometric transformations (rotation, translation, scaling, horizontal flipping)
  - Noise-based perturbations (Gaussian noise)

### Evaluation Data
- **10,000 real CIFAR-10 test images**
- No synthetic images are used during evaluation

This strict separation ensures that reported performance reflects **true generalization** rather than memorization of synthetic samples.

---

## Methodology

Two training configurations are evaluated under identical conditions:

### 1. Baseline Model
- Trained using **real data only** (500 images)
- Serves as a reference for overfitting behavior under extreme data scarcity

### 2. Synthetic-Enhanced Model
- Trained using **real + synthetic images**
- Synthetic samples generated via label-preserving transformations
- Same architecture, optimizer, and training schedule as the baseline

---

## Model and Training Details

- **Architecture:** ResNet-18 (ImageNet pretrained)
- **Loss function:** Cross-entropy loss
- **Optimizer:** Adam
- **Learning rate:** 1e-4
- **Batch size:** 32
- **Epochs:** 50
- **Evaluation:** Performed exclusively on real CIFAR-10 test data

---

## Key Results

| Setup      | Test Accuracy (%) | Training Accuracy (%) | Generalization Gap |
|-----------|------------------|----------------------|--------------------|
| Baseline  | 29.0             | 48.6                 | 0.196              |
| Synthetic | 32.3             | 62.6                 | 0.303              |

### Observations
- Synthetic image augmentation yields **consistent improvements in test accuracy**
- Training convergence is faster and more stable with synthetic data
- Synthetic augmentation increases effective data diversity under low-data constraints
- Overfitting behavior is explicitly analyzed using training–test divergence

---

## Project Structure

```text
synthetic_image_generation_project/
├── src/
│   ├── train_model.py            # Baseline and synthetic training
│   ├── synthetic_ratio.py        # Synthetic data ratio experiments
│   ├── plot_curves.py            # Training vs test accuracy plots
│   └── confusion_matrix.py       # Confusion matrix generation
│
├── models/
│   ├── baseline_resnet18.pth
│   └── synthetic_resnet18.pth
│
├── results/
│   ├── figures/
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── synthetic_ratio.png
│   │
│   └── performance.csv
│
├── paper/
│   └── synthetic_image_generation_ieee.tex
│
├── requirements.txt
└── README.md
```


---

## Reproducibility

- Fixed random seeds for dataset sampling and training
- Identical architecture and hyperparameters across all experiments
- Single unified codebase for all configurations
- Synthetic images used **only during training**

---

## Research Contribution

- Empirical analysis of overfitting under **extreme low-data conditions**
- Demonstrates effectiveness of **lightweight, label-preserving synthetic image generation**
- Highlights **data-centric learning** as a practical alternative to architectural scaling
- Suitable for low-resource and constrained-data applications

---

## Citation

If you use this work in your research, please cite the accompanying paper or this repository.

