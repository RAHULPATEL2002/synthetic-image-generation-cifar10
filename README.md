# Synthetic Image Generation for Reducing Overfitting

This project investigates the role of synthetic image generation in improving generalization
and reducing overfitting in deep learning models under limited data conditions.

## Dataset
- CIFAR-10
- Limited real samples: 50 per class (500 total)
- Synthetic samples: 5000 generated via geometric transformations and noise injection

## Methodology
- Baseline model trained on limited real data
- Synthetic-enhanced model trained using real + synthetic data
- Architecture: ResNet-18 (ImageNet pretrained)
- Evaluation on real CIFAR-10 test set only

## Key Results
- Synthetic data improves test accuracy significantly
- Generalization gap reduced
- Stable training behavior observed

## Project Structure
