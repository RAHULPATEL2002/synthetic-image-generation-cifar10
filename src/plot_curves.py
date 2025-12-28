import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- CREATE FIGURE DIR ----------------
os.makedirs("results/figures", exist_ok=True)

# ---------------- LOAD RESULTS ----------------
baseline_train = np.load("results/baseline_train.npy")
baseline_test = np.load("results/baseline_test.npy")

synthetic_train = np.load("results/synthetic_train.npy")
synthetic_test = np.load("results/synthetic_test.npy")

epochs = range(1, len(baseline_train) + 1)

# ---------------- PLOT ----------------
plt.figure(figsize=(8, 6))

plt.plot(epochs, baseline_train, 'b--', label="Baseline Train")
plt.plot(epochs, baseline_test, 'b', label="Baseline Test")

plt.plot(epochs, synthetic_train, 'r--', label="Synthetic Train")
plt.plot(epochs, synthetic_test, 'r', label="Synthetic Test")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy (Baseline vs Synthetic Data)")
plt.legend()
plt.grid(True)

# ---------------- SAVE ----------------
plt.savefig("results/figures/training_curves.png", dpi=300, bbox_inches="tight")
plt.show()

print("✔ Training curves saved to results/figures/training_curves.png")
