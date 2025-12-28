import numpy as np
import cv2
import os

# ---------------- CONFIG ----------------
SYNTHETIC_PER_IMAGE = 10
ROTATION_RANGE = (-20, 20)
NOISE_STD = 15
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ---------------- LOAD REAL DATA ----------------
X = np.load("data/limited/X_real.npy")
y = np.load("data/limited/y_real.npy")

synthetic_images = []
synthetic_labels = []

# ---------------- SYNTHETIC GENERATION ----------------
for img, label in zip(X, y):
    for _ in range(SYNTHETIC_PER_IMAGE):
        # Rotation
        angle = np.random.uniform(*ROTATION_RANGE)
        M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
        aug = cv2.warpAffine(
            img,
            M,
            (32, 32),
            borderMode=cv2.BORDER_REFLECT
        )

        # Gaussian noise
        noise = np.random.normal(0, NOISE_STD, aug.shape)
        aug = np.clip(aug + noise, 0, 255)

        synthetic_images.append(aug.astype(np.uint8))
        synthetic_labels.append(label)

X_syn = np.array(synthetic_images)
y_syn = np.array(synthetic_labels)

# ---------------- SAVE ----------------
os.makedirs("data/synthetic", exist_ok=True)

np.save("data/synthetic/X_syn.npy", X_syn)
np.save("data/synthetic/y_syn.npy", y_syn)

# ---------------- LOG ----------------
print("Synthetic data created")
print("Synthetic images:", X_syn.shape)
print("Synthetic labels:", y_syn.shape)
