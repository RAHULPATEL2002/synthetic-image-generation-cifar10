import numpy as np
import os
from load_cifar10 import load_cifar10

# ---------------- CONFIG ----------------
DATA_DIR = "data/cifar10"   # <-- RELATIVE PATH (GOOD PRACTICE)
SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42

# ---------------- LOAD DATA ----------------
X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)

np.random.seed(RANDOM_SEED)

limited_idx = []

for cls in range(10):
    cls_idx = np.where(y_train == cls)[0]
    selected = np.random.choice(cls_idx, SAMPLES_PER_CLASS, replace=False)
    limited_idx.extend(selected)

limited_idx = np.array(limited_idx)

X_limited = X_train[limited_idx]
y_limited = y_train[limited_idx]

# ---------------- SAVE DATA ----------------
os.makedirs("data/limited", exist_ok=True)

np.save("data/limited/X_real.npy", X_limited)
np.save("data/limited/y_real.npy", y_limited)

np.save("data/limited/X_test.npy", X_test)
np.save("data/limited/y_test.npy", y_test)

# ---------------- SANITY CHECK ----------------
print("Limited real dataset created")
print("Total samples:", X_limited.shape[0])
print("Samples per class:", SAMPLES_PER_CLASS)
print("Class distribution:", np.bincount(y_limited))
