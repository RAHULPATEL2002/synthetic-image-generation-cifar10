import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import os

# ---------------- CONFIG ----------------
NUM_CLASSES = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

MODEL_PATH = "models/synthetic_resnet18.pth"  # CHANGE if needed

# ---------------- LOAD DATA ----------------
X_test = np.load("data/limited/X_test.npy")
y_test = np.load("data/limited/y_test.npy")

X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float() / 255.0
y_test = torch.tensor(y_test)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------- LOAD MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- PREDICTIONS ----------------
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        outputs = model(xb)
        preds = outputs.argmax(1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(yb.numpy())

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

# ---------------- PLOT ----------------
os.makedirs("results/figures", exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Synthetic Data Enhanced Model")

plt.tight_layout()
plt.savefig("results/figures/confusion_matrix.png", dpi=300)
plt.show()

print("✔ Confusion matrix saved to results/figures/confusion_matrix.png")
