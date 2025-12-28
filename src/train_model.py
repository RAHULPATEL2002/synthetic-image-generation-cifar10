import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRAIN FUNCTION ----------------
def train(X, y, X_test, y_test, label):
    print(f"\n=== Training: {label.upper()} ===")

    # Convert to tensors
    X = torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(y).long()
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float() / 255.0
    y_test = torch.tensor(y_test).long()

    # Dataloaders (IMPORTANT)
    train_loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Model
    model = models.resnet18(pretrained=True)

    # Freeze backbone (VERY IMPORTANT FOR SMALL DATA)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, NUM_CLASSES)
    model.to(DEVICE)

    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        model.train()
        preds, targets = [], []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(yb.cpu().numpy())

        train_accuracy = accuracy_score(targets, preds)
        train_acc.append(train_accuracy)

        # ---------------- EVALUATION ----------------
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(yb.cpu().numpy())

        test_accuracy = accuracy_score(targets, preds)
        test_acc.append(test_accuracy)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    # ---------------- SAVE RESULTS ----------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save(f"results/{label}_train.npy", train_acc)
    np.save(f"results/{label}_test.npy", test_acc)

    torch.save(model.state_dict(), f"models/{label}_resnet18.pth")
    print(f"✔ Model saved: models/{label}_resnet18.pth")

    return train_acc, test_acc


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load datasets
    X_real = np.load("data/limited/X_real.npy")
    y_real = np.load("data/limited/y_real.npy")

    X_syn = np.load("data/synthetic/X_syn.npy")
    y_syn = np.load("data/synthetic/y_syn.npy")

    X_test = np.load("data/limited/X_test.npy")
    y_test = np.load("data/limited/y_test.npy")

    # ---------- BASELINE ----------
    train(
        X_real,
        y_real,
        X_test,
        y_test,
        label="baseline"
    )

    # ---------- SYNTHETIC ----------
    X_train = np.concatenate([X_real, X_syn])
    y_train = np.concatenate([y_real, y_syn])

    train(
        X_train,
        y_train,
        X_test,
        y_test,
        label="synthetic"
    )
