import pickle
import numpy as np
import os


def load_batch(file_path):
    """
    Load a single CIFAR-10 batch file.
    Returns:
        X : numpy array of shape (N, 32, 32, 3)
        y : numpy array of shape (N,)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as fo:
        batch = pickle.load(fo, encoding="bytes")

    X = batch[b"data"]
    y = batch[b"labels"]

    # Reshape to (N, 32, 32, 3)
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return X, np.array(y)


def load_cifar10(data_dir):
    """
    Load CIFAR-10 dataset from directory.

    Args:
        data_dir (str): Path to CIFAR-10 batch files

    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train, y_train = [], []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))

    # Sanity checks
    assert X_train.shape[0] == 50000, "Training set size mismatch"
    assert X_test.shape[0] == 10000, "Test set size mismatch"

    return X_train, y_train, X_test, y_test
