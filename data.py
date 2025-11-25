# data.py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset."""

    def __init__(self, input_data: np.ndarray, labels: np.ndarray):
        self.feats = input_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.feats[index]
        y = self.labels[index]
        return X, y


def load_mnist_normalized():
    """
    Load MNIST via fetch_openml and normalize to roughly [-1, 1].
    Returns X, y as numpy arrays.
    """
    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(int)

    # Your scaling: ((X / 255) - 0.5) * 2
    X = ((X / 255.0) - 0.5) * 2.0

    return X.to_numpy(), y.to_numpy()


def split_train_dev_test(X, y, train_size=60000, random_state=12):
    """
    First 60k -> train+dev, last 10k -> test (as in original notebook).
    Then stratified split train->(train, dev).
    """
    X_train_full, X_test = X[:train_size], X[train_size:]
    y_train_full, y_test = y[:train_size], y[train_size:]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1, random_state=random_state
    )
    (train_idx, dev_idx), = splitter.split(X_train_full, y_train_full)

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_dev, y_dev = X_train_full[dev_idx], y_train_full[dev_idx]

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


def make_dataloaders(
    X_train,
    y_train,
    X_dev,
    y_dev,
    X_test,
    y_test,
    batch_size=64,
    test_batch_size=16,
):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_set = MNISTDataset(X_train, y_train)
    dev_set = MNISTDataset(X_dev, y_dev)
    test_set = MNISTDataset(X_test, y_test)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    return train_loader, dev_loader, test_loader
