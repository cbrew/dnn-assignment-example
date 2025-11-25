# plots.py
import matplotlib.pyplot as plt
import numpy as np


def plot_example_digits(X, y, nrows=2, ncols=5):
    """Replicate your sample visualization: one example per digit."""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    ax = ax.flatten()

    for d in range(10):
        idx = np.where(y == d)[0][0]
        img = X[idx].reshape(28, 28)
        ax[d].imshow(img, cmap="Greys")
        ax[d].set_xticks([])
        ax[d].set_yticks([])
        ax[d].set_title(str(d))

    plt.tight_layout()
    plt.show()


def plot_learning_curves(
    tr_losses,
    tr_accs,
    dev_losses,
    dev_accs,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(tr_losses, label="Train")
    axes[0].plot(dev_losses, label="Dev")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(tr_accs, label="Train")
    axes[1].plot(dev_accs, label="Dev")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
