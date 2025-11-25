"""
Activation-visualization helpers for the MNIST DNN example.

Typical usage from a notebook:

    from mnist_activations import (
        visualize_activations_on_loader,
        visualize_activations_by_label,
    )

    # Simple, whole-batch activations:
    visualize_activations_on_loader(model, dev_gen)

    # Conditional on digit identity (true label):
    visualize_activations_by_label(model, dev_gen, max_batches=3)

"""

from typing import Dict, List, Optional

import math
import torch
import matplotlib.pyplot as plt


# ---------- Core hook + batch capture ----------


def _register_fc_hooks(
    model: torch.nn.Module,
    activations: Dict[str, torch.Tensor],
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks on layers named fc1, fc2, fc3 if present."""
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for name in ["fc1", "fc2", "fc3"]:
        layer = getattr(model, name, None)
        if layer is None:
            continue

        def _make_hook(layer_name: str):
            def hook(module, inp, out):
                # Save a detached CPU copy; shape ~ (batch_size, hidden_units)
                activations[layer_name] = out.detach().cpu()

            return hook

        handles.append(layer.register_forward_hook(_make_hook(name)))

    return handles


def collect_activations_on_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Run a single batch through the model and collect activations."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    activations: Dict[str, torch.Tensor] = {}
    handles = _register_fc_hooks(model, activations)

    with torch.no_grad():
        batch = batch.float().to(device)
        _ = model(batch)

    # Remove hooks to avoid side-effects for later calls
    for h in handles:
        h.remove()

    return activations


# ---------- Simple global plots ----------


def _plot_histograms(activations: Dict[str, torch.Tensor]) -> None:
    """For each layer, plot a histogram of all activation values."""
    num_layers = len(activations)
    if num_layers == 0:
        print("No activations captured.")
        return

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
    if num_layers == 1:
        axes = [axes]

    for ax, (layer_name, act) in zip(axes, activations.items()):
        flat = act.flatten().numpy()
        ax.hist(flat, bins=50)
        ax.set_title(f"{layer_name} activations")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    plt.show()


def _plot_heatmaps(
    activations: Dict[str, torch.Tensor],
    max_neurons: int = 64,
) -> None:
    """Show a coarse heatmap of activations for the first few neurons per layer."""
    num_layers = len(activations)
    if num_layers == 0:
        return

    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 3 * num_layers))
    if num_layers == 1:
        axes = [axes]

    for ax, (layer_name, act) in zip(axes, activations.items()):
        # act shape: (batch_size, num_units)
        batch_size, num_units = act.shape[:2]
        keep = min(num_units, max_neurons)
        sub = act[:, :keep].numpy()  # (batch_size, keep)

        im = ax.imshow(sub, aspect="auto", interpolation="nearest")
        ax.set_title(f"{layer_name} (first {keep} units across batch)")
        ax.set_xlabel("Unit index")
        ax.set_ylabel("Batch index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()


def visualize_activations_on_loader(
    model: torch.nn.Module,
    loader,
    device: Optional[str] = None,
    max_batches: int = 1,
) -> None:
    """
    Grab up to `max_batches` from `loader` and visualize activations (global).

    This is the original "whole batch" view:
      * one histogram per layer
      * one heatmap per layer
    """
    # Take a few batches and concatenate them
    batches = []
    for i, (xb, yb) in enumerate(loader):
        batches.append(xb)
        if i + 1 >= max_batches:
            break

    batch = torch.cat(batches, dim=0)
    activations = collect_activations_on_batch(model, batch, device=device)

    print("Layers captured:", list(activations.keys()))
    _plot_histograms(activations)
    _plot_heatmaps(activations)


# ---------- NEW: label-conditional heatmaps ----------


def _plot_label_heatmaps(
    activations: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    max_neurons: int = 64,
    num_classes: int = 10,
    min_per_class: int = 1,
) -> None:
    """
    For each layer, make a grid of heatmaps, one per digit (0..9 by default).

    Each heatmap has:
      * rows = examples of that digit (from the sampled batches)
      * columns = first `max_neurons` units in that layer

    This uses the *true labels* (correct digit identity) for conditioning.
    """
    if len(activations) == 0:
        print("No activations captured.")
        return

    labels_np = labels.cpu().numpy()

    # Decide which classes to show: those in [0, num_classes) with enough examples
    present_classes = []
    for c in range(num_classes):
        count_c = (labels_np == c).sum()
        if count_c >= min_per_class:
            present_classes.append(c)

    if not present_classes:
        print("No classes with at least", min_per_class, "examples.")
        return

    # Build a grid for the classes (axes grid)
    n_classes = len(present_classes)
    ncols = min(5, n_classes)
    nrows = math.ceil(n_classes / ncols)

    for layer_name, act in activations.items():
        # act has shape (N, num_units)
        num_units = act.shape[1]
        keep = min(num_units, max_neurons)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3 * ncols, 2.5 * nrows),
            squeeze=False,
        )

        for idx, digit in enumerate(present_classes):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]

            mask = labels_np == digit
            sub = act[mask, :keep].numpy()  # shape: (num_examples_of_digit, keep)

            if sub.size == 0:
                ax.axis("off")
                continue

            im = ax.imshow(sub, aspect="auto", interpolation="nearest")
            ax.set_title(f"digit = {digit}")
            ax.set_xlabel("unit")
            ax.set_ylabel("example")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide any unused axes
        for j in range(n_classes, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig.suptitle(f"Activations in {layer_name} by digit", y=1.02)
        fig.tight_layout()
        plt.show()


def visualize_activations_by_label(
    model: torch.nn.Module,
    loader,
    device: Optional[str] = None,
    max_batches: int = 3,
    max_neurons: int = 32,
    num_classes: int = 10,
    min_per_class: int = 1,
) -> None:
    """
    Sort a few batches by digit identity and show layer activations per digit.

    Steps:
      1. Take up to `max_batches` from the DataLoader.
      2. Concatenate them into one big batch (X, y).
      3. Run the entire batch through `model` and collect activations
         from fc1, fc2, fc3.
      4. For each layer, make a grid of heatmaps, one per digit.

    The conditioning is on the *true labels* `y`.

    Args:
        model: trained DNN model.
        loader: DataLoader that yields (images, labels).
        device: torch device string or None for auto-select.
        max_batches: how many batches to pull from the loader.
        max_neurons: how many units to display per layer (columns).
        num_classes: how many distinct labels (e.g., 10 for digits 0–9).
        min_per_class: minimum examples required to show a digit.
    """
    # Collect a few batches
    xs, ys = [], []
    for i, (xb, yb) in enumerate(loader):
        xs.append(xb)
        ys.append(yb)
        if i + 1 >= max_batches:
            break

    if not xs:
        print("Loader appears to be empty.")
        return

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    activations = collect_activations_on_batch(model, X, device=device)

    print("Layers captured:", list(activations.keys()))
    print("Total examples sampled:", X.shape[0])
    _plot_label_heatmaps(
        activations,
        y,
        max_neurons=max_neurons,
        num_classes=num_classes,
        min_per_class=min_per_class,
    )
