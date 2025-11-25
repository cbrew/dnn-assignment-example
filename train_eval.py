# train_eval.py
import torch
from torch import nn
from typing import List, Tuple


def train_model(
    model: torch.nn.Module,
    train_loader,
    dev_loader,
    num_epochs: int,
    lr: float,
    device: str = "cpu",
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """
    Train model and return:
    train_losses, train_accuracies, dev_losses, dev_accuracies
    """
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    tr_losses, tr_accs = [], []
    dev_losses, dev_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            num_correct += (preds == y_batch).sum().item()
            num_samples += X_batch.size(0)

        avg_loss = total_loss / num_samples
        accuracy = num_correct / num_samples

        tr_losses.append(avg_loss)
        tr_accs.append(accuracy)

        dev_loss, dev_acc = evaluate_model(model, dev_loader, loss_fn, device)
        dev_losses.append(dev_loss)
        dev_accs.append(dev_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train loss {avg_loss:.4f}, acc {accuracy:.4f} "
            f"- Dev loss {dev_loss:.4f}, acc {dev_acc:.4f}"
        )

    return tr_losses, tr_accs, dev_losses, dev_accs


def evaluate_model(model, data_loader, loss_fn, device="cpu"):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            num_correct += (preds == y_batch).sum().item()
            num_samples += X_batch.size(0)

    avg_loss = total_loss / num_samples
    accuracy = num_correct / num_samples

    return avg_loss, accuracy
