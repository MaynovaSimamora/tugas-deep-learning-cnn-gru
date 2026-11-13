"""Arsitektur 1: CNN untuk klasifikasi digit MNIST.

Contoh:
    python src/train_cnn.py --epochs 2
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from common import (
    DATA_DIR,
    RESULTS_DIR,
    collect_predictions,
    confusion_matrix,
    get_device,
    per_class_report,
    plot_confusion_matrix,
    plot_curves,
    save_json,
    set_seed,
)

CLASS_NAMES = [str(i) for i in range(10)]


class CNN(nn.Module):
    """Dua blok konvolusi + classifier MLP."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_loaders(batch_size: int, seed: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    val_size = 5000
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0),
        DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0),
    )


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def save_sample_predictions(model, loader, device, out_path):
    model.eval()
    x, y = next(iter(loader))
    x_dev = x[:16].to(device)
    with torch.no_grad():
        pred = model(x_dev).argmax(1).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, img, actual, guess in zip(axes.ravel(), x[:16], y[:16], pred):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(
            f"asli {int(actual)} / pred {int(guess)}",
            fontsize=9,
            color="green" if int(actual) == int(guess) else "red",
        )
        ax.axis("off")
    fig.suptitle("Contoh Prediksi CNN pada Data Uji")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN untuk MNIST")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_loaders(args.batch_size, args.seed)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred, 10)
    report = per_class_report(cm, CLASS_NAMES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(history, RESULTS_DIR / "cnn_curves.png", "CNN - MNIST")
    plot_confusion_matrix(
        cm, CLASS_NAMES, RESULTS_DIR / "cnn_confusion_matrix.png", "CNN - Confusion Matrix (Test)"
    )
    save_sample_predictions(model, test_loader, device, RESULTS_DIR / "cnn_predictions.png")

    payload = {
        "model": "CNN",
        "dataset": "MNIST",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "train_seconds": round(time.time() - started, 1),
        "history": history,
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }
    save_json(payload, RESULTS_DIR / "cnn_metrics.json")

    print(f"\nAkurasi test CNN: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Metrik & grafik tersimpan di: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
