"""Arsitektur 2: GRU (keluarga RNN) untuk klasifikasi teks SMS spam.

Dataset: SMS Spam Collection (UCI), diunduh otomatis dari mirror GitHub.
Contoh:
    python src/train_gru.py --epochs 6
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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

DATA_URLS = [
    # Mirror CDN (cepat & stabil)
    "https://cdn.jsdelivr.net/gh/justmarkham/pycon-2016-tutorial@master/data/sms.tsv",
    # Sumber asli
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
]
CLASS_NAMES = ["ham", "spam"]
TOKEN_RE = re.compile(r"[a-z0-9']+")


def download_dataset(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            print(f"Mengunduh dataset dari {url} ...")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  gagal: {exc}")
    raise RuntimeError(
        "Gagal mengunduh dataset. Unduh manual file sms.tsv dari\n"
        f"  {DATA_URLS[0]}\n"
        f"lalu simpan ke {path} dan jalankan ulang perintahnya."
    ) from last_error


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], dtype=str)
    df = df[df["label"].isin(CLASS_NAMES)].dropna().drop_duplicates(subset=["message"])
    df["label_id"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(token_lists: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    freq: dict[str, int] = {}
    for tokens in token_lists:
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
    words = sorted(w for w, c in freq.items() if c >= min_freq)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word in words:
        vocab[word] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_len]]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class SMSDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def stratified_split(labels: np.ndarray, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        n_val = max(1, int(round(len(idx) * val_ratio)))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        _, hidden = self.gru(emb)  # hidden: (1, batch, hidden_dim)
        return self.fc(self.dropout(hidden[-1]))


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU untuk klasifikasi SMS spam")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=str(DATA_DIR / "sms.tsv"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    df = load_dataframe(download_dataset(Path(args.data)))
    print(f"Total data: {len(df)} | ham={int((df.label_id == 0).sum())} spam={int((df.label_id == 1).sum())}")

    texts = df["message"].tolist()
    labels = df["label_id"].to_numpy()
    tokens = [tokenize(t) for t in texts]

    train_idx, val_idx, test_idx = stratified_split(labels, 0.1, 0.1, args.seed)
    vocab = build_vocab([tokens[i] for i in train_idx])
    print(f"Ukuran vocab: {len(vocab)}")

    sequences = np.array([encode(t, vocab, args.max_len) for t in tokens], dtype=np.int64)

    train_loader = DataLoader(SMSDataset(sequences[train_idx], labels[train_idx]),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SMSDataset(sequences[val_idx], labels[val_idx]),
                            batch_size=256, shuffle=False)
    test_loader = DataLoader(SMSDataset(sequences[test_idx], labels[test_idx]),
                             batch_size=256, shuffle=False)

    model = GRUClassifier(len(vocab)).to(device)
    counts = np.bincount(labels[train_idx], minlength=2)
    class_weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    cm = confusion_matrix(y_true, y_pred, 2)
    report = per_class_report(cm, CLASS_NAMES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(history, RESULTS_DIR / "gru_curves.png", "GRU - SMS Spam")
    plot_confusion_matrix(
        cm, CLASS_NAMES, RESULTS_DIR / "gru_confusion_matrix.png", "GRU - Confusion Matrix (Test)"
    )

    examples = []
    for i in np.where(y_true != y_pred)[0][:5]:
        examples.append(
            {
                "text": texts[test_idx[i]][:160],
                "actual": CLASS_NAMES[int(y_true[i])],
                "predicted": CLASS_NAMES[int(y_pred[i])],
            }
        )

    payload = {
        "model": "GRU",
        "dataset": "SMS Spam Collection (UCI)",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "vocab_size": len(vocab),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "train_seconds": round(time.time() - started, 1),
        "history": history,
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "misclassified_examples": examples,
    }
    save_json(payload, RESULTS_DIR / "gru_metrics.json")

    print(f"\nAkurasi test GRU: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    for row in report:
        print(
            f"  {row['class']:>4}: precision {row['precision']:.3f} "
            f"recall {row['recall']:.3f} f1 {row['f1']:.3f}"
        )
    print(f"Metrik & grafik tersimpan di: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
