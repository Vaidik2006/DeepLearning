"""Train a CNN on ESC-50 folds 1-3 and validate on fold 4."""
import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.audio_utils import waveform_to_logmel, SAMPLE_RATE
from src.model import AudioCNN

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "ESC-50"
MODEL_DIR = ROOT / "models"


class ESC50Dataset(Dataset):
    def __init__(self, frame, audio_dir):
        self.frame = frame.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.classes = sorted(frame["category"].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.audio_dir / row["filename"]
        waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=5.0)
        features = waveform_to_logmel(waveform)
        label = self.class_to_idx[row["category"]]
        return features, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metadata_path = DATASET / "meta" / "esc50.csv"
    audio_dir = DATASET / "audio"
    if not metadata_path.exists():
        raise SystemExit("Dataset not found. Run `python download_dataset.py` first.")

    metadata = pd.read_csv(metadata_path)
    train_frame = metadata[metadata["fold"].isin([1, 2, 3])]
    val_frame = metadata[metadata["fold"] == 4]
    classes = sorted(metadata["category"].unique())
    class_to_idx = {name: i for i, name in enumerate(classes)}

    class AudioDataset(Dataset):
        def __init__(self, frame):
            self.frame = frame.reset_index(drop=True)
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            waveform, _ = librosa.load(audio_dir / row["filename"], sr=SAMPLE_RATE, mono=True, duration=5.0)
            return waveform_to_logmel(waveform), torch.tensor(class_to_idx[row["category"]], dtype=torch.long)

    train_ds = AudioDataset(train_frame)
    val_ds = AudioDataset(val_frame)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = total_loss / max(total, 1)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={avg_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "num_classes": len(classes), "classes": classes},
                MODEL_DIR / "audio_cnn.pt",
            )
            (MODEL_DIR / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model: {MODEL_DIR / 'audio_cnn.pt'}")


if __name__ == "__main__":
    main()
