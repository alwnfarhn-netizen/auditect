"""
train.py  —  Model Training
============================
Melatih model PointNet + LSTM dari dataset yang dikumpulkan dengan collect.py.

Jalankan:
    python train.py --dataset dataset.npz --epochs 80
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model.pointnet import CLASSES
from model.temporal import build_pointnet_lstm, train_model


def main(dataset_path: str, epochs: int, batch_size: int, save_path: str):
    # ── Muat dataset ─────────────────────────────────────────────────────
    print(f"[INFO] Memuat: {dataset_path}")
    data = np.load(dataset_path)
    X, y = data["X"], data["y"]
    print(f"[INFO] X: {X.shape}  y: {y.shape}")
    print()
    print("Distribusi kelas:")
    for i, cls in enumerate(CLASSES):
        n = (y == i).sum()
        bar = "█" * (n // 5)
        print(f"  {cls:20s}: {n:4d}  {bar}")
    print()

    # ── Split ─────────────────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Class weights (menangani imbalanced dataset) ──────────────────────
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print(f"Class weights: {class_weight}\n")

    # ── Build & train ─────────────────────────────────────────────────────
    model = build_pointnet_lstm()
    model.summary()

    history = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
    )

    # ── Evaluasi akhir ────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*50}")
    print(f"Val Loss     : {loss:.4f}")
    print(f"Val Accuracy : {acc*100:.2f}%")
    print(f"Model tersimpan → {save_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training — AuDetect | Autism Behavior Detection System")
    parser.add_argument("--dataset",    default="dataset.npz")
    parser.add_argument("--epochs",     default=80,   type=int)
    parser.add_argument("--batch_size", default=16,   type=int)
    parser.add_argument("--save",       default="weights/stimming_model.keras")
    args = parser.parse_args()
    main(args.dataset, args.epochs, args.batch_size, args.save)
