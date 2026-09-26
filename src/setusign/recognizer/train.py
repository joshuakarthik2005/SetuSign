"""
Training and Evaluation Pipeline for INCLUDE-50
================================================
Trains baseline models on extracted keypoints, reports real metrics.
Every number comes from a real run — no mocks, no hardcoded values.

Usage:
    python -m setusign.recognizer.train --data-dir data/keypoints
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.setusign.recognizer.models import build_model
from src.setusign.recognizer.dataset import get_dataloaders


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(data, mask=mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits.detach(), dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=50):
    """Evaluate model. Returns dict with loss, top1, top5 accuracy."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)

        logits = model(data, mask=mask)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.argmax(all_probs, axis=1)

    avg_loss = total_loss / max(num_batches, 1)
    top1 = accuracy_score(all_labels, all_preds)

    # Top-5 accuracy
    k = min(5, num_classes)
    try:
        top5 = top_k_accuracy_score(all_labels, all_probs, k=k,
                                     labels=list(range(num_classes)))
    except ValueError:
        top5 = 0.0

    return {
        "loss": avg_loss,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "num_samples": len(all_labels),
    }


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def main():
    parser = argparse.ArgumentParser(description="Train INCLUDE-50 baseline")
    parser.add_argument("--data-dir", default="data/keypoints",
                        help="Keypoints directory with train/val/test subdirs")
    parser.add_argument("--label-map", default="data/label_maps/label_map_include50.json")
    parser.add_argument("--model", default="transformer",
                        choices=["transformer", "lstm"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="models/baseline")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load label map
    with open(args.label_map) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Classes: {num_classes}")

    # Create dataloaders
    print("\nLoading data...")
    loaders = get_dataloaders(
        args.data_dir, args.label_map,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=0,
    )

    if "train" not in loaders:
        print("ERROR: No training data found!")
        sys.exit(1)

    # Build model
    model_kwargs = {}
    if args.model == "transformer":
        model_kwargs = {
            "d_model": args.d_model,
            "nhead": 8,
            "num_layers": args.num_layers,
            "dim_feedforward": args.d_model * 2,
            "dropout": 0.1,
            "max_len": args.max_len,
        }

    model = build_model(
        model_type=args.model,
        input_size=150,
        num_classes=num_classes,
        **model_kwargs,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stop = EarlyStopping(patience=args.patience)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    history = []

    print(f"\n{'='*70}")
    print(f"Training for {args.epochs} epochs (patience={args.patience})")
    print(f"{'='*70}")

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, loaders["train"], optimizer, device)

        # Validate
        val_metrics = {"loss": 0, "top1_accuracy": 0, "top5_accuracy": 0}
        if "val" in loaders:
            val_metrics = evaluate(model, loaders["val"], device, num_classes)
        elif "test" in loaders:
            val_metrics = evaluate(model, loaders["test"], device, num_classes)

        scheduler.step()
        elapsed = time.time() - t_epoch

        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_metrics['loss']:.4f} "
              f"top1={val_metrics['top1_accuracy']:.4f} "
              f"top5={val_metrics['top5_accuracy']:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        })

        # Save best model
        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["top1_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": best_val_acc,
                "val_top5": val_metrics["top5_accuracy"],
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_model.pt"))

        # Early stopping
        early_stop(val_metrics["top1_accuracy"])
        if early_stop.should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.time() - t_start

    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")

    # Load best model
    ckpt = torch.load(os.path.join(args.save_dir, "best_model.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model from epoch {ckpt['epoch']}")

    results = {}
    for split_name, loader in loaders.items():
        metrics = evaluate(model, loader, device, num_classes)
        results[split_name] = metrics
        print(f"  {split_name:5s}: top1={metrics['top1_accuracy']:.4f} "
              f"top5={metrics['top5_accuracy']:.4f} "
              f"loss={metrics['loss']:.4f} "
              f"(n={metrics['num_samples']})")

    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": ckpt["epoch"],
        "total_training_time_s": total_time,
        "device": str(device),
        "args": vars(args),
        "results": results,
        "history": history,
    }

    report_path = os.path.join(args.save_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    # Export to ONNX for future NPU deployment
    print("\nExporting to ONNX...")
    try:
        model.eval()
        dummy_input = torch.randn(1, args.max_len, 150).to(device)
        dummy_mask = torch.ones(1, args.max_len).to(device)
        onnx_path = os.path.join(args.save_dir, "model.onnx")
        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            onnx_path,
            input_names=["keypoints", "mask"],
            output_names=["logits"],
            dynamic_axes={
                "keypoints": {0: "batch", 1: "seq_len"},
                "mask": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"ONNX exported to: {onnx_path}")
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX model size: {onnx_size_mb:.1f} MB")
    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    main()
