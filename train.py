"""Train local hospital models or orchestrate multi-hospital federated learning."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import LABEL_NAMES, get_hospital_dataset
from model import build_model, trainable_parameters


def mean_auc(targets: np.ndarray, predictions: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        scores = [roc_auc_score(targets[:, column], predictions[:, column]) for column in range(targets.shape[1]) if np.unique(targets[:, column]).size > 1]
        return float(np.mean(scores)) if scores else float("nan")
    except ImportError:
        acc = float(((predictions >= 0.5) == (targets >= 0.5)).mean())
        return acc


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device, scaler: torch.amp.GradScaler, accumulation_steps: int) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets, predictions = [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, metadata, labels) in enumerate(loader):
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
        if training:
            scaler.scale(loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * labels.size(0)
        targets.append(labels.detach().cpu().numpy())
        predictions.append(outputs.detach().float().cpu().numpy())
    target_array, prediction_array = np.concatenate(targets), np.concatenate(predictions)
    return total_loss / len(loader.dataset), mean_auc(target_array, prediction_array)


def train_local(node_dir: Path, epochs: int, batch_size: int, accumulation_steps: int, seed: int) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_hospital_dataset(node_dir)
    if len(dataset) < 2:
        raise ValueError(f"Hospital node at {node_dir} requires at least two samples")

    split_at = max(1, int(len(dataset) * 0.8))
    train_indices = list(range(split_at))
    val_indices = list(range(split_at, len(dataset))) if split_at < len(dataset) else train_indices

    train_set, validation_set = Subset(dataset, train_indices), Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    print(f"--- Training Local Model on {node_dir.name} ({len(dataset)} samples) ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        with torch.no_grad():
            val_loss, val_auc = run_epoch(model, validation_loader, criterion, None, device, scaler, accumulation_steps)
        print(f"epoch={epoch:03d} loss={train_loss:.5f} train_auc={train_auc:.5f} val_loss={val_loss:.5f} val_auc={val_auc:.5f}")

    checkpoint = node_dir / "checkpoints" / "local_fedchest_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed, "labels": LABEL_NAMES}, checkpoint)
    return checkpoint


def run_federated_learning(root_dir: Path, rounds: int = 3) -> Path:
    """Launch FL Server & 3 Hospital Clients (hospital_nih, hospital_rsna, hospital_siim)."""
    print("================================================================")
    print(" Launching Multi-Hospital Federated Learning Simulation (3 Nodes)")
    print("================================================================")

    # 1. Start FL Server
    server_process = subprocess.Popen([sys.executable, str(root_dir / "server.py")])
    time.sleep(2)

    # 2. Start Hospital Clients
    nodes = ["hospital_nih", "hospital_rsna", "hospital_siim"]
    client_processes = []
    for node in nodes:
        node_path = root_dir / "data" / node
        print(f" -> Launching Flower Client for {node}...")
        proc = subprocess.Popen([sys.executable, str(root_dir / "client.py"), "--node-dir", str(node_path)])
        client_processes.append(proc)

    # 3. Wait for FL Server completion
    server_process.wait()
    for proc in client_processes:
        if proc.poll() is None:
            proc.terminate()

    ckpt = root_dir / "checkpoints" / "global_fedchest_model.pt"
    print(f"\n[FL Success] Aggregated global model saved to {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["local", "fl"], default="local")
    parser.add_argument("--node-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.root_dir.resolve()
    if args.mode == "fl":
        run_federated_learning(root)
    else:
        node_path = (args.node_dir or (root / "data" / "hospital_nih")).resolve()
        checkpoint = train_local(node_path, args.epochs, args.batch_size, args.accumulation_steps, args.seed)
        print(f"Saved local checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()