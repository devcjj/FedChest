"""Flower client for private, hospital-local FedChest training."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import get_hospital_dataset
from model import IMAGE_SIZE, NUM_CLASSES, build_model, trainable_parameters

BATCH_SIZE = 4
LOCAL_EPOCHS = 1
LEARNING_RATE = 1e-3


class ChestXRayClient(fl.client.NumPyClient):
    def __init__(self, node_dir: str | Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model().to(self.device)
        self.dataset = get_hospital_dataset(node_dir)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(trainable_parameters(self.model), lr=LEARNING_RATE)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        del config
        return [
            parameter.detach().cpu().numpy().copy()
            for parameter in self._trainable_parameters()
        ]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        trainable = self._trainable_parameters()
        if len(parameters) != len(trainable):
            raise ValueError(
                f"Expected {len(trainable)} tensors, received {len(parameters)}"
            )
        for parameter, value in zip(trainable, parameters):
            tensor = torch.from_numpy(np.asarray(value)).to(
                device=parameter.device, dtype=parameter.dtype
            )
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Parameter shape mismatch: expected {tuple(parameter.shape)}, "
                    f"received {tuple(tensor.shape)}"
                )
            parameter.data.copy_(tensor)

    def fit(
        self, parameters: Sequence[np.ndarray], config: dict
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        self.set_parameters(parameters)
        self.model.train()
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        last_loss = 0.0

        for _ in range(LOCAL_EPOCHS):
            for images, metadata, targets in loader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    try:
                        outputs = self.model(images, metadata)
                    except TypeError:
                        outputs = self.model(images)
                    probabilities = outputs["probabilities"] if isinstance(outputs, dict) else outputs
                    probabilities = probabilities[:, : targets.shape[1]]
                    if probabilities.min().item() < 0 or probabilities.max().item() > 1:
                        probabilities = torch.sigmoid(probabilities)
                    loss = self.loss_fn(probabilities, targets)
                last_loss = float(loss.detach().cpu())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return self.get_parameters(config), len(self.dataset), {"loss": last_loss}

    def evaluate(
        self, parameters: Sequence[np.ndarray], config: dict
    ) -> tuple[float, int, dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, num_workers=0)
        total_loss = 0.0
        with torch.no_grad():
            for images, metadata, targets in loader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = targets.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    try:
                        outputs = self.model(images, metadata)
                    except TypeError:
                        outputs = self.model(images)
                    probabilities = outputs["probabilities"] if isinstance(outputs, dict) else outputs
                    probabilities = probabilities[:, : targets.shape[1]]
                    if probabilities.min().item() < 0 or probabilities.max().item() > 1:
                        probabilities = torch.sigmoid(probabilities)
                    total_loss += float(self.loss_fn(probabilities, targets).cpu())
        return total_loss / len(loader), len(self.dataset), {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-dir", type=Path, default=None)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    args = parser.parse_args()
    node_dir = args.node_dir or os.environ.get("HOSPITAL_NODE_DIR")
    if not node_dir:
        raise SystemExit("Provide --node-dir or HOSPITAL_NODE_DIR")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=ChestXRayClient(node_dir),
    )
