"""Frozen TorchXRayVision DenseNet with a compact convolutional LoRA fusion head."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn


LOCAL_XRV = Path(__file__).resolve().parent / "torchxrayvision-main"
if LOCAL_XRV.is_dir() and str(LOCAL_XRV) not in sys.path:
    sys.path.insert(0, str(LOCAL_XRV))
_xrv_model_path = LOCAL_XRV / "torchxrayvision" / "baseline_models" / "jfhealthcare" / "model"
if _xrv_model_path.is_dir():
    __path__ = [str(_xrv_model_path)]
import torchxrayvision as xrv


NUM_CLASSES = 14
IMAGE_SIZE = 224
LORA_RANK = 4
LORA_ALPHA = 8
MAX_TRAINABLE_PARAMETERS = 300_000


class LoRAConv2d(nn.Module):
    """A frozen convolution plus a trainable factorized 1x1 residual."""

    def __init__(self, base: nn.Conv2d, rank: int = LORA_RANK, alpha: int = LORA_ALPHA) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + self.up(self.down(inputs)) * self.scale


def _inject_lora(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child))
            count += 1
        else:
            count += _inject_lora(child)
    return count


class MultimodalChestModel(nn.Module):
    """DenseNet visual features fused with age, sex, and view metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if not hasattr(self.backbone, "features") or not hasattr(self.backbone.features, "denseblock4"):
            raise RuntimeError("Unexpected TorchXRayVision DenseNet structure")
        self.lora_layers = _inject_lora(self.backbone.features.denseblock4)
        if self.lora_layers == 0:
            raise RuntimeError("No convolutional layers found in DenseNet denseblock4")
        self.metadata_head = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(1040, NUM_CLASSES), nn.Sigmoid())
        trainable, _ = count_parameters(self)
        if trainable >= MAX_TRAINABLE_PARAMETERS:
            raise ValueError(f"Trainable parameter budget exceeded: {trainable} >= {MAX_TRAINABLE_PARAMETERS}")

    def forward(self, image: Tensor, metadata: Tensor) -> Tensor:
        visual = self.backbone.features(image)
        visual = torch.relu(visual)
        visual = torch.nn.functional.adaptive_avg_pool2d(visual, (1, 1)).flatten(1)
        if visual.shape[1] != 1024:
            raise RuntimeError(f"Expected 1024 visual features, got {visual.shape[1]}")
        fused = torch.cat((visual, self.metadata_head(metadata)), dim=1)
        return self.classifier(fused)


def build_model() -> MultimodalChestModel:
    return MultimodalChestModel()


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = build_model()
    trainable, total = count_parameters(model)
    print(f"trainable={trainable:,}, total={total:,}, lora_layers={model.lora_layers}")
