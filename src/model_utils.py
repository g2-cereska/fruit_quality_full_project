"""Model utilities for training and inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torchvision import models


class FruitQualityNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = FruitQualityNet(num_classes=num_classes)
    return model.to(device)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    class_names: List[str],
    history: Dict[str, List[float]],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "history": history,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Tuple[nn.Module, List[str], Dict[str, List[float]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    history = checkpoint.get("history", {})
    model = build_model(num_classes=len(class_names), device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names, history
