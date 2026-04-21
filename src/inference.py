"""Inference helpers for single-image prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from model_utils import load_checkpoint


class Predictor:
    def __init__(self, checkpoint_path: str | Path, image_size: int = 224) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names, self.history = load_checkpoint(checkpoint_path, self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_path: str | Path, top_k: int = 3) -> Dict[str, object]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        ranking = probabilities.argsort()[::-1][:top_k]
        top_predictions = [
            {
                "label": self.class_names[index],
                "probability": float(probabilities[index]),
            }
            for index in ranking
        ]
        best_label = top_predictions[0]["label"]
        if "__" in best_label:
            produce_name, condition = best_label.rsplit("__", 1)
        else:
            produce_name, condition = best_label.rsplit("_", 1)
        return {
            "best_label": best_label,
            "produce_name": produce_name,
            "condition": condition,
            "top_predictions": top_predictions,
        }
