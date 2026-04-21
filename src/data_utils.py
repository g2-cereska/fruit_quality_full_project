"""Dataset utilities for the fruit quality project."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetSummary:
    root: str
    class_names: List[str]
    class_counts: Dict[str, int]
    total_images: int


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder with friendly error messages for corrupted files."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        try:
            return super().__getitem__(index)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"Failed to read image: {path}") from exc


def load_class_names(config_path: str | Path) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_dataset_structure(dataset_root: str | Path) -> DatasetSummary:
    dataset_root = str(dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}. "
            "Download the Kaggle dataset and point --data_dir to the folder that contains the 28 class folders."
        )

    class_counts: Dict[str, int] = {}
    total_images = 0
    class_names: List[str] = []

    for entry in sorted(os.listdir(dataset_root)):
        class_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(class_path):
            continue
        class_names.append(entry)
        count = 0
        for root, _, files in os.walk(class_path):
            count += sum(Path(file_name).suffix.lower() in VALID_EXTENSIONS for file_name in files)
        class_counts[entry] = count
        total_images += count

    if not class_names:
        raise ValueError(
            "No class folders were found. Expected 28 folders such as Apple__Fresh and Apple__Rotten."
        )

    return DatasetSummary(
        root=dataset_root,
        class_names=class_names,
        class_counts=class_counts,
        total_images=total_images,
    )


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset: torch.utils.data.Dataset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label


def make_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], DatasetSummary]:
    set_seed(seed)
    summary = validate_dataset_structure(data_dir)
    train_tfms, eval_tfms = build_transforms(image_size=image_size)

    raw_dataset = SafeImageFolder(data_dir, transform=None)
    total_size = len(raw_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = TransformSubset(train_subset, train_tfms)
    val_dataset = TransformSubset(val_subset, eval_tfms)
    test_dataset = TransformSubset(test_subset, eval_tfms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, raw_dataset.classes, summary
