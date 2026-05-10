"""Dataset and augmentation utilities for AI image detection."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


def _random_jpeg_compression(image: Image.Image, quality_range: tuple[int, int]) -> Image.Image:
    """Apply random JPEG compression to simulate platform recompression."""
    min_quality, max_quality = quality_range
    quality = random.randint(min_quality, max_quality)

    with io.BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


def default_image_transform(image_size: int = 224) -> transforms.Compose:
    """Return a standard evaluation transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_robust_augmentation(
    image_size: int = 224,
    jpeg_quality_range: tuple[int, int] = (35, 95),
) -> transforms.Compose:
    """Return augmentation transforms that mimic social media degradation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomApply(
                [transforms.Lambda(lambda img: _random_jpeg_compression(img, jpeg_quality_range))], p=0.7
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DeepfakeDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset for loading real and AI-generated images from folder labels."""

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or default_image_transform()
        self.valid_extensions = tuple(ext.lower() for ext in valid_extensions)
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[tuple[Path, float]]:
        """Collect `(image_path, label)` records from `real/` and `fake/` folders."""
        samples: list[tuple[Path, float]] = []
        class_to_label = {"real": 0.0, "fake": 1.0}

        for class_name, label in class_to_label.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists() or not class_dir.is_dir():
                continue

            for image_path in class_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in self.valid_extensions:
                    samples.append((image_path, label))

        if not samples:
            raise ValueError(
                "No images were found. Ensure the dataset has 'real' and 'fake' subfolders with image files."
            )

        return samples

    def __len__(self) -> int:
        """Return number of image samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return transformed image tensor and binary label tensor."""
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, label_tensor
