"""Basic training entrypoint for Argus baseline experiments."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset import DeepfakeDataset, build_robust_augmentation
from src.models.baselines import CNNBaseline


def train(args: argparse.Namespace) -> None:
    """Train the baseline model on a real-vs-fake folder dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hook in your production dataset root with --data-dir.
    dataset = DeepfakeDataset(root_dir=args.data_dir, transform=build_robust_augmentation())
    if len(dataset) == 0:
        raise RuntimeError(
            "No samples found. Ensure --data-dir points to a non-empty dataset with 'real/' and 'fake/' folders."
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CNNBaseline(pretrained=True).to(device)

    # Future JEPA integration point:
    # 1) Load/download frozen JEPA encoder weights.
    # 2) Extract features and feed them to JepaLinearProbe instead of CNNBaseline.

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        samples_seen = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            samples_seen += batch_size

        epoch_loss = running_loss / samples_seen
        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss: {epoch_loss:.4f}")

        # Placeholder: add validation DataLoader/evaluation here.
        print("Validation step placeholder: compute metrics on a held-out set.")


def parse_args() -> argparse.Namespace:
    """Parse training script arguments."""
    parser = argparse.ArgumentParser(description="Train Argus CNN baseline.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset with real/ and fake/ folders.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
