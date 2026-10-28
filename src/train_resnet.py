
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from torchvision.models import ResNet18_Weights

from dataset import create_dataloaders


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_resnet.pth"
HISTORY_PATH = PROJECT_DIR / "resnet_training_history.csv"

NUM_CLASSES = 6
EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the existing dataset splits.
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Epochs: {EPOCHS}")

    # Load ResNet18 with pretrained ImageNet weights.
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace the original 1000-class output layer with six classes.
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model = model.to(device)

    # CrossEntropyLoss expects raw logits.
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_accuracy = 0.0
    history = []

    for epoch in range(EPOCHS):
        # ---------------- Training ----------------
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            # Convert grayscale [B, 1, H, W] to RGB-shaped
            # [B, 3, H, W] by repeating the same channel.
            images = images.repeat(1, 3, 1, 1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += batch_size

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        # ---------------- Validation ----------------
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.repeat(1, 3, 1, 1).to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size

                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += batch_size

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        # Save the best model based on validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "class_names": ["Cr", "In", "Pa", "PS", "RS", "Sc"],
                },
                CHECKPOINT_PATH,
            )

            print(
                f"  Saved best model: "
                f"validation accuracy = {best_val_accuracy:.4f}"
            )

    # Save training history.
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print("\nResNet18 training completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()