from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import create_dataloaders
from model import SteelDefectCNN


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_cnn.pth"


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    Train for one epoch when an optimizer is provided.
    Otherwise, evaluate without updating model parameters.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training" if training else "Validation",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    # Reproducibility
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=32
    )

    # Initialize model
    model = SteelDefectCNN(num_classes=6).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    epochs = 20
    best_val_accuracy = 0.0

    history = []

    print(f"\nTraining images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Epochs: {epochs}\n")

    for epoch in range(epochs):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save the checkpoint with the best validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "class_to_idx": {
                        "Cr": 0,
                        "In": 1,
                        "Pa": 2,
                        "PS": 3,
                        "RS": 4,
                        "Sc": 5,
                    },
                },
                BEST_MODEL_PATH,
            )

            print("  -> Best model checkpoint saved.")

    # Save training history for later plotting.
    import pandas as pd

    history_path = PROJECT_DIR / "training_history.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()