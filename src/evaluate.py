from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import create_dataloaders
from model import SteelDefectCNN


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_cnn.pth"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load the saved best model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = SteelDefectCNN(num_classes=6).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create the data loaders.
    _, _, test_loader = create_dataloaders(batch_size=32)

    all_labels = []
    all_predictions = []

    # Evaluate without calculating gradients.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics.
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"Test images: {len(all_labels)}")
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(6)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report")
    print(report)

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(6)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    # Generate and save the confusion matrix.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(6)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Custom CNN - Test Confusion Matrix")
    plt.tight_layout()

    matrix_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.show()

    print(f"\nConfusion matrix saved to: {matrix_path}")
    print(
        "Classification report saved to: "
        f"{RESULTS_DIR / 'classification_report.csv'}"
    )


if __name__ == "__main__":
    main()