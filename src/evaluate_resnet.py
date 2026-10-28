
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torchvision import models
import torch.nn as nn

from dataset import create_dataloaders, IDX_TO_CLASS


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_resnet.pth"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_misclassified_image():
    image_path = (
        PROJECT_DIR
        / "data"
        / "NEU-CLS"
        / "NEU-CLS"
        / "In_240.bmp"
    )

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("L")

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title("In_240.bmp | Actual: Inclusion | Predicted: Pitted Surface")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load the saved best ResNet18 checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    # Re-create ResNet18 with the same six-class output layer.
    # No pretrained weights are needed because we load our checkpoint.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load the existing test split.
    _, _, test_loader = create_dataloaders(batch_size=32)

    all_labels = []
    all_predictions = []

    test_df = pd.read_csv(PROJECT_DIR / "data" / "splits" / "test.csv")

    # Evaluate without updating model weights.
    with torch.no_grad():
        for images, labels in test_loader:
            # Convert grayscale images to three identical channels.
            images = images.repeat(1, 3, 1, 1).to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Get test filenames in the same order as the test loader.
    all_filenames = test_df["filename"].tolist()

    assert len(all_filenames) == len(all_labels), (
        "Test filenames and predictions do not match."
    )

    # Print incorrect predictions.
    print("\nIncorrect Test Predictions")
    print("-" * 60)

    incorrect_count = 0

    for filename, true_label, predicted_label in zip(
        all_filenames, all_labels, all_predictions
    ):
        if true_label != predicted_label:
            incorrect_count += 1

            print(f"Filename:        {filename}")
            print(f"Actual class:    {IDX_TO_CLASS[true_label]}")
            print(f"Predicted class: {IDX_TO_CLASS[predicted_label]}")
            print("-" * 60)

    if incorrect_count == 0:
        print("No incorrect predictions.")
    print("DEBUG: Incorrect prediction section executed.")
    print(f"Total incorrect predictions: {incorrect_count}")

    # Calculate test accuracy.
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Best validation accuracy: "
        f"{checkpoint['val_accuracy']:.4f}"
    )
    print(f"Test images: {len(all_labels)}")
    print(f"\nResNet18 test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Print classification report.
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\nResNet18 Classification Report")
    print(report)

    # Save classification report.
    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_path = RESULTS_DIR / "resnet_classification_report.csv"

    pd.DataFrame(report_dict).transpose().to_csv(report_path)

    # Generate and save the confusion matrix.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(values_format="d")
    plt.title("ResNet18 - Test Confusion Matrix")
    plt.tight_layout()

    matrix_path = RESULTS_DIR / "resnet_confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.show()

    print(f"\nConfusion matrix saved to: {matrix_path}")
    print(f"Classification report saved to: {report_path}")
    display_misclassified_image()

if __name__ == "__main__":
    main()
