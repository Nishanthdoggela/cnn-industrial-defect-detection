from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.transforms import functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import create_dataloaders


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_resnet.pth"
TEST_CSV = PROJECT_DIR / "data" / "splits" / "test.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

CLASS_DESCRIPTIONS = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted surface",
    "RS": "Rolled-in scale",
    "Sc": "Scratches",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    # Re-create the same ResNet18 architecture used during training.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ResNet18's final convolutional block is a suitable Grad-CAM target.
    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )

    test_df = pd.read_csv(TEST_CSV)

    # Select a small, reproducible sample of test images.
    # This does not use the test set to tune the model.
    sample_df = test_df.sample(
        n=min(12, len(test_df)),
        random_state=42,
    ).reset_index(drop=True)

    summary = []

    for index, row in sample_df.iterrows():
        image_path = PROJECT_DIR / str(row["filepath"])
        true_label = str(row["label"])

        # Load the original grayscale image.
        original_image = Image.open(image_path).convert("L").resize(
            (200, 200)
        )

        # Grad-CAM visualization uses RGB, so repeat the grayscale channel.
        rgb_image = original_image.convert("RGB")

        # Convert image to a tensor: [1, 3, 200, 200].
        input_tensor = TF.to_tensor(rgb_image).unsqueeze(0).to(device)

        # Get the model's prediction.
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_idx = int(probabilities.argmax().item())

        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx].item())

        # Generate the Grad-CAM heatmap for the predicted class.
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,
        )[0]

        rgb_array = np.asarray(rgb_image).astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            rgb_array,
            grayscale_cam,
            use_rgb=True,
        )

        correct = predicted_label == true_label

        # Save one original-image / heatmap comparison.
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        axes[0].imshow(rgb_array, cmap="gray")
        axes[0].set_title(
            f"Original\nActual: {true_label} "
            f"({CLASS_DESCRIPTIONS[true_label]})"
        )
        axes[0].axis("off")

        axes[1].imshow(visualization)
        axes[1].set_title(
            f"Grad-CAM\nPredicted: {predicted_label}\n"
            f"Confidence: {confidence:.1%}"
        )
        axes[1].axis("off")

        status = "Correct" if correct else "Incorrect"
        fig.suptitle(f"Test sample {index + 1}: {status}")
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"gradcam_{index + 1:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary.append(
            {
                "image": str(row["filepath"]),
                "actual_class": true_label,
                "predicted_class": predicted_label,
                "confidence": confidence,
                "correct": correct,
                "visualization": str(output_path.relative_to(PROJECT_DIR)),
            }
        )

        print(
            f"[{index + 1:02d}/{len(sample_df)}] "
            f"Actual: {true_label} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.2%} | "
            f"{status}"
        )

    # Save prediction details alongside the visualizations.
    summary_path = OUTPUT_DIR / "gradcam_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print("\nGrad-CAM generation completed.")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()