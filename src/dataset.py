from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "data" / "splits"

# Consistent class-to-number mapping
CLASS_TO_IDX = {
    "Cr": 0,
    "In": 1,
    "Pa": 2,
    "PS": 3,
    "RS": 4,
    "Sc": 5,
}

IDX_TO_CLASS = {
    index: name for name, index in CLASS_TO_IDX.items()
}

# Preprocessing for the custom CNN
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])


class SteelDefectDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        required_columns = {"filepath", "label"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"CSV must contain columns: {required_columns}"
            )

        unknown_labels = set(self.data["label"]) - set(CLASS_TO_IDX)
        if unknown_labels:
            raise ValueError(f"Unknown labels: {unknown_labels}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["filepath"])

        # Support both absolute paths and project-relative paths.
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        # If an old CSV contains a path from another computer,
        # recover the image using its filename and the dataset directory.
        if not image_path.exists():
            image_path = PROJECT_DIR / "data" / "NEU-CLS" / "NEU-CLS" / Path(row["filename"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}. "
                "Check the dataset location and CSV file paths."
            )

        label_name = row["label"]

        with Image.open(image_path) as image:
            image = image.convert("L")

            if self.transform:
                image = self.transform(image)

        label = CLASS_TO_IDX[label_name]

        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(batch_size=32):
    train_dataset = SteelDefectDataset(
        SPLIT_DIR / "train.csv",
        transform=transform,
    )

    val_dataset = SteelDefectDataset(
        SPLIT_DIR / "validation.csv",
        transform=transform,
    )

    test_dataset = SteelDefectDataset(
        SPLIT_DIR / "test.csv",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print("Dataset loading verification")
    print("-" * 40)

    print(f"Training images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images:       {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))

    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image data type:   {images.dtype}")
    print(f"Label data type:   {labels.dtype}")
    print(f"Pixel range:       {images.min().item():.3f} to "
          f"{images.max().item():.3f}")

    print("\nClass mapping:")
    for name, index in CLASS_TO_IDX.items():
        print(f"{index}: {name}")

    print("\nData loading verification successful!")