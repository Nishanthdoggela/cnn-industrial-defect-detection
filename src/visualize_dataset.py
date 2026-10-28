from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/NEU-CLS/NEU-CLS")

# Defect classes and their meanings
classes = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted Surface",
    "RS": "Rolled-in Scale",
    "Sc": "Scratches",
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (class_code, class_name) in zip(axes.flat, classes.items()):

    # Find the first image belonging to this class
    image_paths = sorted(DATA_DIR.glob(f"{class_code}_*.bmp"))

    if not image_paths:
        ax.set_title(f"{class_name} (image not found)")
        ax.axis("off")
        continue

    image_path = image_paths[0]

    with Image.open(image_path) as image:
        ax.imshow(image.copy(), cmap="gray")

    ax.set_title(f"{class_name}\n{image_path.name}")
    ax.axis("off")

plt.suptitle("NEU-CLS Steel Surface Defect Dataset", fontsize=15)
plt.tight_layout()
plt.show()