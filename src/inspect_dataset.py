from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/NEU-CLS/NEU-CLS")

class_counts = Counter()

for path in DATA_DIR.iterdir():
    if not path.is_file():
        continue

    if path.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png"}:
        continue

    # Extract the class prefix from filenames such as Cr_1.bmp.
    class_name = path.stem.split("_")[0]
    class_counts[class_name] += 1

print("Dataset class distribution")
print("-" * 35)

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count} images")

print("-" * 35)
print(f"Total images: {sum(class_counts.values())}")
print(f"Total classes: {len(class_counts)}")