from pathlib import Path
from itertools import combinations

import imagehash
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "data" / "splits"

SPLITS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

# A smaller distance means greater perceptual similarity.
# This threshold identifies candidates for visual inspection.
HAMMING_THRESHOLD = 5

records = []

print("Calculating perceptual hashes...")

for split_name, csv_path in SPLITS.items():
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        image_path = Path(row["filepath"])

        with Image.open(image_path) as image:
            perceptual_hash = imagehash.phash(image.convert("L"))

        records.append({
            "split": split_name,
            "filepath": str(image_path),
            "label": row["label"],
            "hash": perceptual_hash,
        })

print(f"Images processed: {len(records)}")
print()

# Compare images across different splits only.
for split_a, split_b in combinations(SPLITS.keys(), 2):
    images_a = [r for r in records if r["split"] == split_a]
    images_b = [r for r in records if r["split"] == split_b]

    matches = []

    for a in images_a:
        for b in images_b:
            distance = a["hash"] - b["hash"]

            if distance <= HAMMING_THRESHOLD:
                matches.append({
                    "distance": distance,
                    "file_a": a["filepath"],
                    "label_a": a["label"],
                    "file_b": b["filepath"],
                    "label_b": b["label"],
                })

    matches.sort(key=lambda item: item["distance"])

    print(f"{split_a.upper()} vs {split_b.upper()}")
    print(f"Potential near-duplicate pairs: {len(matches)}")

    for match in matches[:20]:
        print(
            f"Distance={match['distance']} | "
            f"{match['file_a']} [{match['label_a']}] | "
            f"{match['file_b']} [{match['label_b']}]"
        )

    if len(matches) > 20:
        print(f"...and {len(matches) - 20} additional pairs")

    print("-" * 70)