from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "NEU-CLS" / "NEU-CLS"
OUTPUT_DIR = PROJECT_DIR / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}

EXPECTED_CLASSES = {"Cr", "In", "Pa", "PS", "RS", "Sc"}

records = []

print("Scanning dataset and calculating image hashes...")

for path in sorted(DATA_DIR.iterdir()):
    if not path.is_file():
        continue

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    label = path.stem.split("_")[0]

    # Hash the original file contents.
    with open(path, "rb") as image_file:
        image_hash = hashlib.sha256(image_file.read()).hexdigest()

    records.append({
        "filepath": str(path.resolve()),
        "filename": path.name,
        "label": label,
        "image_hash": image_hash,
    })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(f"No images found in {DATA_DIR}")

actual_classes = set(df["label"].unique())

if actual_classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Unexpected classes. Found: {sorted(actual_classes)}"
    )

# Exact duplicate images must have the same label.
label_counts = df.groupby("image_hash")["label"].nunique()

if (label_counts > 1).any():
    raise ValueError(
        "Identical image content has conflicting class labels. "
        "Inspect those images before splitting."
    )

# Assign one group ID to each unique image content.
df["group_id"] = pd.factorize(df["image_hash"])[0]

# Each duplicate group has one class label.
groups = (
    df.groupby("group_id", as_index=False)
    .agg(label=("label", "first"))
)

print(f"Total image files: {len(df)}")
print(f"Unique image contents: {len(groups)}")
print(f"Exact duplicate files: {len(df) - len(groups)}")

# Split unique image groups, not individual files.
train_groups, temporary_groups = train_test_split(
    groups,
    test_size=0.30,
    random_state=42,
    stratify=groups["label"],
)

val_groups, test_groups = train_test_split(
    temporary_groups,
    test_size=0.50,
    random_state=42,
    stratify=temporary_groups["label"],
)

# Map every original image to the split of its duplicate group.
group_to_split = {}

for group_df, split_name in [
    (train_groups, "train"),
    (val_groups, "validation"),
    (test_groups, "test"),
]:
    for group_id in group_df["group_id"]:
        group_to_split[group_id] = split_name

df["split"] = df["group_id"].map(group_to_split)

# Keep the original CSV columns expected by the existing project.
for split_name in ["train", "validation", "test"]:
    split_df = df.loc[
        df["split"] == split_name,
        ["filepath", "filename", "label", "split"],
    ].copy()

    split_df.to_csv(
        OUTPUT_DIR / f"{split_name}.csv",
        index=False,
    )

    print(f"\n{split_name.capitalize()}: {len(split_df)} images")
    print(
        split_df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

# Verify that each original image is assigned exactly once.
assert len(df) == len(records)
assert df["split"].notna().all()
assert df["filename"].nunique() == len(df)

# Verify no identical image content crosses split boundaries.
hash_split_counts = df.groupby("image_hash")["split"].nunique()
assert (hash_split_counts == 1).all()

print("\n" + "=" * 50)
print("Dataset splitting completed successfully.")
print(f"Total image files: {len(df)}")
print("Exact duplicate groups stay within a single split.")
print(f"CSV files saved in: {OUTPUT_DIR}")