from pathlib import Path
import hashlib
import pandas as pd
from itertools import combinations

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "data" / "splits"

split_files = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

records = []

for split_name, csv_path in split_files.items():
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        image_path = Path(row["filepath"])

        with open(image_path, "rb") as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()

        records.append({
            "split": split_name,
            "filepath": str(image_path),
            "label": row["label"],
            "hash": image_hash,
        })

data = pd.DataFrame(records)

print(f"Total images checked: {len(data)}\n")

for split_a, split_b in combinations(
    ["train", "validation", "test"], 2
):
    a = data[data["split"] == split_a]
    b = data[data["split"] == split_b]

    matches = a.merge(
        b,
        on="hash",
        suffixes=("_a", "_b"),
    )

    print(f"{split_a.upper()} vs {split_b.upper()}")
    print(f"Exact duplicate pairs: {len(matches)}")

    if not matches.empty:
        print(
            matches[
                ["filepath_a", "filepath_b", "label_a", "label_b"]
            ].to_string(index=False)
        )

    print("-" * 60)