from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_DIR / "training_history.csv"
OUTPUT_DIR = PROJECT_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

history = pd.read_csv(HISTORY_PATH)

# Plot accuracy
plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training vs Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

accuracy_path = OUTPUT_DIR / "accuracy_curve.png"
plt.savefig(accuracy_path, dpi=150)
plt.show()

# Plot loss
plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss",
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = OUTPUT_DIR / "loss_curve.png"
plt.savefig(loss_path, dpi=150)
plt.show()

print(f"Accuracy plot saved to: {accuracy_path}")
print(f"Loss plot saved to: {loss_path}")
