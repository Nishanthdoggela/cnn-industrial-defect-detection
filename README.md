# Industrial Surface Defect Detection Using Deep Learning

A computer vision project for classifying steel surface defects using a custom Convolutional Neural Network (CNN) and transfer learning with ResNet18. The project also includes Grad-CAM visualizations to help interpret model predictions and a Streamlit application for interactive image classification.

## Overview

Surface defects in manufactured steel can affect product quality and production efficiency. Manual inspection can be time-consuming and may vary between inspectors.

This project explores deep learning as an approach to automated steel surface defect classification. It trains and evaluates two image-classification models and provides a user interface for testing images and visualizing the model's attention.

**Best recorded result:** ResNet18 achieved 99.63% test accuracy on a test set of 270 images.

This result applies to the dataset and experimental setup used in this project. It should not be interpreted as a guarantee of performance on images from real industrial production lines.

## Features

- Six-class steel surface defect classification.
- Custom CNN implemented with PyTorch.
- Transfer learning using pretrained ResNet18.
- Training and validation loss and accuracy tracking.
- Test-set evaluation with precision, recall, F1-score, and confusion matrices.
- Grad-CAM visualizations for model interpretability.
- Streamlit interface for uploading and classifying images.
- Duplicate-aware dataset splitting to prevent identical image content from appearing across training, validation, and test sets.

## Dataset

The project uses the NEU-CLS steel surface defect classification dataset.

The dataset contains 1,800 grayscale images, with 300 images in each of six classes. The images are 200 × 200 pixels.

| Label | Defect category |
|---|---|
| `Cr` | Crazing |
| `In` | Inclusion |
| `Pa` | Patches |
| `PS` | Pitted surface |
| `RS` | Rolled-in scale |
| `Sc` | Scratches |

### Dataset split

After identifying one pair of byte-for-byte identical images, the splitting script was updated to keep identical image content in the same partition.

| Partition | Images |
|---|---:|
| Training | 1,259 |
| Validation | 271 |
| Test | 270 |
| **Total** | **1,800** |

The duplicate-aware split groups files by SHA-256 hash before assigning groups to partitions. This prevents exact duplicate images from crossing the training, validation, and test boundaries.

The dataset is not included in this repository. Obtain it from an appropriate dataset source, check its terms of use, and place the extracted images in the directory structure expected by `src/dataset.py` and `src/split_dataset.py`.

## Models

### 1. Custom CNN

The first model is a custom Convolutional Neural Network implemented using PyTorch.

It learns image features through convolutional layers and uses those features to predict one of the six defect classes.

The model provides a baseline for comparing a CNN trained specifically on the steel defect dataset against a model initialized with pretrained weights.

### 2. ResNet18 with transfer learning

The second model uses ResNet18 initialized with pretrained ImageNet weights.

The original classification layer is replaced with a fully connected layer that outputs six class scores. The model is then fine-tuned on the steel defect images.

The current training implementation uses:

- Architecture: ResNet18
- Initialization: pretrained ImageNet weights
- Optimizer: Adam
- Learning rate: `0.0001`
- Weight decay: `0.0001`
- Batch size: `32`
- Training duration: up to 20 epochs
- Loss function: Cross-entropy loss
- Model selection: best validation accuracy

Grayscale images are repeated across three channels to match ResNet18's expected input channel count.

The model checkpoint is saved as `checkpoints/best_resnet.pth`.

## Experimental Results

### Test-set performance

| Metric | Custom CNN | ResNet18 |
|---|---:|---:|
| Accuracy | 91.85% | **99.63%** |
| Macro precision | 92.08% | 99.64% |
| Macro recall | 91.85% | 99.63% |
| Macro F1-score | 91.91% | 99.63% |
| Correct predictions | 248 / 270 | 269 / 270 |

The custom CNN's recorded results came from an earlier dataset split. ResNet18 was retrained and evaluated using the duplicate-aware split. Because the split procedures differ, these figures are historical experiment results rather than a strictly controlled architecture-only comparison.

### ResNet18 per-class performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Crazing (`Cr`) | 100.00% | 100.00% | 100.00% |
| Inclusion (`In`) | 100.00% | 97.78% | 98.88% |
| Patches (`Pa`) | 100.00% | 100.00% | 100.00% |
| Pitted surface (`PS`) | 97.83% | 100.00% | 98.90% |
| Rolled-in scale (`RS`) | 100.00% | 100.00% | 100.00% |
| Scratches (`Sc`) | 100.00% | 100.00% | 100.00% |

The ResNet18 model made one test-set error: an inclusion image was classified as a pitted surface defect.

## Evaluation and Visualizations

The `results/` directory contains recorded evaluation artifacts, including:

- Classification reports in CSV format.
- Confusion matrices.
- Accuracy and loss plots.
- Grad-CAM visualizations for selected images.

These artifacts help assess the model's classification performance and inspect which image regions contribute to predictions.

### Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) produces a coarse heatmap showing image regions that influence a model's class prediction.

In this project, Grad-CAM is used to help interpret ResNet18 predictions. The heatmaps are diagnostic visualizations, not proof that the model uses physically meaningful defect features.

## Project Structure

```text
cnn-industrial-defect-detection/
├── app.py
├── checkpoints/
│   ├── best_cnn.pth
│   └── best_resnet.pth
├── data/
│   └── NEU-CLS/
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── classification_report.csv
│   ├── resnet_confusion_matrix.png
│   ├── resnet_classification_report.csv
│   └── gradcam visualizations
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_resnet.py
│   ├── evaluate_resnet.py
│   ├── gradcam_resnet.py
│   ├── plot_training.py
│   ├── split_dataset.py
│   ├── check_duplicates.py
│   ├── check_near_duplicates.py
│   ├── inspect_dataset.py
│   └── visualize_dataset.py
├── training_history.csv
├── resnet_training_history.csv
├── .gitignore
└── README.md
```

The exact contents of `results/` may change as additional experiments are run.

## Installation

### Prerequisites

- Python 3.12
- pip
- Git
- An NVIDIA GPU is optional; CPU execution may be slower.

### 1. Clone the repository

After publishing this project on GitHub, replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's actual clone URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd cnn-industrial-defect-detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your NVIDIA driver and CUDA environment. Consult the official PyTorch installation instructions if GPU support is not detected.

### 4. Download and prepare the dataset

The NEU-CLS image dataset is not included in this repository. Obtain the dataset from its legitimate source and extract the images into:

```text
data/
└── NEU-CLS/
    └── NEU-CLS/
        ├── Cr_1.bmp
        ├── In_1.bmp
        └── ...
```

The folder should contain the six defect classes: `Cr`, `In`, `Pa`, `PS`, `RS`, and `Sc`.

The repository includes the existing dataset split CSV files under `data/splits/`. These files contain relative image paths, so preserve the directory structure above when preparing the dataset.

**Note:** Generating new splits with `src/split_dataset.py` may change the existing train, validation, and test assignments. Use that script only when you intentionally want to regenerate the splits.

## Usage

Run the commands from the repository root with the virtual environment activated.

### Generate dataset splits

```powershell
python src/split_dataset.py
```

### Train the custom CNN

```powershell
python src/train.py
```

### Evaluate the custom CNN

```powershell
python src/evaluate.py
```

### Train ResNet18

```powershell
python src/train_resnet.py
```

### Evaluate ResNet18

```powershell
python src/evaluate_resnet.py
```

### Generate training plots

```powershell
python src/plot_training.py
```

### Generate Grad-CAM visualizations

```powershell
python src/gradcam_resnet.py
```

### Launch the Streamlit application

Install Streamlit if it is not already included in the project dependencies, then run:

```powershell
streamlit run app.py
```

**Note:** The scripts and application depend on the expected dataset paths and checkpoint files. Verify these paths against the current source code and repository contents before running commands on another machine.

## Limitations

- The dataset contains 1,800 images from six predefined defect categories.
- A small benchmark test set may not represent variation in real manufacturing environments.
- Performance may change with different cameras, illumination, surface finishes, defect sizes, and production conditions.
- The recorded ResNet18 result comes from one train/validation/test partition and is not an estimate of performance across all factories or steel products.
- Grad-CAM provides approximate visual explanations and does not establish causal reasoning.
- The CNN and ResNet18 results were generated under different split procedures and should not be treated as a controlled architecture-only comparison.
- Further validation on independent industrial images is needed before deployment.

## Future Improvements

- Compare models using identical, reproducible dataset partitions.
- Evaluate robustness to lighting, noise, image acquisition differences, and previously unseen samples.
- Add data augmentation and systematically tune hyperparameters.
- Measure inference latency and model size.
- Investigate calibration and confidence-based rejection of uncertain predictions.
- Validate on independently collected industrial images.
- Add automated tests and reproducible dependency management.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Streamlit
- Grad-CAM

## Disclaimer

This project is an educational computer vision experiment. Its reported benchmark performance should not be interpreted as a guarantee of production performance or as a substitute for validated industrial quality-control procedures.