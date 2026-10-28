from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models
from torchvision.transforms import functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==========================================================
# 1. Configuration
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_resnet.pth"

CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

CLASS_DETAILS = {
    "Cr": {
        "name": "Crazing",
        "description": (
            "Fine, interconnected cracks that create a network-like "
            "pattern on the steel surface."
        ),
    },
    "In": {
        "name": "Inclusion",
        "description": (
            "Foreign material or inclusions associated with irregular "
            "regions in the steel surface."
        ),
    },
    "Pa": {
        "name": "Patches",
        "description": (
            "Patch-like regions with a surface appearance that differs "
            "from the surrounding steel."
        ),
    },
    "PS": {
        "name": "Pitted Surface",
        "description": (
            "A surface containing small pits or depressions."
        ),
    },
    "RS": {
        "name": "Rolled-in Scale",
        "description": (
            "Scale-related surface irregularities associated with "
            "the steel rolling process."
        ),
    },
    "Sc": {
        "name": "Scratches",
        "description": (
            "Linear marks or grooves visible on the steel surface."
        ),
    },
}

# Results from the latest evaluation on the regenerated test split.
TEST_ACCURACY = 99.63
TEST_IMAGES = 270
CORRECT_PREDICTIONS = 269

st.set_page_config(
    page_title="SteelGuard | Defect Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 2. Dashboard styling
# ==========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    .hero-card {
        padding: 1.7rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(115deg, #172554, #1e3a5f);
        color: white;
        margin-bottom: 1.4rem;
    }

    .hero-card h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
    }

    .hero-card p {
        color: #dbeafe;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 0.7rem;
        margin-bottom: 0.3rem;
    }

    .subtle-text {
        color: #64748b;
        font-size: 0.92rem;
    }

    .prediction-card {
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        background: rgba(148, 163, 184, 0.08);
        margin: 0.5rem 0 1rem 0;
    }

    .prediction-label {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    .prediction-value {
        font-size: 1.7rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        padding: 0.8rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 3. Load the trained model
# ==========================================================

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, device, checkpoint


# ==========================================================
# 4. Prediction and Grad-CAM
# ==========================================================

def predict_image(model, device, image):
    # Preserve the existing model's grayscale preprocessing.
    grayscale_image = image.convert("L").resize((200, 200))
    rgb_image = grayscale_image.convert("RGB")

    input_tensor = TF.to_tensor(rgb_image).unsqueeze(0).to(device)

    # Predict the defect class.
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_idx = int(probabilities.argmax().item())
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx].item())

    # Generate Grad-CAM for the predicted class.
    target_layers = [model.layer4[-1]]

    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,
        )[0]

    rgb_array = np.asarray(rgb_image).astype(np.float32) / 255.0

    # Create the heatmap overlay.
    visualization = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True,
    )

    # Convert the grayscale CAM into a visible heatmap.
    heatmap = plt.get_cmap("jet")(grayscale_cam)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities.cpu().numpy(),
        "processed_image": rgb_array,
        "heatmap": heatmap,
        "overlay": visualization,
    }


# ==========================================================
# 5. Sidebar
# ==========================================================

with st.sidebar:
    st.title("🔬 SteelGuard")
    st.caption("Steel Surface Defect Intelligence")

    st.markdown("---")
    st.subheader("Model Overview")

    st.write("**Architecture:** ResNet18")
    st.write("**Learning approach:** Transfer learning")
    st.write("**Framework:** PyTorch")
    st.write("**Dataset:** NEU-CLS")
    st.write("**Input:** 200 × 200 grayscale image")
    st.write("**Output:** 6 defect categories")

    st.markdown("---")
    st.subheader("Benchmark Performance")

    st.metric(
        "Test accuracy",
        f"{TEST_ACCURACY:.2f}%",
    )

    st.caption(
        f"{CORRECT_PREDICTIONS} of {TEST_IMAGES} test images "
        "were classified correctly."
    )

    st.markdown("---")
    st.subheader("Defect Reference")

    for code in CLASS_NAMES:
        details = CLASS_DETAILS[code]

        with st.expander(f"{code} — {details['name']}"):
            st.write(details["description"])

    st.markdown("---")
    st.caption(
        "Research demonstration only. Not validated for "
        "safety-critical industrial decisions."
    )


# ==========================================================
# 6. Main dashboard
# ==========================================================

st.markdown(
    """
    <div class="hero-card">
        <h1>Steel Surface Defect Detection</h1>
        <p>
            AI-assisted classification and visual explanations
            for six categories of steel surface defects.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Defect Classes", "6")

with metric2:
    st.metric("Test Images", f"{TEST_IMAGES}")

with metric3:
    st.metric("Test Accuracy", f"{TEST_ACCURACY:.2f}%")

with metric4:
    st.metric(
        "Correct Test Predictions",
        f"{CORRECT_PREDICTIONS}/{TEST_IMAGES}",
    )

st.markdown("")
st.markdown(
    '<div class="section-heading">Inspect a Steel Surface</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtle-text">'
    "Upload an image to classify its surface pattern and inspect "
    "the regions highlighted by Grad-CAM."
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "This application was evaluated on the NEU-CLS benchmark. "
    "Performance on images from new cameras, production lines, "
    "lighting conditions, or steel types may differ."
)


# ==========================================================
# 7. Image upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Supported formats: PNG, JPG, JPEG, BMP.",
)

if uploaded_file is None:
    st.markdown("---")
    st.subheader("How to use the application")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 1. Upload")
        st.write(
            "Choose a steel surface image from your computer."
        )

    with step2:
        st.markdown("### 2. Analyze")
        st.write(
            "Run the trained ResNet18 model to obtain a prediction."
        )

    with step3:
        st.markdown("### 3. Explain")
        st.write(
            "Review class probabilities and the Grad-CAM overlay."
        )

else:
    try:
        image = Image.open(uploaded_file)
        image.load()

        original_image = image.copy()

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "The uploaded file could not be read as an image. "
            "Please try another PNG, JPG, JPEG, or BMP file."
        )
        st.stop()

    upload_col, details_col = st.columns([1.15, 1])

    with upload_col:
        st.subheader("Uploaded Image")
        st.image(
            original_image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with details_col:
        st.subheader("Image Information")

        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Original dimensions:** {image.size[0]} × {image.size[1]}")
        st.write(f"**Image mode:** {image.mode}")

        st.caption(
            "The model converts the image to grayscale and resizes "
            "it to 200 × 200 pixels before prediction."
        )

    analyze = st.button(
        "🔍 Analyze Defect",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        try:
            with st.spinner(
                "Loading model and generating prediction..."
            ):
                model, device, checkpoint = load_model()
                results = predict_image(model, device, original_image)

            st.session_state["analysis_results"] = results
            st.session_state["analysis_filename"] = uploaded_file.name

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The image could not be analyzed. "
                "Check the model checkpoint and installed dependencies."
            )
            st.exception(error)

    # Keep the most recent result visible during Streamlit reruns.
    results = st.session_state.get("analysis_results")
    result_filename = st.session_state.get("analysis_filename")

    if results is not None and result_filename == uploaded_file.name:
        predicted_class = results["predicted_class"]
        confidence = results["confidence"]
        details = CLASS_DETAILS[predicted_class]

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Prediction Results</div>',
            unsafe_allow_html=True,
        )

        result_col1, result_col2 = st.columns([1.3, 1])

        with result_col1:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted defect</div>
                    <div class="prediction-value">
                        {details['name']}
                    </div>
                    <div class="subtle-text">
                        Class code: {predicted_class}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(details["description"])

        with result_col2:
            st.metric(
                "Top-class probability",
                f"{confidence:.2%}",
            )

            st.progress(
                confidence,
                text="Predicted-class probability",
            )

            if confidence >= 0.90:
                st.success(
                    "High model confidence for this prediction."
                )
            elif confidence >= 0.70:
                st.warning(
                    "Moderate model confidence. Review the image."
                )
            else:
                st.warning(
                    "Low model confidence. Manual review is recommended."
                )

            st.caption(
                "These thresholds are illustrative, not calibrated "
                "reliability guarantees. Confidence is not the same "
                "as the probability that the prediction is correct."
            )

        # --------------------------------------------------
        # Class probabilities
        # --------------------------------------------------

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Class Probabilities</div>',
            unsafe_allow_html=True,
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    f"{code} — {CLASS_DETAILS[code]['name']}"
                    for code in CLASS_NAMES
                ],
                "Probability": results["probabilities"],
            }
        ).sort_values("Probability", ascending=True)

        chart_col, table_col = st.columns([1.2, 1])

        with chart_col:
            st.bar_chart(
                probability_df.set_index("Class"),
                x_label="Probability",
                horizontal=True,
            )

        with table_col:
            display_df = probability_df.sort_values(
                "Probability",
                ascending=False,
            ).copy()

            display_df["Probability"] = display_df["Probability"].map(
                lambda value: f"{value:.2%}"
            )

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
            )

        # --------------------------------------------------
        # Grad-CAM explainability
        # --------------------------------------------------

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Visual Explainability</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Grad-CAM highlights image regions that influence the "
            "model's prediction. These visualizations are approximate "
            "explanations, not proof that the model used the correct "
            "defect characteristics."
        )

        cam_col1, cam_col2, cam_col3 = st.columns(3)

        with cam_col1:
            st.markdown("**Processed Image**")
            st.image(
                results["processed_image"],
                caption="200 × 200 RGB representation",
                use_container_width=True,
            )

        with cam_col2:
            st.markdown("**Grad-CAM Heatmap**")
            st.image(
                results["heatmap"],
                caption="Highlighted activation regions",
                use_container_width=True,
            )

        with cam_col3:
            st.markdown("**Grad-CAM Overlay**")
            st.image(
                results["overlay"],
                caption="Heatmap over the processed image",
                use_container_width=True,
            )

        st.caption(
            "Grad-CAM uses the model's final convolutional block. "
            "Highlighted regions should be interpreted cautiously."
        )


# ==========================================================
# 8. Footer
# ==========================================================

st.markdown("---")

footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:
    st.caption(
        "Steel Surface Defect Detection | ResNet18 | PyTorch | Grad-CAM"
    )

with footer_col2:
    st.caption("Educational and research demonstration")