import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import streamlit as st

from src.config import CHECKPOINT_PATH, CLASS_NAMES, DEFECTIVE_IDX, read_data_source
from src.gradcam import explain_image
from src.model import load_checkpoint

st.set_page_config(page_title="Defect Detector", layout="centered")


@st.cache_resource
def get_model(path: str, mtime: float):
    return load_checkpoint(Path(path), device="cpu")


st.title("Product Defect Detector")
st.caption("Transfer learning (ResNet) + Grad-CAM explanations")

if not CHECKPOINT_PATH.exists():
    st.error("No trained model found. Train one first:\n\n```\npython download_data.py\npython train.py\n```")
    st.stop()

model, ckpt = get_model(str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime)
default_threshold = float(ckpt.get("decision_threshold", 0.5))
data_source = ckpt.get("data_source") or read_data_source()

if data_source == "synthetic":
    st.warning("This model was trained on **synthetic** data, so it will not judge real products correctly "
               "(it will mostly say Defective). Run `python download_data.py --force`, then `python train.py`.")
else:
    st.caption("Trained on the casting-product dataset (top-down photos of pump impellers). "
               "Other kinds of products are outside what it knows.")

with st.sidebar:
    st.header("Decision rule")
    threshold = st.slider(
        "Flag as Defective when P(defect) ≥", 0.05, 0.95, default_threshold, 0.05,
        help="Lower = catches more defects (fewer false negatives) but raises more false alarms. "
             "The default was tuned in train.py so that a missed defect costs more than a false alarm.",
    )
    st.markdown(f"Model: **{ckpt['arch']}**  \nBest val F1: **{ckpt['val_f1']:.3f}**  \n"
                f"Data: **{data_source}**  \nTuned default threshold: **{default_threshold:.2f}**")
    if default_threshold <= 0.2:
        st.warning("The tuned threshold is very low, so many good parts will be flagged. "
                   "Raise the slider, or retrain with a smaller `--fn-cost` (e.g. 3).")
    st.markdown("A **missed defect** (false negative) usually costs far more than a **false alarm**, "
                "so the threshold is deliberately cautious. See the README section on the cost of false negatives.")

uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    bgr = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read that file as an image.")
        st.stop()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Analyzing..."):
        result = explain_image(model, rgb, device="cpu", target_class=DEFECTIVE_IDX)

    p_defect = float(result["probs"][DEFECTIVE_IDX])
    is_defective = p_defect >= threshold
    label = CLASS_NAMES[DEFECTIVE_IDX] if is_defective else CLASS_NAMES[1 - DEFECTIVE_IDX]
    confidence = p_defect if is_defective else 1.0 - p_defect

    if is_defective:
        st.error(f"### {label}  —  confidence {confidence:.1%}")
    else:
        st.success(f"### {label}  —  confidence {confidence:.1%}")
    st.progress(min(max(p_defect, 0.0), 1.0), text=f"P(defect) = {p_defect:.1%}  (threshold {threshold:.2f})")

    left, right = st.columns(2)
    left.image(result["resized"], caption="Input (224×224)")
    right.image(result["overlay"], caption="Grad-CAM: red = where the model sees defect evidence")

    st.caption("The heatmap is coarse (7×7 resolution upsampled), so read it as 'roughly here'. "
               "If it highlights the background instead of the part, don't trust the prediction.")
else:
    st.info("Upload a photo to get started.")
