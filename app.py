"""Streamlit app for hierarchical bird image classification."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from inference import (
    CONFIDENCE_THRESHOLD,
    default_checkpoint,
    load_model,
    load_oriented_image,
    predict_hierarchical,
)


@st.cache_resource
def load_classifier():
    return load_model(default_checkpoint())


st.set_page_config(page_title="Bird Classifier", page_icon="🐦", layout="centered")
st.title("Bird Classifier")
st.markdown(
    "Upload a bird photo and the app will first identify the **bird family**, "
    "then predict the **species** within that family when a species model is available. "
    f"Family predictions need at least **{CONFIDENCE_THRESHOLD * 100:.0f}%** confidence "
    "for a single answer."
)

uploaded_file = st.file_uploader("Choose a bird image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = load_oriented_image(uploaded_file)
    width, height = image.size
    if height > width:
        st.image(image, caption="Uploaded image", width=min(420, width))
    else:
        st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        family_model, family_class_names, device = load_classifier()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Identifying bird..."):
        result = predict_hierarchical(image, family_model, family_class_names, device)

    family = result["family"]
    species = result["species"]

    if family["confident"]:
        st.success(family["message"])
    else:
        st.warning(family["message"])

    if species and species.get("options"):
        if species["confident"]:
            st.success(species["message"])
        else:
            st.info(species["message"])
    elif species:
        st.info(species["message"])

    st.subheader("Family predictions")
    for rank, (family_name, confidence) in enumerate(family["options"], start=1):
        st.write(f"{rank}. **{family_name}** — {confidence * 100:.1f}%")

    if species and species.get("options"):
        st.subheader("Species predictions")
        for rank, (species_name, confidence) in enumerate(species["options"], start=1):
            st.write(f"{rank}. **{species_name}** — {confidence * 100:.1f}%")

else:
    st.info("Upload an image to get started.")
    if not default_checkpoint().exists():
        st.warning(
            "No trained model found yet. Train one with:\n\n"
            "`python model/model_training.py --data-dir data/families`\n"
            "`python model/train_species_models.py`"
        )
