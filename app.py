import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="TruthLens | Deepfake Screening", page_icon="🔎", layout="centered")

st.title("🔎 TruthLens")
st.subheader("A first step toward checking what's real")
st.write("Upload a face image and the model will estimate whether it resembles a real or manipulated face image.")
st.caption("Research/demo project — results are estimates, not proof of authenticity.")

@st.cache_resource(show_spinner="Loading the pretrained model… this may take a few minutes the first time.")
def load_detector():
    return pipeline(
        "image-classification",
        model="aaronkantrowitz/ai-image-detection",
        trust_remote_code=True,
    )

uploaded = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image…"):
                detector = load_detector()
                results = detector(image)

            if not results:
                st.error("The model did not return a result. Please try another image.")
            else:
                best = max(results, key=lambda item: float(item.get("score", 0)))
                st.write("Raw model output:", results)
                label = str(best.get("label", "unknown")).lower()
                score = float(best.get("score", 0))

                if "fake" in label:
                    st.warning(f"Model result: likely manipulated ({score:.1%} model confidence)")
                elif "real" in label:
                    st.success(f"Model result: likely real ({score:.1%} model confidence)")
                else:
                    st.info(f"Model result: {best.get('label', 'Unknown')} ({score:.1%} model confidence)")

                st.write("**Model scores**")
                for result in sorted(results, key=lambda item: float(item.get("score", 0)), reverse=True):
                    st.write(f"{result.get('label', 'Unknown')}: {float(result.get('score', 0)):.1%}")

                st.info(
                    "This is a pretrained model, not a forensic verdict. Its performance can vary with "
                    "image quality, compression, editing methods, and the kinds of faces it encountered "
                    "during training. Do not use this result alone to judge a person or an image."
                )
    except Exception as exc:
        st.error("Something went wrong while reading or analyzing this image.")
        st.caption(f"Technical detail: {exc}")

with st.expander("How it works"):
    st.write(
        "TruthLens sends the uploaded image to a pretrained deepfake image-classification model. "
        "The model returns labels and scores. This first version uses an existing model rather than "
        "training a neural network from scratch; later, the project can be extended with evaluation "
        "on a labelled dataset and an explanation of model limitations."
    )

st.divider()
st.caption("Built as a learning project. Images are processed by the model at inference time; avoid uploading private or sensitive images.")
