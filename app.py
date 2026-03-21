import json
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "2-epoches-test.keras"
META_PATH = ARTIFACTS_DIR / "meta.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_meta():
    if not META_PATH.exists():
        return None
    with META_PATH.open("r") as handle:
        return json.load(handle)

def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r") as handle:
        return json.load(handle)

def preprocess_image(image: Image.Image, img_size: int):
    # Resize to match model input
    # Note: Model has Rescaling layer, so we keep range [0, 255]
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((img_size, img_size))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    return img_array

def main():
    st.set_page_config(page_title="Binary Classifier (TF)", layout="centered")
    st.title("Binary Image Classifier")

    meta = load_meta()
    metrics = load_metrics()

    if meta is None or not MODEL_PATH.exists():
        st.warning(f"Model not found at {MODEL_PATH}. Run: python main.py")
        return

    class_names = meta.get("class_names", ["Class 0", "Class 1"])
    img_size = meta.get("img_size", 96)

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)

        input_tensor = preprocess_image(image, img_size)
        
        # Predict
        predictions = model.predict(input_tensor)
        score = float(predictions[0][0])
        
        # Sigmoid output: 0.0 to 1.0
        # < 0.5 is class 0, >= 0.5 is class 1
        predicted_class = class_names[1] if score >= 0.5 else class_names[0]
        confidence = score if score >= 0.5 else 1 - score
        
        st.success(f"Prediction: {predicted_class} ({confidence:.2%})")
        
        st.write("Raw probabilities:")
        st.write({
            class_names[0]: 1 - score,
            class_names[1]: score
        })

    if metrics:
        st.subheader("Latest training metrics")
        st.json(metrics)

if __name__ == "__main__":
    main()
