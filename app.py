import json
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "algae.keras"
META_PATH = ARTIFACTS_DIR / "meta.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

        :root {
            --ink:    #111111;
            --paper:  #f4f1eb;
            --rule:   #111111;
            --muted:  #888888;
            --accent: #c0392b;
            --field:  #e8e4dc;
        }

        /* Paint every possible root wrapper */
        body,
        #root, #root > div,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"] > div,
        [data-testid="stDecoration"],
        .stApp { background-color: #f4f1eb !important; }

        /* Reset & base */
        *, *::before, *::after { box-sizing: border-box; }

        html, body, [class*="css"], .main, .block-container,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="stHeader"],
        [data-testid="stBottom"],
        section[data-testid="stSidebar"],
        .stApp, .css-1d391kg, .css-18e3th9 {
            background: var(--paper) !important;
            color: var(--ink) !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }

        /* Target every possible Streamlit wrapper */
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stWidgetLabel"],
        .stMarkdown, .element-container {
            background: transparent !important;
        }

        .block-container {
            max-width: 1300px !important;
            padding: 2rem 2.5rem 4rem !important;
        }

        /* Hide branding */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── MASTHEAD ── */
        .masthead {
            border-top: 4px solid var(--ink);
            border-bottom: 1px solid var(--ink);
            padding: 1.4rem 0 1rem;
            margin-bottom: 0.4rem;
            text-align: center;
        }
        .masthead-kicker {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }
        .masthead-title {
            font-family: 'DM Serif Display', serif;
            font-size: 3.6rem;
            font-weight: 400;
            line-height: 1;
            color: var(--ink);
            letter-spacing: -0.01em;
            margin: 0;
        }
        .masthead-sub {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--muted);
            margin-top: 0.6rem;
            border-top: 1px solid var(--ink);
            padding-top: 0.6rem;
        }

        /* ── SECTION LABELS ── */
        h3 {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.65rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.25em !important;
            text-transform: uppercase !important;
            color: var(--ink) !important;
            margin: 2rem 0 0.8rem !important;
            padding-bottom: 0.4rem !important;
            border-bottom: 1px solid var(--ink) !important;
        }

        /* ── UPLOAD AREA ── */
        [data-testid="stFileUploader"] {
            background: var(--field) !important;
            border: 1px solid var(--ink) !important;
            border-radius: 0 !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploader"] label { color: var(--muted) !important; }
        [data-testid="stFileDropzone"] {
            background: var(--field) !important;
            border: 1px dashed var(--muted) !important;
            border-radius: 0 !important;
        }

        /* ── IMAGES ── */
        [data-testid="stImage"] img {
            border: 1px solid var(--ink) !important;
            border-radius: 0 !important;
            filter: grayscale(15%);
        }

        /* ── EXPANDER ── */
        [data-testid="stExpander"] {
            border: 1px solid var(--rule) !important;
            border-radius: 0 !important;
            background: var(--field) !important;
        }
        [data-testid="stExpander"] summary {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.15em !important;
            text-transform: uppercase !important;
            color: var(--ink) !important;
        }

        /* ── METRICS ── */
        [data-testid="metric-container"] {
            background: var(--field) !important;
            border: 1px solid var(--ink) !important;
            border-radius: 0 !important;
            padding: 1.2rem 1rem !important;
        }
        [data-testid="metric-container"] label {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.6rem !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            color: var(--muted) !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'DM Serif Display', serif !important;
            font-size: 2.2rem !important;
            color: var(--ink) !important;
            font-weight: 400 !important;
        }

        /* ── DIVIDER ── */
        hr {
            border: none !important;
            border-top: 1px solid var(--ink) !important;
            margin: 2.5rem 0 !important;
        }

        /* ── SPINNER ── */
        .stSpinner > div { border-top-color: var(--accent) !important; }

        /* ── ALERTS / ERRORS ── */
        [data-testid="stAlert"] {
            border-radius: 0 !important;
            border-left: 3px solid var(--accent) !important;
            background: var(--field) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem !important;
        }

        /* ── RESULT CARD ── */
        .result-block {
            border: 1px solid var(--ink);
            padding: 2rem 2rem 1.8rem;
            margin: 1rem 0;
            background: var(--paper);
            position: relative;
        }
        .result-tag {
            font-size: 0.6rem;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }
        .result-class {
            font-family: 'DM Serif Display', serif;
            font-size: 3rem;
            line-height: 1;
            color: var(--ink);
            margin: 0.2rem 0 0.8rem;
        }
        .result-confidence {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: var(--muted);
            letter-spacing: 0.1em;
        }

        /* ── SCORE BLOCK ── */
        .score-block {
            border: 1px solid var(--ink);
            border-left: 4px solid var(--accent);
            padding: 2rem 1.5rem;
            margin: 1rem 0;
            background: var(--paper);
        }
        .score-tag {
            font-size: 0.6rem;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }
        .score-value {
            font-family: 'DM Serif Display', serif;
            font-size: 2.6rem;
            color: var(--accent);
            line-height: 1;
        }
        .score-note {
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--muted);
            margin-top: 0.5rem;
        }

        /* ── PROBABILITY BAR ── */
        .prob-row {
            margin-bottom: 1.4rem;
        }
        .prob-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.35rem;
        }
        .prob-name {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--ink);
        }
        .prob-pct {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--muted);
        }
        .prob-track {
            width: 100%;
            height: 6px;
            background: var(--field);
            border: 1px solid #cccccc;
        }
        .prob-fill-neg {
            height: 100%;
            background: var(--accent);
        }
        .prob-fill-pos {
            height: 100%;
            background: var(--ink);
        }

        /* ── PLACEHOLDER ── */
        .img-placeholder {
            border: 1px dashed #aaaaaa;
            height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--field);
            color: var(--muted);
            font-size: 0.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }

        /* ── FOOTER ── */
        .footer-rule {
            border-top: 4px solid var(--ink);
            padding-top: 1rem;
            margin-top: 0.5rem;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--muted);
            text-align: center;
        }

        /* General text */
        p, span, label { color: var(--muted) !important; font-size: 0.82rem !important; }
        strong, b { color: var(--ink) !important; }
        code {
            background: var(--field) !important;
            border-radius: 0 !important;
            color: var(--ink) !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_meta():
    if not META_PATH.exists():
        return None
    with META_PATH.open("r") as f:
        return json.load(f)


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r") as f:
        return json.load(f)


def preprocess_image(image: Image.Image, img_size: int):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def render_probability_bars(class_names, score):
    prob_neg = 1 - score
    prob_pos = score

    mono = "font-family:'IBM Plex Mono',monospace"

    def bar(label, prob, fill_color):
        return f"""
        <div style="margin-bottom:1.8rem">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.45rem">
                <span style="{mono};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#111111;font-weight:500">{label}</span>
                <span style="{mono};font-size:1.1rem;font-weight:700;color:#111111;letter-spacing:0.04em">{prob*100:.1f}%</span>
            </div>
            <div style="width:100%;height:8px;background:#e8e4dc;border:1px solid #cccccc">
                <div style="width:{prob*100:.2f}%;height:100%;background:{fill_color}"></div>
            </div>
            <div style="{mono};font-size:0.65rem;color:#888888;margin-top:0.3rem">score: {prob:.6f}</div>
        </div>
        """

    st.markdown(
        bar(class_names[0], prob_neg, "#c0392b") +
        bar(class_names[1], prob_pos, "#111111"),
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="Sargassum and Gracilaria Classifier",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    apply_custom_styles()

    # Masthead
    st.markdown("""
        <div class="masthead">
            <div class="masthead-kicker">Machine Learning / Classification System</div>
            <div class="masthead-title">Sargassum and Gracilaria classifier</div>
            <div class="masthead-sub">Binary Classification &nbsp;&mdash;&nbsp; Confidence Scoring &nbsp;&mdash;&nbsp; TensorFlow</div>
        </div>
    """, unsafe_allow_html=True)

    meta = load_meta()
    metrics = load_metrics()

    if meta is None or not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run `python main.py` to train and save the model.")
        return

    class_names = meta.get("class_names", ["Class 0", "Class 1"])
    img_size = meta.get("img_size", 96)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Image Input")
        uploaded = st.file_uploader(
            "Upload JPG or PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        with st.expander("Model Configuration"):
            st.markdown(f"""
**Classes:** `{class_names[0]}` / `{class_names[1]}`

**Input size:** {img_size} x {img_size} px

**Architecture:** Binary deep neural network

**Framework:** TensorFlow / Keras
            """)

    with col_right:
        st.markdown("### Preview")
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)
        else:
            st.markdown("<div class='img-placeholder'>Awaiting image input</div>", unsafe_allow_html=True)

    if uploaded is not None:
        st.divider()

        with st.spinner("Running inference..."):
            image = Image.open(uploaded)
            input_tensor = preprocess_image(image, img_size)
            predictions = model.predict(input_tensor, verbose=0)
            score = float(predictions[0][0])

            predicted_class = class_names[1] if score >= 0.5 else class_names[0]
            confidence = score if score >= 0.5 else 1 - score

        st.markdown("### Classification Output")
        res_col1, res_col2 = st.columns([1.5, 1])

        with res_col1:
            st.markdown(f"""
                <div class="result-block">
                    <div class="result-tag">Predicted class</div>
                    <div class="result-class">{predicted_class}</div>
                    <div class="result-confidence">Confidence &nbsp; {confidence:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                <div class="score-block">
                    <div class="score-tag">Confidence score</div>
                    <div class="score-value">{score*100:.2f}%</div>
                    <div class="score-note">Sigmoid output &nbsp; [0, 1]</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### Probability Distribution")
        render_probability_bars(class_names, score)

    if metrics:
        st.divider()
        st.markdown("### Model Performance")

        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flat_metrics[f"{key} {sub_key}"] = sub_value
            elif isinstance(value, (int, float)):
                flat_metrics[key] = value

        if flat_metrics:
            metric_items = list(flat_metrics.items())
            num_cols = min(4, len(metric_items))
            cols = st.columns(num_cols)

            for idx in range(num_cols):
                if idx < len(metric_items):
                    key, value = metric_items[idx]
                    with cols[idx]:
                        display_name = key.replace("_", " ").upper()
                        st.metric(display_name, f"{value:.4f}" if isinstance(value, float) else value)

            if len(metric_items) > 4:
                with st.expander("Additional Metrics"):
                    remaining_cols = st.columns(4)
                    for idx, (key, value) in enumerate(metric_items[4:]):
                        with remaining_cols[idx % 4]:
                            display_name = key.replace("_", " ").upper()
                            st.metric(display_name, f"{value:.4f}" if isinstance(value, float) else value)
        else:
            st.info("No numeric metrics available.")

    st.divider()
    st.markdown("""
        <div class="footer-rule">
            Neural Classification System &nbsp;&mdash;&nbsp; TensorFlow Engine &nbsp;&mdash;&nbsp; Streamlit Interface
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()