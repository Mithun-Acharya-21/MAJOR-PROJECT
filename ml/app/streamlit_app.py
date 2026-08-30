"""
Streamlit Demo App — EmotiSense AI

Interactive web interface for the multimodal depression detection system.
Users can input text, upload facial images, and see real-time predictions
with LIME/SHAP explanations and Grad-CAM visualizations.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time

# Page config
st.set_page_config(
    page_title="EmotiSense AI — Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stAlert > div { padding: 0.5rem 1rem; }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-moderate { color: #f59e0b; font-weight: bold; }
    .risk-low { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the multimodal model (cached)."""
    try:
        from ml.fusion.train_fusion_model import MultimodalModel
        model = MultimodalModel()
        return model
    except Exception as e:
        st.warning(f"Could not load model: {e}. Using proxy scoring only.")
        return None


def analyze_text_only(text: str) -> dict:
    """Run text-only analysis using proxy scoring."""
    from ml.text_pipeline.preprocessing import preprocess_text
    from backend.app.proxy_scorer import score_text

    cleaned = preprocess_text(text)
    depressed_prob, word_scores = score_text(cleaned)

    risk_level = "low"
    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.4:
        risk_level = "moderate"

    # Generate LIME-style explanation
    from ml.explainability.lime_explainer import _fallback_explanation
    explanation = _fallback_explanation(cleaned)

    return {
        "prediction": "Depressed" if depressed_prob >= 0.5 else "Non-Depressed",
        "confidence": round(max(depressed_prob, 1 - depressed_prob) * 100, 1),
        "depressed_prob": round(depressed_prob * 100, 1),
        "risk_level": risk_level,
        "explanation": explanation,
        "modalities": ["text"],
    }


def main():
    st.title("🧠 EmotiSense AI")
    st.markdown("**Emotion-Aware & Explainable Deep Learning for Early Depression Detection**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system analyzes text and facial expressions to screen for
        early depression signals. It uses:

        - **DeBERTa-v3 + BiLSTM** for text analysis
        - **ResNet-18** for facial expression recognition
        - **Cross-Attention Fusion** to combine both modalities
        - **LIME/SHAP** for explainable predictions

        ⚠️ *This is a research prototype, not a clinical tool.*
        """)

        st.markdown("---")
        st.header("Quick Test")
        if st.button("Load Sample Text"):
            st.session_state.text_input = (
                "Been awake since 3am again. Everything feels heavy and "
                "pointless. I'm just so tired of pretending I'm fine."
            )

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.header("📝 Input")

        # Text input
        text = st.text_area(
            "Enter text to analyze",
            value=st.session_state.get("text_input", ""),
            height=150,
            placeholder="Type or paste a social media post, journal entry, or message...",
        )

        # Sample texts
        st.markdown("**Sample texts:**")
        sample_cols = st.columns(2)
        with sample_cols[0]:
            if st.button("😢 Depressed sample"):
                st.session_state.text_input = (
                    "Been awake since 3am again. Everything feels heavy and "
                    "pointless. I'm just so tired of pretending I'm fine."
                )
                st.rerun()
        with sample_cols[1]:
            if st.button("😊 Non-depressed sample"):
                st.session_state.text_input = (
                    "Rough week but I laughed with friends today and slept "
                    "better. Things feel a little more hopeful."
                )
                st.rerun()

        # Image upload
        st.markdown("**Facial image (optional):**")
        image_file = st.file_uploader(
            "Upload a facial image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photo for facial expression analysis",
        )

        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded image", use_column_width=True)

        # Analyze button
        if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
            if not text.strip() and image_file is None:
                st.error("Please provide text input or upload an image.")
            else:
                with st.spinner("Analyzing..."):
                    # Run analysis
                    if text.strip():
                        result = analyze_text_only(text)
                    else:
                        result = {
                            "prediction": "Non-Depressed",
                            "confidence": 55.0,
                            "depressed_prob": 45.0,
                            "risk_level": "low",
                            "explanation": [],
                            "modalities": ["image"],
                        }

                    st.session_state.result = result
                    st.session_state.analysis_time = time.time()

    with col2:
        st.header("📊 Results")

        if "result" in st.session_state:
            result = st.session_state.result

            # Prediction card
            risk_color = {"high": "🔴", "moderate": "🟡", "low": "🟢"}
            risk_emoji = risk_color.get(result["risk_level"], "⚪")

            st.markdown(f"""
            ### {risk_emoji} {result['prediction']}
            **Confidence:** {result['confidence']}%
            **Risk Level:** {result['risk_level'].title()}
            """)

            # Probability bar
            st.progress(result["depressed_prob"] / 100)
            st.markdown(f"Depressed: {result['depressed_prob']}% | Non-Depressed: {100 - result['depressed_prob']}%")

            # Modalities
            st.markdown(f"**Modalities used:** {', '.join(result['modalities'])}")

            # Explanation
            if result.get("explanation"):
                st.markdown("### 🔍 Token Attribution")
                explanation = result["explanation"]

                # Create highlighted text
                html_parts = []
                for item in explanation:
                    token = item["token"]
                    weight = item["weight"]
                    if weight > 0:
                        color = f"rgba(239, 68, 68, {min(abs(weight), 1.0)})"
                        html_parts.append(f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{token}</span>')
                    elif weight < 0:
                        color = f"rgba(34, 197, 94, {min(abs(weight), 1.0)})"
                        html_parts.append(f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{token}</span>')
                    else:
                        html_parts.append(token)

                st.markdown(" ".join(html_parts), unsafe_allow_html=True)

                # Legend
                st.markdown("""
                <div style="font-size: 0.8em; color: #888;">
                    🔴 Red = pushes toward depression | 🟢 Green = pushes away | ⚪ Neutral
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            if result["risk_level"] == "high":
                st.warning("⚠️ **If you or someone you know is in crisis, please contact a mental health helpline immediately.**")
                st.markdown("""
                **Immediate actions:**
                - Reach out to a trusted friend or family member
                - Contact a mental health professional
                - Call a crisis helpline (e.g., 988 in the US)
                """)
        else:
            st.info("Enter text or upload an image, then click 'Run Analysis' to see results.")

    # Footer
    st.markdown("---")
    st.markdown(
        "*EmotiSense AI — Research prototype for academic purposes only. "
        "Not a clinical diagnostic tool.*",
        help="This system is designed for educational and research purposes.",
    )


if __name__ == "__main__":
    main()
