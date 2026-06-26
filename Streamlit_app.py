import os
import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NAION Risk Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Load env + models (cached so they load once per session)
# ---------------------------------------------------------------------------
load_dotenv()

@st.cache_resource
def load_models():
    from inference import _load_segmentation_model, _load_yolo_model
    yolo_path  = os.getenv("YOLO_MODEL_PATH", "weights/yolov8n_disc_localizer.pt")
    seg_model  = _load_segmentation_model()
    yolo_model = _load_yolo_model(yolo_path)
    return seg_model, yolo_model


# ---------------------------------------------------------------------------
# Helper — run full pipeline on uploaded image
# ---------------------------------------------------------------------------
def run_pipeline(image_rgb, seg_model, yolo_model):
    from inference import predict
    from feature_extraction import extract_features, compute_naion_risk_score

    result   = predict(image_rgb, seg_model, yolo_model)
    features = extract_features(
        result["disc_mask_full"],
        result["cup_mask_full"],
        image_rgb
    )
    score, risk_level, breakdown = compute_naion_risk_score(features)
    return result, features, score, risk_level, breakdown


# ---------------------------------------------------------------------------
# Helper — draw overlay
# ---------------------------------------------------------------------------
def draw_overlay(image_rgb, disc_mask, cup_mask):
    overlay   = image_rgb.copy()
    disc_c, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_c,  _ = cv2.findContours(cup_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, disc_c, -1, (0, 255, 0), 3)
    cv2.drawContours(overlay, cup_c,  -1, (255, 0, 0), 3)
    return overlay


# ---------------------------------------------------------------------------
# Helper — feature distribution chart
# ---------------------------------------------------------------------------
def plot_feature_gauge(value, min_val, max_val, threshold, label, higher_is_riskier=False):
    """Mini gauge chart showing where this patient sits relative to threshold."""
    fig, ax = plt.subplots(figsize=(4, 0.6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Background bar
    ax.barh(0, max_val - min_val, left=min_val, height=0.4,
            color="#2d2d2d", zorder=1)

    # Value bar
    if higher_is_riskier:
        color = "#e74c3c" if value > threshold else "#2ecc71"
    else:
        color = "#e74c3c" if value < threshold else "#2ecc71"

    ax.barh(0, value - min_val, left=min_val, height=0.4,
            color=color, zorder=2)

    # Threshold line
    ax.axvline(threshold, color="white", linewidth=1.5, linestyle="--", zorder=3)

    # Value text
    ax.text(value, 0, f" {value:.3f}", va="center", ha="left",
            color="white", fontsize=8, fontweight="bold")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Fundus_photograph_of_normal_right_eye.jpg/220px-Fundus_photograph_of_normal_right_eye.jpg",
                 caption="Example fundus image", use_column_width=True)

        st.markdown("## About")
        st.markdown("""
        This tool analyzes fundus photographs to assess the risk of
        **Non-Arteritic Anterior Ischemic Optic Neuropathy (NAION)**.

        **Pipeline:**
        - 🎯 YOLOv8n disc localization
        - 🔬 EfficientNet-b4 segmentation
        - 📐 Clinical feature extraction
        - 📊 Literature-based risk scoring

        **Risk criteria based on:**
        - Hayreh SS (2009)
        - Contreras I et al. (2018)
        - Arnold AC (2003)
        """)

        st.markdown("---")
        st.markdown("⚠️ **Disclaimer**")
        st.caption(
            "This tool is for research purposes only. "
            "Not validated for clinical use. "
            "Always consult a qualified ophthalmologist."
        )

        st.markdown("---")
        st.markdown("**Risk Score Guide**")
        st.markdown("🟢 0–39: Low risk")
        st.markdown("🟠 40–64: Moderate risk")
        st.markdown("🔴 65–100: High risk")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    st.title("👁️ NAION Risk Analyzer")
    st.markdown("Upload a fundus photograph to receive an automated NAION risk assessment.")
    st.markdown("---")

    # --- File upload ---
    uploaded_file = st.file_uploader(
        "Upload fundus image (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="High-quality fundus photographs work best"
    )

    if uploaded_file is None:
        st.info("👆 Upload a fundus image to get started.")
        return

    # --- Load image ---
    pil_img   = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_img)

    # --- Load models ---
    with st.spinner("Loading models..."):
        try:
            seg_model, yolo_model = load_models()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return

    # --- Run pipeline ---
    with st.spinner("Analyzing fundus image..."):
        try:
            result, features, score, risk_level, breakdown = run_pipeline(
                image_rgb, seg_model, yolo_model
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    # --- Layout: two columns ---
    col1, col2 = st.columns([1, 1])

    # -----------------------------------------------------------------------
    # Left column — image + overlay
    # -----------------------------------------------------------------------
    with col1:
        st.markdown("### Fundus Image Analysis")

        tab1, tab2 = st.tabs(["Segmentation Overlay", "Original Image"])

        with tab1:
            overlay = draw_overlay(
                image_rgb,
                result["disc_mask_full"],
                result["cup_mask_full"]
            )
            st.image(overlay, caption="Green = Disc  |  Red = Cup", use_column_width=True)

            yolo_conf = result.get("yolo_conf")
            if yolo_conf:
                st.caption(f"YOLO disc detection confidence: {yolo_conf:.3f}")
            else:
                st.caption("⚠️ YOLO fallback used — disc not detected confidently")

        with tab2:
            st.image(image_rgb, caption="Original image", use_column_width=True)

    # -----------------------------------------------------------------------
    # Right column — risk score + features
    # -----------------------------------------------------------------------
    with col2:
        st.markdown("### NAION Risk Assessment")

        # Risk score display
        score_color = {"Low": "green", "Moderate": "orange", "High": "red"}.get(risk_level, "gray")
        score_emoji = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}.get(risk_level, "⚪")

        if risk_level == "Invalid":
            st.error("⚠️ Segmentation failed — could not assess risk")
            return

        st.markdown(
            f"<h1 style='text-align:center; color:{score_color};'>"
            f"{score_emoji} {score}/100</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='text-align:center; color:{score_color};'>"
            f"Risk Level: {risk_level}</h3>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Score breakdown
        if breakdown:
            st.markdown("#### Score Breakdown")
            for rule, pts in breakdown.items():
                if pts > 0:
                    st.markdown(f"- **+{pts}** — {rule}")

        st.markdown("---")

        # Feature table
        st.markdown("#### Clinical Features")

        if features:
            feature_data = {
                "Feature": [
                    "vCDR", "hCDR", "Area CDR", "CDI",
                    "Rim Thickness Ratio", "Inferior Rim Ratio",
                    "ISNT Violation", "Disc Roundness",
                    "Disc Area (px)", "Cup Area (px)",
                    "Disc-Fovea Ratio (ref)"
                ],
                "Value": [
                    f"{features['vCDR']:.3f}",
                    f"{features['hCDR']:.3f}",
                    f"{features['area_CDR']:.3f}",
                    f"{features['CDI']:.3f}",
                    f"{features['rim_thickness_ratio']:.3f}",
                    f"{features['inferior_rim_ratio']:.3f}",
                    str(features['isnt_violation']),
                    f"{features['disc_roundness']:.3f}",
                    str(features['disc_area_px']),
                    str(features['cup_area_px']),
                    f"{features['disc_fovea_ratio']:.3f}",
                ],
                "Threshold": [
                    "< 0.5 = crowded",
                    "< 0.5 = crowded",
                    "< 0.2 = small cup",
                    "> 0.78 = crowded",
                    "< 0.12 = thin",
                    "< 0.90 = thinning",
                    "True = risk",
                    "> 0.93 = round",
                    "", "",
                    "ref only"
                ]
            }
            st.dataframe(
                pd.DataFrame(feature_data),
                use_container_width=True,
                hide_index=True
            )

    # -----------------------------------------------------------------------
    # Bottom section — distribution charts
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Feature Analysis — Where This Eye Sits")
    st.caption("Red dashed line = clinical threshold. Bar color = risk indicator.")

    if features:
        gauge_cols = st.columns(3)

        gauges = [
            ("vCDR",                 features["vCDR"],                 0.0, 1.0,  0.5,  "vCDR",                False),
            ("CDI",                  features["CDI"],                  0.5, 1.0,  0.78, "CDI",                 True),
            ("Rim Thickness Ratio",  features["rim_thickness_ratio"],  0.0, 0.30, 0.12, "Rim Thickness",       False),
            ("Area CDR",             features["area_CDR"],             0.0, 0.7,  0.20, "Area CDR",            False),
            ("Inferior Rim Ratio",   features["inferior_rim_ratio"],   0.7, 1.2,  0.90, "Inferior Rim",        False),
            ("Disc Roundness",       features["disc_roundness"],       0.5, 1.0,  0.93, "Disc Roundness",      True),
        ]

        for idx, (label, value, min_v, max_v, thresh, short, higher_risk) in enumerate(gauges):
            with gauge_cols[idx % 3]:
                st.markdown(f"**{label}**")
                fig = plot_feature_gauge(value, min_v, max_v, thresh, short, higher_risk)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # -----------------------------------------------------------------------
    # Download results
    # -----------------------------------------------------------------------
    st.markdown("---")
    if features and risk_level != "Invalid":
        result_dict = {
            "risk_score":  score,
            "risk_level":  risk_level,
            **{k: v for k, v in features.items()},
            "yolo_conf":   result.get("yolo_conf"),
        }
        df_result = pd.DataFrame([result_dict])
        csv_bytes = df_result.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_bytes,
            file_name="naion_risk_result.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
