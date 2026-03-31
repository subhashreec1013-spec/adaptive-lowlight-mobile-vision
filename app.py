# ===========================
# IMPORTS
# ===========================
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import matplotlib.pyplot as plt

# 🔥 PIPELINE
from pipeline.main_pipeline import LowLightEnhancementPipeline
from core.optical_flow import compute_sequence_flow
from core.motion_mask import create_motion_masks
from core.fusion import enhance_low_light

pipeline = LowLightEnhancementPipeline()

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="LuminaEnhance Pro",
    page_icon="🔆",
    layout="wide"
)

# ===========================
# CUSTOM CSS (ENHANCED UI)
# ===========================
st.markdown("""
<style>
body {background-color: #0f172a;}
.block-container {padding-top: 2rem;}

.title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #9ca3af;
    font-size: 1.2rem;
    margin-bottom: 20px;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1f2937;
    margin-bottom: 20px;
}

.feature {
    padding: 15px;
    border-radius: 12px;
    background: #1f2937;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# HERO SECTION
# ===========================
st.markdown("<div class='title'>🔆 LuminaEnhance Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Low-Light Enhancement using Adaptive AI</div>", unsafe_allow_html=True)

# Feature Row
f1, f2, f3, f4 = st.columns(4)
f1.markdown("<div class='feature'>🧠 Adaptive AI</div>", unsafe_allow_html=True)
f2.markdown("<div class='feature'>📸 Multi-Frame Fusion</div>", unsafe_allow_html=True)
f3.markdown("<div class='feature'>🎯 Motion Aware</div>", unsafe_allow_html=True)
f4.markdown("<div class='feature'>👤 Face Protection</div>", unsafe_allow_html=True)

# ===========================
# PIPELINE FUNCTION
# ===========================
def process_pipeline(images):
    frames = [np.array(img.convert("RGB")) for img in images]

    flows = compute_sequence_flow(frames)
    masks = create_motion_masks(flows)

    scene = pipeline.scene_analyzer.analyze(frames, flows)
    params = pipeline.adaptive_controller.decide_parameters(scene)

    frames = frames[:params["num_frames"]]

    enhanced = enhance_low_light(frames, flows, masks, params)
    enhanced = pipeline.temporal_smoother.smooth(enhanced)

    return enhanced, scene, params, masks

# ===========================
# COMPARISON SLIDER (FIXED)
# ===========================
def comparison_slider(orig, enh):
    alpha = st.slider("🔄 Compare", 0, 100, 50)

    orig = orig.convert("RGB")

    if isinstance(enh, np.ndarray):
        enh = Image.fromarray(enh)

    enh = enh.convert("RGB")

    orig_np = np.array(orig).astype(float)
    enh_np = np.array(enh).astype(float)

    blend = (alpha/100)*enh_np + (1-alpha/100)*orig_np
    st.image(blend.astype(np.uint8), use_container_width=True)

# ===========================
# UPLOAD SECTION
# ===========================
st.markdown("### 📤 Upload Camera Frames (Burst Mode)")
files = st.file_uploader(
    "Upload Images",
    type=['png','jpg','jpeg'],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ===========================
# MAIN PROCESS
# ===========================
if files:
    images = [Image.open(f) for f in files]

    st.markdown("### 📷 Input Frames")
    cols = st.columns(min(4, len(images)))
    for i, img in enumerate(images):
        cols[i % 4].image(img, use_container_width=True)

    if st.button("🚀 Enhance"):

        with st.spinner("🧠 Running Adaptive Pipeline..."):
            start = time.time()
            enhanced, scene, params, masks = process_pipeline(images)
            end = time.time()

        st.success(f"✅ Completed in {end-start:.2f}s")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📷 Original")
            st.image(images[0], use_container_width=True)

        with col2:
            st.markdown("### ✨ Enhanced")
            st.image(enhanced, use_container_width=True)

        # ===========================
        # COMPARISON
        # ===========================
        st.markdown("### 🔄 Before vs After")
        comparison_slider(images[0], enhanced)

        # ===========================
        # MOTION MASK
        # ===========================
        st.markdown("### 🎯 Motion Mask")
        if masks:
            st.image(masks[0]['soft'], use_container_width=True)

        # ===========================
        # SCENE ANALYSIS
        # ===========================
        st.markdown("### 🧠 Scene Analysis")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brightness", f"{scene['brightness']:.2f}")
        c2.metric("Noise", f"{scene['noise']:.2f}")
        c3.metric("Contrast", f"{scene['contrast']:.2f}")
        c4.metric("Motion", f"{scene['motion']:.2f}")

        # ===========================
        # ADAPTIVE PARAMETERS
        # ===========================
        st.markdown("### ⚙️ Adaptive Decisions")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Gamma", params["gamma"])
        d2.metric("CLAHE", params["clahe_clip"])
        d3.metric("Denoise", params["denoise"])
        d4.metric("Frames Used", params["num_frames"])

        # ===========================
        # GRAPH
        # ===========================
        st.markdown("### 📊 Scene Graph")

        fig, ax = plt.subplots()
        ax.bar(scene.keys(), scene.values())
        st.pyplot(fig)

        # ===========================
        # DOWNLOAD
        # ===========================
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

        with open(tmp.name, "rb") as f:
            st.download_button("⬇ Download Enhanced Image", f)