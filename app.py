# ===========================
# IMPORTS
# ===========================
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

from pipeline.main_pipeline import LowLightEnhancementPipeline
from core.optical_flow import compute_sequence_flow
from core.motion_mask import create_motion_masks
from core.fusion import enhance_low_light

pipeline = LowLightEnhancementPipeline()

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="LuminaEnhance Pro", page_icon="🔆", layout="wide")

# ===========================
# CSS
# ===========================
st.markdown("""
<style>
body {background: linear-gradient(135deg, #0f172a, #020617);}

.title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.feature {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown("<div class='title'>🔆 LuminaEnhance Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Adaptive Low-Light Enhancement for Mobile Cameras</div>", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)
f1.markdown("<div class='feature'>🧠 AI Adaptive</div>", unsafe_allow_html=True)
f2.markdown("<div class='feature'>📸 Multi-Frame</div>", unsafe_allow_html=True)
f3.markdown("<div class='feature'>🎯 Motion Aware</div>", unsafe_allow_html=True)
f4.markdown("<div class='feature'>👤 Face Safe</div>", unsafe_allow_html=True)

# ===========================
# PIPELINE
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
# SLIDER
# ===========================
def comparison_slider(orig, enh):
    if "slider_val" not in st.session_state:
        st.session_state.slider_val = 50

    alpha = st.slider("Compare", 0, 100, st.session_state.slider_val)

    st.session_state.slider_val = alpha

    orig = orig.convert("RGB")
    if isinstance(enh, np.ndarray):
        enh = Image.fromarray(enh)

    enh = enh.convert("RGB")

    blend = (alpha/100)*np.array(enh) + (1-alpha/100)*np.array(orig)

    st.image(blend.astype(np.uint8), width=400)

# ===========================
# UPLOAD
# ===========================
st.markdown("### 📤 Upload Burst Images")

files = st.file_uploader(
    "Upload",
    type=['png','jpg','jpeg'],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ===========================
# MAIN
# ===========================
if files:
    images = [Image.open(f) for f in files]

    st.markdown("### 📷 Input Frames")
    cols = st.columns(3)

    for i, img in enumerate(images):
        cols[i % 3].image(img, width=200)

    # ===========================
    # BUTTON
    # ===========================
    if st.button("🚀 Enhance"):

        with st.spinner("Processing..."):
            start = time.time()
            enhanced, scene, params, masks = process_pipeline(images)
            end = time.time()

        # 🔥 STORE RESULTS
        st.session_state["enhanced"] = enhanced
        st.session_state["scene"] = scene
        st.session_state["params"] = params
        st.session_state["masks"] = masks
        st.session_state["original"] = images[0]

        st.success(f"Done in {end-start:.2f}s")

# ===========================
# 🔥 SHOW RESULTS (OUTSIDE BUTTON)
# ===========================
if "enhanced" in st.session_state:

    enhanced = st.session_state["enhanced"]
    scene = st.session_state["scene"]
    params = st.session_state["params"]
    masks = st.session_state["masks"]
    original = st.session_state["original"]

    st.markdown("---")
    st.markdown("## 🎯 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(images[0], caption="Original", width=350)

    with col2:
        st.image(enhanced, caption="Enhanced", width=350)

    # SLIDER
    st.markdown("### 🔄 Compare")
    comparison_slider(original, enhanced)

    # ===========================
# MOTION MASK (DEBUG VIEW)
# ===========================
if "masks" in st.session_state:

    masks = st.session_state["masks"]
    original = st.session_state["original"]

    if masks:
        st.markdown("### 🎯 Motion Mask")

        mask_vis = cv2.normalize(
            masks[0]['soft'],
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        col1, col2 = st.columns(2)

        with col1:
            st.image(original, caption="Original", width=250)

        with col2:
            st.image(mask_vis, caption="Motion Mask", width=250)

        # Debug values
        st.write("Mask min:", np.min(masks[0]['soft']))
        st.write("Mask max:", np.max(masks[0]['soft']))

    # METRICS
    st.markdown("### 🧠 Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brightness", f"{scene['brightness']:.2f}")
    c2.metric("Noise", f"{scene['noise']:.2f}")
    c3.metric("Contrast", f"{scene['contrast']:.2f}")
    c4.metric("Motion", f"{scene['motion']:.2f}")

    # PARAMETERS
    st.markdown("### ⚙️ Adaptive Parameters")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Gamma", params["gamma"])
    d2.metric("CLAHE", params["clahe_clip"])
    d3.metric("Denoise", params["denoise"])
    d4.metric("Frames", params["num_frames"])

    # GRAPH
    st.markdown("### 📊 Graph")

    g1, g2 = st.columns([1, 2])

    with g1:
       fig, ax = plt.subplots(figsize=(3, 2))
       ax.bar(list(scene.keys()), list(scene.values()))
       ax.set_title("Scene", fontsize=10)
       st.pyplot(fig)

    with g2:
       st.write("")  # empty space for balance

    # DOWNLOAD
    _, buffer = cv2.imencode(".png", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

    st.download_button(
        label="⬇ Download Image",
        data=buffer.tobytes(),
        file_name="enhanced.png",
        mime="image/png"
    )