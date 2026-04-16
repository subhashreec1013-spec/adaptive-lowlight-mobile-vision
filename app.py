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
st.set_page_config(page_title="LuminaEnhance Pro", page_icon="✦", layout="wide")

# ===========================
# PRO UI CSS
# ===========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #080a0f !important;
    color: #e2e4ea !important;
    font-family: 'DM Sans', sans-serif;
}

/* Grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
.viewerBadge_container__r5tak { display: none !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f17; }
::-webkit-scrollbar-thumb { background: #d4a017; border-radius: 2px; }

/* ── HERO HEADER ── */
.hero-wrapper {
    position: relative;
    text-align: center;
    padding: 56px 24px 40px;
    overflow: hidden;
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 700px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(212,160,23,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #d4a017;
    border: 1px solid rgba(212,160,23,0.3);
    padding: 5px 14px;
    border-radius: 2px;
    margin-bottom: 20px;
    background: rgba(212,160,23,0.05);
}

.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #d4a017;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 8vw, 6.5rem);
    line-height: 0.95;
    letter-spacing: 0.03em;
    color: #f0f2f7;
    margin-bottom: 6px;
}

.hero-title span {
    color: #d4a017;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.04em;
    margin-top: 12px;
    font-style: italic;
}

/* ── STAT PILLS ── */
.pills-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 28px;
}

.pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 8px 18px;
    border-radius: 2px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #9ca3af;
    transition: all 0.25s ease;
    letter-spacing: 0.02em;
}

.pill:hover {
    background: rgba(212,160,23,0.07);
    border-color: rgba(212,160,23,0.3);
    color: #d4a017;
}

.pill-icon {
    font-size: 1rem;
    opacity: 0.85;
}

/* ── DIVIDER ── */
.luxury-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 32px 0 24px;
}

.luxury-divider::before,
.luxury-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,160,23,0.25), transparent);
}

.luxury-divider-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #d4a017;
    opacity: 0.7;
    white-space: nowrap;
}

/* ── SECTION HEADERS ── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin: 36px 0 16px;
}

.section-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #d4a017;
    opacity: 0.6;
    letter-spacing: 0.1em;
}

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.06em;
    color: #f0f2f7;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    background: rgba(212,160,23,0.03) !important;
    border: 1px dashed rgba(212,160,23,0.2) !important;
    border-radius: 4px !important;
    padding: 8px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(212,160,23,0.5) !important;
    background: rgba(212,160,23,0.06) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 24px !important;
}

.uploadLabel {
    color: #d4a017 !important;
}

/* ── ENHANCE BUTTON ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #d4a017 0%, #b8860b 50%, #d4a017 100%) !important;
    background-size: 200% auto !important;
    color: #080a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 14px 48px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background-position 0.4s ease, transform 0.2s ease, box-shadow 0.3s ease !important;
    box-shadow: 0 0 30px rgba(212,160,23,0.15), 0 4px 20px rgba(0,0,0,0.4) !important;
    margin-top: 16px !important;
}

[data-testid="stButton"] > button:hover {
    background-position: right center !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 50px rgba(212,160,23,0.3), 0 8px 30px rgba(0,0,0,0.5) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #d4a017 !important;
    border: 1px solid rgba(212,160,23,0.4) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    border-radius: 2px !important;
    padding: 12px 32px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: rgba(212,160,23,0.08) !important;
    border-color: #d4a017 !important;
    box-shadow: 0 0 20px rgba(212,160,23,0.15) !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 4px !important;
    padding: 18px 20px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: border-color 0.3s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(212,160,23,0.25) !important;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #d4a017, transparent);
    opacity: 0.6;
}

[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.8rem !important;
    color: #f0f2f7 !important;
    letter-spacing: 0.05em !important;
}

/* ── IMAGE CONTAINERS ── */
[data-testid="stImage"] {
    border-radius: 4px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
    transition: box-shadow 0.3s ease !important;
}

[data-testid="stImage"]:hover {
    box-shadow: 0 8px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(212,160,23,0.15) !important;
}

/* ── CAPTIONS ── */
[data-testid="stCaptionContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4b5563 !important;
    margin-top: 8px !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] > div {
    border-color: #d4a017 !important;
    border-top-color: transparent !important;
}

[data-testid="stSpinner"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: #6b7280 !important;
}

/* ── SUCCESS ── */
[data-testid="stAlert"] {
    background: rgba(212,160,23,0.07) !important;
    border: 1px solid rgba(212,160,23,0.25) !important;
    border-radius: 2px !important;
    color: #d4a017 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #d4a017 !important;
}

[data-testid="stSlider"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    color: #6b7280 !important;
}

/* ── MATPLOTLIB CHART ── */
[data-testid="stPyplotFigure"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

/* ── COLUMNS ── */
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── RESULT BADGE ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.25);
    background: rgba(34,197,94,0.06);
    padding: 6px 14px;
    border-radius: 2px;
    margin-bottom: 24px;
}

.result-badge::before {
    content: '✓';
    font-size: 0.75rem;
}

/* ── FRAME GRID WRAPPER ── */
.frame-index {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #4b5563;
    letter-spacing: 0.1em;
    text-align: center;
    margin-top: 4px;
}

/* ── FOOTER ── */
.pro-footer {
    text-align: center;
    margin-top: 64px;
    padding: 32px 0;
    border-top: 1px solid rgba(255,255,255,0.05);
}

.pro-footer p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #374151;
}

.pro-footer span {
    color: #d4a017;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# HERO HEADER
# ===========================
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">Adaptive Multi-Frame Enhancement Engine</div>
    <div class="hero-title">LUMINA<span>ENHANCE</span></div>
    <div class="hero-title" style="font-size: clamp(1.8rem, 4vw, 3rem); color: #3a3d47; margin-top: -6px;">PRO EDITION</div>
    <div class="hero-subtitle">Neural-guided burst photography restoration — from shadow to clarity</div>
    <div class="pills-row">
        <div class="pill"><span class="pill-icon">🧠</span> AI Adaptive Engine</div>
        <div class="pill"><span class="pill-icon">📸</span> Multi-Frame Fusion</div>
        <div class="pill"><span class="pill-icon">🎯</span> Motion-Aware Masking</div>
        <div class="pill"><span class="pill-icon">👤</span> Face-Safe Processing</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===========================
# PIPELINE
# ===========================
def process_pipeline(images):
    frames = [np.array(img.convert("RGB")) for img in images]
    enhanced, scene, params, masks = pipeline.process(frames)
    return enhanced, scene, params, masks

# ===========================
# COMPARISON SLIDER
# ===========================
def comparison_slider(orig, enh):
    st.markdown("""
    <div class="luxury-divider">
        <span class="luxury-divider-label">Interactive Comparison</span>
    </div>
    """, unsafe_allow_html=True)

    alpha = st.slider("◀  Original / Enhanced  ▶", 0, 100, 50, label_visibility="visible")

    orig = orig.convert("RGB")
    enh = Image.fromarray(enh).convert("RGB")
    blend = (alpha / 100) * np.array(enh) + (1 - alpha / 100) * np.array(orig)

    st.image(blend.astype(np.uint8), width=500)

# ===========================
# UPLOAD SECTION
# ===========================
st.markdown("""
<div class="luxury-divider">
    <span class="luxury-divider-label">Begin</span>
</div>
<div class="section-header">
    <span class="section-number">01 —</span>
    <span class="section-title">Upload Burst Frames</span>
</div>
""", unsafe_allow_html=True)

files = st.file_uploader(
    "Drop your burst images here — PNG, JPG, or JPEG",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    label_visibility="visible"
)

# ===========================
# MAIN PROCESSING
# ===========================
if files:
    images = [Image.open(f) for f in files]

    st.markdown("""
    <div class="section-header">
        <span class="section-number">02 —</span>
        <span class="section-title">Input Frames</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(min(len(images), 3))
    for i, img in enumerate(images):
        with cols[i % 3]:
            st.image(img, width=320)
            st.markdown(f"<div class='frame-index'>FRAME_{i+1:02d}.RAW</div>", unsafe_allow_html=True)

    if st.button("✦  ENHANCE NOW  ✦"):
        with st.spinner("Processing burst sequence..."):
            start = time.time()
            enhanced, scene, params, masks = process_pipeline(images)
            end = time.time()

        st.session_state["enhanced"] = enhanced
        st.session_state["scene"] = scene
        st.session_state["params"] = params
        st.session_state["masks"] = masks
        st.session_state["original"] = images[0]
        st.session_state["elapsed"] = end - start

        st.success(f"Enhancement complete — processed in {end - start:.2f}s")

# ===========================
# RESULTS
# ===========================
if "enhanced" in st.session_state:

    enhanced = st.session_state["enhanced"]
    scene    = st.session_state["scene"]
    params   = st.session_state["params"]
    masks    = st.session_state["masks"]
    original = st.session_state["original"]
    elapsed  = st.session_state.get("elapsed", 0)

    st.markdown("""
    <div class="luxury-divider">
        <span class="luxury-divider-label">Output</span>
    </div>
    <div class="section-header">
        <span class="section-number">03 —</span>
        <span class="section-title">Enhancement Result</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(original, caption="Original · Source Frame", width=420)
    with col2:
        st.image(enhanced, caption="Enhanced · LuminaEnhance Pro", width=420)

    comparison_slider(original, enhanced)

    # ── MOTION MASK ──
    if masks is not None and len(masks) > 0:
        st.markdown("""
        <div class="section-header" style="margin-top:40px;">
            <span class="section-number">04 —</span>
            <span class="section-title">Motion Mask Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        mask = masks[0]['soft']
        mask_vis = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.image(original, caption="Source · Reference Frame", width=350)
        with col2:
            st.image(mask_vis, caption="Motion Mask · Soft Confidence Map", width=350)

    # ── SCENE ANALYSIS ──
    st.markdown("""
    <div class="section-header" style="margin-top:40px;">
        <span class="section-number">05 —</span>
        <span class="section-title">Scene Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4, gap="small")
    a1.metric("Brightness", f"{scene['brightness']:.2f}")
    a2.metric("Noise", f"{scene['noise']:.2f}")
    a3.metric("Contrast", f"{scene['contrast']:.2f}")
    a4.metric("Motion", f"{scene['motion']:.2f}")

    # ── ADAPTIVE PARAMS ──
    st.markdown("""
    <div class="section-header" style="margin-top:32px;">
        <span class="section-number">06 —</span>
        <span class="section-title">Adaptive Parameters</span>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4, gap="small")
    p1.metric("Gamma", params["gamma"])
    p2.metric("CLAHE Clip", params["clahe_clip"])
    p3.metric("Denoise Str.", params["denoise"])
    p4.metric("Frames Used", params["num_frames"])

    # ── GRAPH ──
    st.markdown("""
    <div class="section-header" style="margin-top:32px;">
        <span class="section-number">07 —</span>
        <span class="section-title">Scene Profile</span>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0d0f17')
    ax.set_facecolor('#0d0f17')

    keys = list(scene.keys())
    vals = list(scene.values())
    colors = ['#d4a017' if v == max(vals) else '#2a2d38' for v in vals]
    bars = ax.bar(keys, vals, color=colors, width=0.5, edgecolor='#1e2130', linewidth=1)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.2f}', ha='center', va='bottom',
                color='#9ca3af', fontsize=8,
                fontfamily='monospace')

    ax.set_title("Scene Characteristics Profile", color='#4b5563',
                 fontsize=9, fontfamily='monospace', loc='left', pad=10)
    ax.tick_params(colors='#4b5563', labelsize=8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#1e2130')
    ax.yaxis.set_visible(False)
    ax.set_ylim(0, max(vals) * 1.25)
    plt.tight_layout()
    st.pyplot(fig)

    # ── DOWNLOAD ──
    st.markdown("""
    <div class="luxury-divider" style="margin-top:40px;">
        <span class="luxury-divider-label">Export</span>
    </div>
    """, unsafe_allow_html=True)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    st.download_button(
        label="⬇  DOWNLOAD ENHANCED IMAGE",
        data=buffer.tobytes(),
        file_name="lumina_enhanced.png",
        mime="image/png"
    )

# ===========================
# FOOTER
# ===========================
st.markdown("""
<div class="pro-footer">
    <p><span>✦ LuminaEnhance Pro</span> — Adaptive Multi-Frame Low-Light Enhancement &nbsp;|&nbsp; Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)