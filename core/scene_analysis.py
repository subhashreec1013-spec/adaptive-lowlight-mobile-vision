"""
scene_analysis.py
Scene property analysis (FIXED VERSION)
"""

import cv2
import numpy as np


class SceneAnalyzer:
    def __init__(self):
        pass

    def analyze(self, frames, flows=None):
        """
        Analyze scene properties:
        - Brightness (0-1)
        - Noise     (0-1)
        - Contrast  (0-1)
        - Motion    (0-1)
        """
        print("Analyzing scene...")

        frame = frames[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # ─────────────────────────────────────────
        # BRIGHTNESS  — mean pixel / 255
        # ─────────────────────────────────────────
        brightness_norm = np.mean(gray) / 255.0

        # ─────────────────────────────────────────
        # NOISE — FIX: was identical to contrast (both used np.std)
        # Proper estimator: blur the frame, measure residual high-freq energy
        # A clean image has very low residual; a noisy image has high residual.
        # Normalize by 255 so the result is always in 0-1.
        # ─────────────────────────────────────────
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = np.abs(gray - blurred)
        noise_norm = np.clip(np.mean(residual) / 30.0, 0.0, 1.0)
        # divide by 30 → typical noisy dark image gives ~0.3-0.5, clean gives ~0.05-0.15

        # ─────────────────────────────────────────
        # CONTRAST — FIX: use Michelson contrast (max-min)/(max+min)
        # Much more meaningful than std for choosing CLAHE strength.
        # ─────────────────────────────────────────
        p5  = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        denom = p95 + p5 + 1e-6
        contrast_norm = np.clip((p95 - p5) / denom, 0.0, 1.0)

        # ─────────────────────────────────────────
        # MOTION — mean optical flow magnitude, capped at 1
        # ─────────────────────────────────────────
        motion_norm = 0.0
        if flows is not None and len(flows) > 0:
            mag, _ = cv2.cartToPolar(flows[0][..., 0], flows[0][..., 1])
            motion_norm = float(np.clip(np.mean(mag) / 10.0, 0.0, 1.0))

        scene_info = {
            "brightness": brightness_norm,
            "noise":      noise_norm,
            "contrast":   contrast_norm,
            "motion":     motion_norm,
        }

        print("Scene Analysis:", scene_info)
        return scene_info