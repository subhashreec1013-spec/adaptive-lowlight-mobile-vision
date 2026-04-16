"""
motion_mask.py
Stable motion mask generation (FINAL VERSION)
"""

import cv2
import numpy as np


def create_motion_masks(flows):
    masks = []

    for flow in flows:

        # =========================
        # STEP 1: Motion Magnitude
        # =========================
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # =========================
        # STEP 2: Robust Normalization (IMPORTANT)
        # =========================
        mag = mag / (np.percentile(mag, 95) + 1e-6)
        mag = np.clip(mag, 0, 1)

        # =========================
        # STEP 3: Smooth Noise
        # =========================
        mag = cv2.GaussianBlur(mag, (7, 7), 0)

        # =========================
        # STEP 4: Adaptive Threshold
        # =========================
        threshold = np.mean(mag) + 0.5 * np.std(mag)
        motion_mask = (mag > threshold).astype(np.float32)

        # =========================
        # STEP 5: Smooth Mask Edges
        # =========================
        motion_mask = cv2.GaussianBlur(motion_mask, (7, 7), 0)

        masks.append({
            "soft": motion_mask
        })

    return masks