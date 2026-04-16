"""
optical_flow.py
Stable optical flow computation (FINAL VERSION)
"""

import cv2
import numpy as np


class OpticalFlowCalculator:

    def __init__(self):
        pass

    def compute(self, prev, next):
        """
        Compute optical flow between two frames
        """

        # =========================
        # STEP 1: Ensure same size
        # =========================
        if prev.shape != next.shape:
            next = cv2.resize(next, (prev.shape[1], prev.shape[0]))

        # =========================
        # STEP 2: Convert to grayscale
        # =========================
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)

        # =========================
        # STEP 3: Denoise (IMPORTANT)
        # =========================
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)

        # =========================
        # STEP 4: Optical Flow
        # =========================
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        return flow


def compute_sequence_flow(frames):
    """
    Compute optical flow for sequence of frames
    """

    calculator = OpticalFlowCalculator()
    flows = []

    # Edge case
    if len(frames) < 2:
        return flows

    for i in range(len(frames) - 1):
        flow = calculator.compute(frames[i], frames[i + 1])
        flows.append(flow)

    return flows