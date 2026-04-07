"""
optical_flow.py
Compute optical flow between frames (SAFE VERSION)
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

        # ✅ FIX 1: Ensure same size
        if prev.shape != next.shape:
            next = cv2.resize(next, (prev.shape[1], prev.shape[0]))

        # ✅ FIX 2: Convert to grayscale
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)

        # Compute flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        return flow


def compute_sequence_flow(frames):
    """
    Compute optical flow for sequence of frames
    """

    calculator = OpticalFlowCalculator()
    flows = []

    for i in range(len(frames) - 1):
        flow = calculator.compute(frames[i], frames[i + 1])
        flows.append(flow)

    return flows