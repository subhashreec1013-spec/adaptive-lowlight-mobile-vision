
"""
optical_flow.py
Computes optical flow between frames
"""

import cv2
import numpy as np
from config import config

class OpticalFlowCalculator:
    def __init__(self, method=None):
        self.method = method or config.optical_flow.method
    
    def compute(self, prev_frame, next_frame):
        """Compute optical flow between two frames"""
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_frame
            next_gray = next_frame
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None,
            config.optical_flow.pyr_scale,
            config.optical_flow.levels,
            config.optical_flow.winsize,
            config.optical_flow.iterations,
            config.optical_flow.poly_n,
            config.optical_flow.poly_sigma,
            0
        )
        return flow
    
    def visualize_flow(self, flow):
        """Visualize optical flow"""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def compute_sequence_flow(frames):
    """Compute optical flow for a sequence"""
    calculator = OpticalFlowCalculator()
    flows = []
    
    for i in range(len(frames) - 1):
        flow = calculator.compute(frames[i], frames[i+1])
        flows.append(flow)
    
    # Save visualization
    if flows:
        flow_vis = calculator.visualize_flow(flows[0])
        cv2.imwrite('results/optical_flow_vis.png', flow_vis)
    
    return flows