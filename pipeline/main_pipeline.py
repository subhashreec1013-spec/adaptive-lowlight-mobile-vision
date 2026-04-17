"""
main_pipeline.py
Clean + Correct Adaptive Pipeline (FINAL)
"""

import cv2
import numpy as np
import os

from config import config

from core.optical_flow import compute_sequence_flow
from core.motion_mask import create_motion_masks
from core.fusion import enhance_low_light
from core.scene_analysis import SceneAnalyzer
from core.adaptive_controller import AdaptiveController


class LowLightEnhancementPipeline:
    def __init__(self):
        self.config = config

        self.scene_analyzer = SceneAnalyzer()
        self.adaptive_controller = AdaptiveController()

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        print("=" * 60)
        print("LOW-LIGHT ENHANCEMENT PIPELINE (CLEAN)")
        print("=" * 60)

    def process(self, frames):
        """
        Main pipeline for given frames
        (USED BY STREAMLIT)
        """
        valid_frames = [f for f in frames if f is not None]

        h, w = valid_frames[0].shape[:2]
        frames = [cv2.resize(f, (w, h)) for f in valid_frames]

        # =========================
        # STEP 1: FLOW
        # =========================
        flows = compute_sequence_flow(frames)

        # =========================
        # STEP 2: SCENE ANALYSIS
        # =========================
        scene = self.scene_analyzer.analyze(frames, flows)

        # =========================
        # STEP 3: ADAPTIVE PARAMS
        # =========================
        params = self.adaptive_controller.decide_parameters(scene)

        # =========================
        # STEP 4: FRAME SELECTION
        # =========================
        frames = frames[:params["num_frames"]]

        # 🔥 IMPORTANT: recompute flow after trimming
        flows = compute_sequence_flow(frames)
        masks = create_motion_masks(flows)

        # =========================
        # STEP 5: ENHANCEMENT
        # =========================
        enhanced = enhance_low_light(frames, flows, masks, params)

        return enhanced, scene, params, masks


# OPTIONAL (for testing only)
def main():
    print("Standalone pipeline test")

    # dummy test image
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frames = [img, img]

    pipeline = LowLightEnhancementPipeline()
    enhanced, _, _, _ = pipeline.process(frames)

    cv2.imwrite("results/output.png", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()