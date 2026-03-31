"""
main_pipeline.py
Main pipeline - Adaptive Intelligent Version
"""

import cv2
import numpy as np
import os
import time

from config import config

# Core modules
from dataset.dataset_preparation import create_dataloaders
from core.optical_flow import compute_sequence_flow
from core.motion_mask import create_motion_masks
from core.fusion import enhance_low_light
from core.scene_analysis import SceneAnalyzer
from core.adaptive_controller import AdaptiveController
from core.temporal_smoothing import TemporalSmoother


class LowLightEnhancementPipeline:
    def __init__(self):
        self.config = config

        # 🔥 NEW (INTELLIGENCE MODULES)
        self.scene_analyzer = SceneAnalyzer()
        self.adaptive_controller = AdaptiveController()
        self.temporal_smoother = TemporalSmoother()
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        print("=" * 60)
        print("LOW-LIGHT IMAGE ENHANCEMENT PIPELINE")
        print("=" * 60)

    def run(self):
        """Run complete pipeline"""
        print("\n[Step 1] Loading data...")
        dataloader = create_dataloaders()

        for batch_idx, batch in enumerate(dataloader):
            print(f"\nProcessing sequence {batch_idx + 1}")

            # =========================
            # GET FRAMES
            # =========================
            frames_tensor = batch['frames']
            frames = [
                frames_tensor[0][i].permute(1, 2, 0).numpy()
                for i in range(frames_tensor.shape[1])
            ]
            frames = [(f * 255).astype(np.uint8) for f in frames]

            # =========================
            # STEP 2: OPTICAL FLOW
            # =========================
            print("[Step 2] Computing optical flow...")
            flows = compute_sequence_flow(frames)

            # =========================
            # STEP 3: MOTION MASK
            # =========================
            print("[Step 3] Generating motion masks...")
            masks = create_motion_masks(flows)

            # =========================
            # STEP 4: SCENE ANALYSIS 🚨
            # =========================
            scene_info = self.scene_analyzer.analyze(frames, flows)

            # =========================
            # STEP 5: ADAPTIVE CONTROL 🚨
            # =========================
            params = self.adaptive_controller.decide_parameters(scene_info)
            frames = frames[:params['num_frames']]  # Smart frame selection

            # =========================
            # STEP 6: ENHANCEMENT 🚨
            # =========================
            print("[Step 6] Enhancing images with adaptive parameters...")
            enhanced = enhance_low_light(
                frames,
                flows,
                masks,
                params  # 🔥 NEW PARAMS
            )
            # =========================
            # STEP 7: TEMPORAL SMOOTHING
            # =========================
            enhanced = self.temporal_smoother.smooth(enhanced)
            # =========================
            # SAVE RESULTS
            # =========================
            print("[Step 7] Saving results...")
            cv2.imwrite(
                os.path.join(self.results_dir, f"enhanced_{batch_idx}.png"),
                cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            )

            print(f"✅ Sequence {batch_idx + 1} complete")

        print("\n✅ PIPELINE COMPLETE")
        print(f"Results saved to: {self.results_dir}/")


def main():
    print("\n" + "=" * 60)
    print("LOW-LIGHT IMAGE ENHANCEMENT - ADAPTIVE PIPELINE")
    print("=" * 60)

    pipeline = LowLightEnhancementPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()