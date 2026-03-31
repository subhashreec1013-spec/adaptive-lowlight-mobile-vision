import cv2
import numpy as np


class SceneAnalyzer:
    def __init__(self):
        pass

    def analyze(self, frames, flows=None):
        """
        Analyze scene properties:
        - Brightness
        - Noise
        - Motion
        - Contrast
        """

        print("Analyzing scene...")

        # Use first frame as reference
        frame = frames[0]

        # =========================
        # BRIGHTNESS
        # =========================
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)

        # =========================
        # NOISE ESTIMATION
        # =========================
        noise = np.std(gray)

        # =========================
        # CONTRAST
        # =========================
        contrast = np.std(gray)

        # =========================
        # MOTION LEVEL
        # =========================
        motion = 0
        if flows is not None and len(flows) > 0:
            mag, _ = cv2.cartToPolar(flows[0][..., 0], flows[0][..., 1])
            motion = np.mean(mag)

        # =========================
        # NORMALIZATION (0 to 1)
        # =========================
        brightness_norm = brightness / 255.0
        noise_norm = noise / 100.0
        contrast_norm = contrast / 100.0
        motion_norm = min(motion / 10.0, 1.0)

        # =========================
        # RESULT DICTIONARY
        # =========================
        scene_info = {
            "brightness": brightness_norm,
            "noise": noise_norm,
            "contrast": contrast_norm,
            "motion": motion_norm
        }

        print("Scene Analysis:", scene_info)

        return scene_info