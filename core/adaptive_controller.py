import numpy as np


class AdaptiveController:
    def __init__(self):
        pass

    def decide_parameters(self, scene_info):
        """
        Decide enhancement parameters based on scene analysis
        """

        print("Applying adaptive decision logic...")

        brightness = scene_info["brightness"]
        noise = scene_info["noise"]
        contrast = scene_info["contrast"]
        motion = scene_info["motion"]

        # =========================
        # GAMMA DECISION
        # =========================
        if brightness < 0.2:
            gamma = 0.5
        elif brightness < 0.4:
            gamma = 0.7
        else:
            gamma = 1.0

        # =========================
        # CLAHE DECISION
        # =========================
        if contrast < 0.3:
            clahe_clip = 3.5
        elif contrast < 0.5:
            clahe_clip = 2.5
        else:
            clahe_clip = 2.0

        # =========================
        # DENOISE DECISION
        # =========================
        if noise > 0.4:
            denoise = 20
        elif noise > 0.2:
            denoise = 10
        else:
            denoise = 5

        # =========================
        # FRAME USAGE (SMART)
        # =========================
        if motion > 0.3:
            num_frames = 2   # reduce ghosting
        else:
            num_frames = 3

        # =========================
        # RESULT
        # =========================
        params = {
            "gamma": gamma,
            "clahe_clip": clahe_clip,
            "denoise": denoise,
            "num_frames": num_frames
        }

        print("Adaptive Parameters:", params)

        return params