"""
adaptive_controller.py
Adaptive parameter selection based on scene analysis
"""

class AdaptiveController:

    def __init__(self):
        pass

    def decide_parameters(self, scene):
        """
        Decide enhancement parameters based on scene features
        """

        # ✅ FIX: Proper indentation starts here
        brightness = scene["brightness"]
        noise = scene["noise"]
        motion = scene["motion"]

        # 🔥 Safe gamma
        if brightness < 0.3:
            gamma = 0.8
        else:
            gamma = 1.0

        # 🔥 Safe CLAHE
        if noise > 0.3:
            clahe_clip = 2.0
        else:
            clahe_clip = 1.5

        # 🔥 Denoising
        if noise > 0.3:
            denoise = 15
        else:
            denoise = 10

        # 🔥 Frame selection
        if motion > 0.5:
            num_frames = 2
        else:
            num_frames = 3

        return {
            "gamma": gamma,
            "clahe_clip": clahe_clip,
            "denoise": denoise,
            "num_frames": num_frames
        }