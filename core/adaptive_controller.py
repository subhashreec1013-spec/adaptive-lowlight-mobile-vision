"""
adaptive_controller.py
Adaptive parameter selection based on scene analysis (FIXED VERSION)
"""

class AdaptiveController:

    def __init__(self):
        pass

    def decide_parameters(self, scene):
        """
        Decide enhancement parameters based on scene features.
        """

        brightness = scene["brightness"]
        noise      = scene["noise"]
        motion     = scene["motion"]

        # ─────────────────────────────────────────
        # GAMMA  (controls brightness lift)
        # FIX: Lower gamma = more brightening.
        # Very dark images need the most lift (lowest gamma).
        # Old code gave dark images gamma=0.8 (almost no lift).
        # ─────────────────────────────────────────
        if brightness < 0.15:
            gamma = 0.40        # very dark  → strong lift
        elif brightness < 0.30:
            gamma = 0.50        # dark       → good lift
        elif brightness < 0.50:
            gamma = 0.65        # mid-dark   → moderate lift
        else:
            gamma = 0.80        # reasonable → gentle lift

        # ─────────────────────────────────────────
        # CLAHE CLIP  (controls local contrast)
        # FIX: Dark + low-noise images need higher clip to recover detail.
        # Noisy images need a lower clip to avoid amplifying noise.
        # Old code gave all images 1.5–2.0 which is too weak.
        # ─────────────────────────────────────────
        if noise > 0.4:
            clahe_clip = 2.5    # noisy  → moderate, don't amplify noise
        elif noise > 0.2:
            clahe_clip = 3.0    # medium → balanced
        else:
            clahe_clip = 3.5    # clean  → stronger contrast recovery

        # ─────────────────────────────────────────
        # DENOISE  (h param for fastNlMeansDenoisingColored)
        # FIX: Scale properly to noise level.
        # h=5  → very mild (clean images, preserve detail)
        # h=8  → moderate
        # h=12 → strong (very noisy, accept some blur)
        # Old code used 10/15 for wrong cases and wrong direction.
        # ─────────────────────────────────────────
        if noise > 0.5:
            denoise = 12        # very noisy → stronger denoising
        elif noise > 0.3:
            denoise = 8         # moderate   → balanced
        else:
            denoise = 5         # clean      → light touch only

        # ─────────────────────────────────────────
        # FRAME COUNT  (how many burst frames to fuse)
        # High motion → fewer frames avoids ghosting
        # Low motion  → more frames = better noise averaging
        # (unchanged, this logic was correct)
        # ─────────────────────────────────────────
        if motion > 0.5:
            num_frames = 2
        else:
            num_frames = 3

        return {
            "gamma":      gamma,
            "clahe_clip": clahe_clip,
            "denoise":    denoise,
            "num_frames": num_frames
        }