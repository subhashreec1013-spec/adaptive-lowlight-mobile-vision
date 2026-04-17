"""
fusion.py
Adaptive Multi-frame fusion (FULLY FIXED VERSION)
"""

import cv2
import numpy as np
from core.semantic_mask import SemanticMaskGenerator


class MultiExposureFusion:
    def __init__(self):
        # Exposure brackets applied as LINEAR scale factors, not gamma.
        # These create darker / same / brighter versions of the gamma-lifted frame.
        self.exposure_factors = [0.75, 1.0, 1.25]

    def create_exposure_bracket(self, image, factor):
        """
        Scale brightness linearly (safe, no gamma stacking).
        """
        img_f = image.astype(np.float32) * factor
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def compute_weight_maps(self, images):
        weights = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Contrast weight
            contrast = cv2.Laplacian(gray_blur, cv2.CV_32F)
            contrast = cv2.normalize(np.abs(contrast), None, 0, 1, cv2.NORM_MINMAX)

            # Saturation weight
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0

            # Well-exposedness weight — target 0.5 (midtone)
            img_f = img.astype(np.float32) / 255.0
            well_exposed = np.exp(-0.5 * ((img_f - 0.5) ** 2) / 0.2)
            well_exposed = np.prod(well_exposed, axis=2)

            weight = (contrast + 0.3) * (saturation + 0.3) * (well_exposed + 0.4)
            weight = np.clip(weight, 0.01, 1.0)
            weight = cv2.GaussianBlur(weight, (7, 7), 0)
            weights.append(weight)

        weight_sum = np.maximum(np.sum(weights, axis=0), 1e-3)
        return [w / weight_sum for w in weights]

    def fuse_images(self, images, weights):
        fused = np.zeros_like(images[0], dtype=np.float32)
        for img, w in zip(images, weights):
            w3 = np.stack([w] * 3, axis=2)
            fused += img.astype(np.float32) * w3
        return np.clip(fused, 0, 255).astype(np.uint8)


def post_process(image, clahe_clip=3.0, denoise_h=6):
    """
    CLAHE contrast enhancement + single denoise pass.
    FIX: denoise now uses the adaptive param instead of hardcoded value.
    FIX: only ONE denoise call (old code called it twice).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # Single denoise — strength driven by adaptive params
    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced, None, denoise_h, denoise_h, 7, 21
    )
    return enhanced


def enhance_low_light(frames, flows=None, masks=None, params=None):

    print("Starting low-light enhancement...")

    # ── Adaptive params ──────────────────────────────────────────────────────
    gamma      = params.get("gamma",      0.50) if params else 0.50
    clahe_clip = params.get("clahe_clip", 3.0)  if params else 3.0
    denoise_h  = params.get("denoise",    6)     if params else 6
    print(f"Using Gamma: {gamma}, CLAHE clip: {clahe_clip}, Denoise h: {denoise_h}")

    fusion = MultiExposureFusion()

    # ── Step 1: Gamma lift (once, on every frame) ────────────────────────────
    lifted = []
    for frame in frames:
        img_f = frame.astype(np.float32) / 255.0
        img_lifted = np.power(img_f, gamma)
        lifted.append(np.clip(img_lifted * 255, 0, 255).astype(np.uint8))

    # ── Step 2: Exposure brackets from the LIFTED frame (no double gamma) ────
    # FIX: Old code applied gamma again inside adjust_exposure → double gamma
    #      → image was being gamma-corrected twice → way too bright / washed out
    reference = lifted[0]
    exposures = [
        fusion.create_exposure_bracket(reference, f)
        for f in fusion.exposure_factors
    ]

    # ── Step 3: Weight maps ──────────────────────────────────────────────────
    weights = fusion.compute_weight_maps(exposures)

    # ── Step 4: Motion mask (optional) ──────────────────────────────────────
    if masks is not None and len(masks) > 0:
        print("Applying motion-aware weighting...")
        motion_mask = masks[0]['soft']
        if motion_mask.shape != weights[0].shape:
            motion_mask = cv2.resize(
                motion_mask,
                (weights[0].shape[1], weights[0].shape[0])
            )
        motion_mask = cv2.GaussianBlur(motion_mask, (7, 7), 0)
        for i in range(len(weights)):
            weights[i] = weights[i] * (0.9 + 0.1 * motion_mask)
        weight_sum = np.maximum(np.sum(weights, axis=0), 1e-3)
        weights = [w / weight_sum for w in weights]

    # ── Step 5: Fuse ─────────────────────────────────────────────────────────
    fused = fusion.fuse_images(exposures, weights)

    # ── Step 6: Post-process (CLAHE + single denoise) ────────────────────────
    enhanced = post_process(fused, clahe_clip=clahe_clip, denoise_h=denoise_h)

    # ── Step 7: Semantic face protection ────────────────────────────────────
    semantic = SemanticMaskGenerator()
    enhanced = semantic.apply_semantic_protection(frames[0], enhanced)

    # ── Step 8: Final gentle lift ────────────────────────────────────────────
    # alpha=1.05, beta=8 → just a nudge, not a second full brighten
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=8)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # ── Save debug outputs ───────────────────────────────────────────────────
    cv2.imwrite('results/fused.png',    cv2.cvtColor(fused,    cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/enhanced.png', cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

    print("Enhancement complete!")
    return enhanced