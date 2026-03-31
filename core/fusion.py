"""
fusion.py
Adaptive Multi-exposure fusion with motion-aware weighting (FINAL PATENT VERSION)
"""

import cv2
import numpy as np
from config import config
from core.semantic_mask import SemanticMaskGenerator


class MultiExposureFusion:
    def __init__(self):
        self.exposure_levels = [0.5, 1.0, 1.5, 2.0]

    def adjust_exposure(self, image, gamma):
        """Adjust exposure using gamma correction"""
        img_norm = image.astype(np.float32) / 255.0
        adjusted = np.power(img_norm, 1 / gamma)
        return np.clip(adjusted * 255, 0, 255).astype(np.uint8)

    def compute_weight_maps(self, images):
        """Compute weight maps for fusion"""
        weights = []

        for img in images:
            # Contrast
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            contrast = cv2.Laplacian(gray, cv2.CV_32F)
            contrast = np.abs(contrast)
            contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)

            # Saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0

            # Well-exposedness
            img_float = img.astype(np.float32) / 255.0
            well_exposed = 1.0 - np.abs(img_float - 0.5) * 2
            well_exposed = np.prod(well_exposed, axis=2)

            # Combine
            weight = contrast * saturation * well_exposed
            weight = cv2.normalize(weight, None, 0, 1, cv2.NORM_MINMAX)

            weights.append(weight)

        # Normalize weights
        weight_sum = np.sum(weights, axis=0) + 1e-6
        weights = [w / weight_sum for w in weights]

        return weights

    def fuse_images(self, images, weights):
        """Fuse multiple exposures"""
        fused = np.zeros_like(images[0].astype(np.float32))

        for img, weight in zip(images, weights):
            weight_3ch = np.stack([weight] * 3, axis=2)
            fused += img.astype(np.float32) * weight_3ch

        return np.clip(fused, 0, 255).astype(np.uint8)


# ================================
# UPDATED POST PROCESS (ADAPTIVE)
# ================================
def post_process(image, clahe_clip=2.0, denoise=10):
    """Apply post-processing (CLAHE + Denoising)"""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8, 8)
    )

    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Denoising
    if denoise > 0:
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced, None, denoise, denoise, 7, 21
        )

    return enhanced


# ================================
# MAIN FUNCTION (ADAPTIVE + MOTION)
# ================================
def enhance_low_light(frames, flows=None, masks=None, params=None):
    """
    Main enhancement function (Adaptive + Motion-Aware Fusion)
    """
    print("Starting low-light enhancement...")

    fusion = MultiExposureFusion()

    # ================================
    # STEP 0: APPLY ADAPTIVE PARAMETERS 🚨
    # ================================
    if params is not None:
        gamma = params.get("gamma", 1.0)
        clahe_clip = params.get("clahe_clip", 2.0)
        denoise = params.get("denoise", 10)
    else:
        gamma = 1.0
        clahe_clip = 2.0
        denoise = 10

    print(f"Using params → Gamma: {gamma}, CLAHE: {clahe_clip}, Denoise: {denoise}")

    # ================================
    # STEP 1: CREATE EXPOSURES
    # ================================
    exposures = []
    for g in fusion.exposure_levels:
        adjusted = fusion.adjust_exposure(frames[0], g * gamma)
        exposures.append(adjusted)

    # ================================
    # STEP 2: COMPUTE WEIGHTS
    # ================================
    weights = fusion.compute_weight_maps(exposures)

    # ================================
    # STEP 3: APPLY MOTION MASK
    # ================================
    if masks is not None and len(masks) > 0:
        print("Applying motion-aware weighting...")

        motion_mask = masks[0]['soft']

        # Resize if needed
        if motion_mask.shape != weights[0].shape:
            motion_mask = cv2.resize(
                motion_mask,
                (weights[0].shape[1], weights[0].shape[0])
            )

        # Apply mask
        for i in range(len(weights)):
            weights[i] = weights[i] * motion_mask

        # Normalize again
        weight_sum = np.sum(weights, axis=0) + 1e-6
        weights = [w / weight_sum for w in weights]

    # ================================
    # STEP 4: FUSION
    # ================================
    fused = fusion.fuse_images(exposures, weights)

    # ================================
    # STEP 5: POST-PROCESSING   
    # ================================
    enhanced = post_process(fused, clahe_clip, denoise)
    # ================================
    # STEP 6: SEMANTIC PROTECTION 🚨
    # ================================
    semantic = SemanticMaskGenerator()
    enhanced = semantic.apply_semantic_protection(frames[0], enhanced)

    # ================================
    # SAVE DEBUG OUTPUT
    # ================================
    cv2.imwrite('results/fused.png', fused)
    cv2.imwrite('results/enhanced.png', enhanced)

    print("Enhancement complete!")
    return enhanced