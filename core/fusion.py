"""
fusion.py
Adaptive Multi-frame fusion with motion-aware weighting (FINAL BALANCED VERSION)
"""

import cv2
import numpy as np
from core.semantic_mask import SemanticMaskGenerator


class MultiExposureFusion:
    def __init__(self):
        self.exposure_levels = [0.9, 1.0, 1.1]

    def adjust_exposure(self, image, gamma):
        img_norm = image.astype(np.float32) / 255.0
        adjusted = np.power(img_norm, gamma)
        return np.clip(adjusted * 255, 0, 255).astype(np.uint8)

    def compute_weight_maps(self, images):
        weights = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Contrast
            contrast = cv2.Laplacian(gray_blur, cv2.CV_32F)
            contrast = np.abs(contrast)
            contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)

            # Saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0

            # Improved exposure weighting
            img_float = img.astype(np.float32) / 255.0
            well_exposed = np.exp(-0.5 * ((img_float - 0.6) ** 2) / 0.2)
            well_exposed = np.prod(well_exposed, axis=2)

            # Balanced weight (tuned)
            weight = (contrast + 0.3) * (saturation + 0.3) * (well_exposed + 0.4)
            weight = np.clip(weight, 0.01, 1.0)

            weight = cv2.GaussianBlur(weight, (7, 7), 0)
            weights.append(weight)

        weight_sum = np.maximum(np.sum(weights, axis=0), 1e-3)
        weights = [w / weight_sum for w in weights]

        return weights

    def fuse_images(self, images, weights):
        fused = np.zeros_like(images[0].astype(np.float32))

        for img, weight in zip(images, weights):
            weight_3ch = np.stack([weight] * 3, axis=2)
            fused += img.astype(np.float32) * weight_3ch

        return np.clip(fused, 0, 255).astype(np.uint8)


# ================================
# POST PROCESS (BALANCED)
# ================================
def post_process(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Balanced CLAHE (not too strong)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Mild denoise (avoid blur)
    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced, None, 6, 6, 7, 21
    )

    return enhanced


# ================================
# MAIN FUNCTION
# ================================
def enhance_low_light(frames, flows=None, masks=None, params=None):

    print("Starting low-light enhancement...")

    fusion = MultiExposureFusion()

    # Balanced gamma (important)
    gamma = 0.7

    print(f"Using Gamma: {gamma}")

    # ================================
    # EXPOSURE ADJUSTMENT
    # ================================
    exposures = []

    for frame in frames:
        adjusted = fusion.adjust_exposure(frame, gamma)
        exposures.append(adjusted)

    # ================================
    # WEIGHTS
    # ================================
    weights = fusion.compute_weight_maps(exposures)

    # ================================
    # MOTION MASK (optional)
    # ================================
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

    # ================================
    # FUSION
    # ================================
    fused = fusion.fuse_images(exposures, weights)

    # ================================
    # POST PROCESS
    # ================================
    enhanced = post_process(fused)

    # ================================
    # SEMANTIC PROTECTION
    # ================================
    semantic = SemanticMaskGenerator()
    enhanced = semantic.apply_semantic_protection(frames[0], enhanced)

    # ================================
    # SIMPLE & STABLE FINAL ENHANCEMENT
    # ================================

    # Use first frame as base (more stable)
    base = frames[0].copy()

    # Convert to LAB
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Strong but safe CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Mild brightness + contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=25)

    # Light denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    # SAVE OUTPUTS
    cv2.imwrite('results/fused.png', fused)
    cv2.imwrite('results/enhanced.png', enhanced)

    print("Enhancement complete!")

    return enhanced