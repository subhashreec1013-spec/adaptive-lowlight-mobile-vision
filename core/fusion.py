"""
fusion.py
Adaptive Multi-frame fusion with motion-aware weighting (FINAL WORKING VERSION)
"""

import cv2
import numpy as np
from core.semantic_mask import SemanticMaskGenerator


class MultiExposureFusion:
    def __init__(self):
        self.exposure_levels = [0.9, 1.0, 1.1]

    def adjust_exposure(self, image, gamma):
        img_norm = image.astype(np.float32) / 255.0
        adjusted = np.power(img_norm, 1 / gamma)
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

            # Well-exposedness
            img_float = img.astype(np.float32) / 255.0
            well_exposed = np.exp(-0.5 * ((img_float - 0.5) ** 2) / 0.08)
            well_exposed = np.prod(well_exposed, axis=2)

            # Balanced weight
            weight = (contrast + 0.3) * (saturation + 0.3) * (well_exposed + 0.3)
            weight = np.clip(weight, 0.01, 1.0)

            weight = cv2.GaussianBlur(weight, (7, 7), 0)
            weights.append(weight)

        # Stable normalization
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
# POST PROCESS
# ================================
def post_process(image, clahe_clip=2.0, denoise=8):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=min(1.5, clahe_clip),
        tileGridSize=(8, 8)
    )

    l_enhanced = clahe.apply(l)

    lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    denoise = min(10, denoise)

    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced, None, denoise, denoise, 7, 21
    )

    return enhanced


# ================================
# MAIN FUNCTION
# ================================
def enhance_low_light(frames, flows=None, masks=None, params=None):

    print("Starting low-light enhancement...")

    fusion = MultiExposureFusion()

    # PARAMETERS
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
    # USE ALL FRAMES (REAL MULTI-FRAME)
    # ================================
    exposures = []

    for frame in frames:
        adjusted = fusion.adjust_exposure(frame, gamma)
        exposures.append(adjusted)

    # WEIGHTS
    weights = fusion.compute_weight_maps(exposures)

    # ================================
    # MOTION MASK
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
    enhanced = post_process(fused, clahe_clip, denoise)

    # ================================
    # SEMANTIC PROTECTION
    # ================================
    semantic = SemanticMaskGenerator()
    enhanced = semantic.apply_semantic_protection(frames[0], enhanced)

    # SAVE (optional)
    cv2.imwrite('results/fused.png', fused)
    cv2.imwrite('results/enhanced.png', enhanced)

    print("Enhancement complete!")

    return enhanced