"""
semantic_mask.py
Semantic face protection mask (FIXED VERSION)
"""

import cv2
import numpy as np


class SemanticMaskGenerator:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def generate_face_mask(self, image):
        """
        Detect faces and create a soft blended mask.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        mask = np.zeros(gray.shape, dtype=np.float32)

        for (x, y, w, h) in faces:
            mask[y:y+h, x:x+w] = 1.0

        # Soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return mask

    def apply_semantic_protection(self, original, enhanced):
        """
        Blend enhanced and original in face regions to prevent over-processing.

        FIX: Convert both inputs to float32 BEFORE multiplying.
        Old code multiplied uint8 arrays directly → values overflowed 255
        mid-calculation → blown-out whites + color corruption.
        """
        face_mask = self.generate_face_mask(original)

        # No faces detected — return enhanced as-is, no blending needed
        if np.max(face_mask) == 0:
            return enhanced

        # Normalize mask to 0–1
        face_mask = face_mask / (np.max(face_mask) + 1e-6)
        face_mask_3ch = np.stack([face_mask] * 3, axis=2)

        # FIX: Cast to float32 BEFORE any multiplication
        # uint8 * float = silent overflow → garbage pixels
        original_f = original.astype(np.float32)
        enhanced_f = enhanced.astype(np.float32)

        # Face regions → blend toward original (protect from over-brightening)
        # Background → keep fully enhanced
        output = enhanced_f * (1.0 - face_mask_3ch) + original_f * face_mask_3ch

        # Safe clip and cast back
        return np.clip(output, 0, 255).astype(np.uint8)