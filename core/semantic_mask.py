import cv2
import numpy as np


class SemanticMaskGenerator:
    def __init__(self):
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def generate_face_mask(self, image):
        """
        Detect faces and create mask
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
            # Create soft mask (smooth edges)
            face_region = np.ones((h, w), dtype=np.float32)

            # Place in full mask
            mask[y:y+h, x:x+w] = face_region

        # Blur mask for smooth transition
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return mask

    def apply_semantic_protection(self, image, enhanced):
        """
        Reduce enhancement strength in face regions
        """
        face_mask = self.generate_face_mask(image)

        if np.max(face_mask) == 0:
            # No face detected
            return enhanced

        # Normalize mask
        face_mask = face_mask / (np.max(face_mask) + 1e-6)

        # Expand to 3 channels
        face_mask_3ch = np.stack([face_mask] * 3, axis=2)

        # Blend:
        # face → more original
        # background → enhanced
        output = enhanced * (1 - face_mask_3ch) + image * face_mask_3ch

        return output.astype(np.uint8)