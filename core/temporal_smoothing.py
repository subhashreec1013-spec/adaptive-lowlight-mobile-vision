import numpy as np


class TemporalSmoother:
    def __init__(self, alpha=0.7):
        """
        alpha → smoothing strength
        higher = smoother output
        """
        self.alpha = alpha
        self.prev_frame = None

    def smooth(self, current_frame):
        """
        Apply temporal smoothing using EMA
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return current_frame

        smoothed = (
            self.alpha * self.prev_frame +
            (1 - self.alpha) * current_frame
        )

        smoothed = smoothed.astype(np.uint8)

        self.prev_frame = smoothed

        return smoothed