"""Common interface each pose-estimation backend implements, so the
comparison script can run all three the same way. Every estimator returns
keypoints normalized to [0, 1] of frame width/height, so pipeline.common.drawing
can render them identically regardless of backend.
"""
from abc import ABC, abstractmethod

import numpy as np

from pipeline.common.keypoints import PersonPose


class PoseEstimator(ABC):
    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before any predict() calls."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        """Run pose estimation on a single BGR frame.

        Returns one PersonPose per detected person, with 17 keypoints each
        in COCO-17 order, x/y normalized to [0, 1] of frame width/height.
        """
