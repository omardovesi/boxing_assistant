import numpy as np

from pipeline.common.keypoints import Keypoint, PersonPose
from pipeline.models.base import PoseEstimator

# YOLOv8-pose keypoint order is already COCO-17, so no remapping is needed.


class YOLOv8Estimator(PoseEstimator):
    name = "yolov8"

    def __init__(self, variant: str = "s", conf_threshold: float = 0.3, device: str = "auto"):
        self.variant = variant
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(f"yolov8{self.variant}-pose.pt")

    def predict(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        h, w = frame_bgr.shape[:2]
        device = None if self.device == "auto" else self.device
        results = self._model.predict(
            frame_bgr, conf=self.conf_threshold, device=device, verbose=False
        )

        people = []
        for result in results:
            if result.keypoints is None:
                continue
            xy = result.keypoints.xy.cpu().numpy()      # (n_people, 17, 2), pixel coords
            conf = result.keypoints.conf
            conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])

            for person_xy, person_conf in zip(xy, conf):
                keypoints = [
                    Keypoint(x=float(x) / w, y=float(y) / h, confidence=float(c))
                    for (x, y), c in zip(person_xy, person_conf)
                ]
                people.append(PersonPose(keypoints=keypoints))
        return people
