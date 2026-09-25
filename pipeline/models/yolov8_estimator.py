import numpy as np

from pipeline.common.keypoints import Keypoint, PersonPose
from pipeline.models.base import PoseEstimator

# YOLOv8-pose keypoint order is already COCO-17, so no remapping is needed.

# ByteTrack over BoT-SORT: fixed tripod camera makes camera-motion compensation
# unnecessary, and ByteTrack keeps low-confidence boxes alive through clinches.
TRACKER_CONFIG = "bytetrack.yaml"


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
        results = self._model.predict(
            frame_bgr, conf=self.conf_threshold, device=self._device(), verbose=False
        )
        return _to_people(results, frame_bgr.shape[:2])

    def track(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        """Like predict(), but each PersonPose also gets a track_id that stays
        the same for that person across consecutive track() calls."""
        results = self._model.track(
            frame_bgr,
            conf=self.conf_threshold,
            device=self._device(),
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )
        return _to_people(results, frame_bgr.shape[:2])

    def reset_tracking(self) -> None:
        """Forget all tracks, so the next track() call starts fresh IDs (use
        before processing a new video). Reloading the model is the simplest
        reliable way to drop ultralytics' persisted tracker state."""
        self.load()

    def _device(self):
        return None if self.device == "auto" else self.device


def _to_people(results, frame_hw: tuple[int, int]) -> list[PersonPose]:
    """Convert ultralytics results to PersonPose objects with x/y and bbox
    normalized to [0, 1]. track_id is filled in only when tracking produced one."""
    h, w = frame_hw
    people = []
    for result in results:
        if result.keypoints is None:
            continue
        xy = result.keypoints.xy.cpu().numpy()      # (n_people, 17, 2), pixel coords
        conf = result.keypoints.conf
        conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
        ids = result.boxes.id if result.boxes is not None else None
        ids = ids.int().cpu().tolist() if ids is not None else None

        for i, (person_xy, person_conf) in enumerate(zip(xy, conf)):
            keypoints = [
                Keypoint(x=float(x) / w, y=float(y) / h, confidence=float(c))
                for (x, y), c in zip(person_xy, person_conf)
            ]
            bbox = None
            if boxes is not None:
                x1, y1, x2, y2 = boxes[i]
                bbox = (float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h)
            track_id = ids[i] if ids is not None else None
            people.append(PersonPose(keypoints=keypoints, bbox=bbox, track_id=track_id))
    return people
