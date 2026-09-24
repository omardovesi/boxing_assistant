## 2026-09-24
  - Built pipeline/ from scratch: video I/O, drawing, PoseEstimator interface.
  - Compared MediaPipe/RTMPose/YOLOv8-pose on real clips (short clip + clinch clip).
  - MediaPipe: poor body capture, ruled out.
  - RTMPose: good quality, 0.18 fps CPU (~28min/10s clip) — needs onnxruntime-gpu + CUDA Toolkit to be practical.
  - YOLOv8-pose: 14fps CPU, good quality, held up on clinch (brief drops, quick recovery). Picked as pose model -> DECISIONS.md 005.
  - Gotcha: RTX 5060 is sm_120 (Blackwell) - cu124 torch installs fine but fails at runtime; needed cu128.
  - Cleaned up: removed mediapipe/rtmpose code+deps, renamed compare_pose_models.py -> run_pose_estimation.py.
  - Next: fighter selection UI/logic + cross-frame ID tracking for pipeline.analyze.

  