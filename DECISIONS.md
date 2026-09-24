# Decisions

Format: what we chose, what else we considered, why.

## 001 — Build the CV pipeline as a CLI before the web app
Date: 2026-09-24
Alternatives: build frontend and backend first.
Why: pose estimation and tracking are the riskiest parts. If they
don't work on real sparring footage, the rest doesn't matter.
A CLI lets me test them fast.

## 002 — Process uploaded video offline, show insights synced to playback
Date: 2026-09-24
Alternatives: true real-time analysis during playback.
Why: footage is pre-recorded, so real-time adds complexity with no
user benefit. Timeline-synced feedback gives the same experience.

## 003 — Insights come from measured metrics, LLM only explains them
Date: 2026-09-24
Alternatives: send video frames to an LLM and ask for feedback.
Why: metrics are verifiable and tied to timestamps. An LLM watching
video directly could invent observations I can't check.

## 004 — Keep videos out of Git
Date: 2026-09-24
Alternatives: commit sample clips or use Git LFS.
Why: files are large and some test footage isn't mine to publish.
Local videos live in data/ (gitignored); the web app will use cloud storage.

## 005 — Pose model
Status: resolved. Chose YOLOv8-pose (yolov8s-pose, via ultralytics).
Alternatives: MediaPipe Pose (BlazePose), RTMPose (via rtmlib, since the
official mmpose/mmcv/mmdet stack is fragile to install on Windows).
Why: built a comparison script (pipeline/run_pose_estimation.py, since
simplified to only run YOLOv8) that ran all three on real sparring footage,
including a clinch-heavy clip. MediaPipe's body tracking was visually poor
and ruled out immediately. RTMPose gave good visual quality but only ran at
~0.18 fps on CPU (~28 minutes for a 10s clip) — GPU acceleration would need
a separately installed CUDA Toolkit + cuDNN, since onnxruntime-gpu (unlike
torch) doesn't ship those bundled. YOLOv8-pose held up well on wrist/hand
tracking through fast punches, kept both fighters as separate people through
a clinch (with brief, quickly-recovered dropouts during full occlusion —
expected, and a job for person tracking, not the pose model), and gave
honest low confidence on occluded joints rather than confidently-wrong
positions. It's also the easiest of the three to run on GPU here, since
torch wheels are self-contained.

Note: this machine's GPU (RTX 5060, Blackwell/sm_120) needed the `cu128`
PyTorch build — the `cu124` build installs and reports
`torch.cuda.is_available() == True`, but actually fails at runtime with
"CUDA error: no kernel image is available for execution on the device"
since it only ships kernels up to sm_90. Worth remembering for any future
CUDA-related install on this machine.