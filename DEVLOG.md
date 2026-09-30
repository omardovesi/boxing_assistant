## 2026-09-24
  - Built pipeline/ from scratch: video I/O, drawing, PoseEstimator interface.
  - Compared MediaPipe/RTMPose/YOLOv8-pose on real clips (short clip + clinch clip).
  - MediaPipe: poor body capture, ruled out.
  - RTMPose: good quality, 0.18 fps CPU (~28min/10s clip) — needs onnxruntime-gpu + CUDA Toolkit to be practical.
  - YOLOv8-pose: 14fps CPU, good quality, held up on clinch (brief drops, quick recovery). Picked as pose model -> DECISIONS.md 005.
  - Gotcha: RTX 5060 is sm_120 (Blackwell) - cu124 torch installs fine but fails at runtime; needed cu128.
  - Cleaned up: removed mediapipe/rtmpose code+deps, renamed compare_pose_models.py -> run_pose_estimation.py.
  - Next: fighter selection UI/logic + cross-frame ID tracking for pipeline.analyze.

  
## 2026-09-25
  - Added ByteTrack person tracking + pipeline.analyze CLI (--preview to pick an ID, --fighter to draw one skeleton).
  - Problem: handheld clips lost the fighter's ID (up to 64% of frames missing on clinch_clip).
  - Switched to BoT-SORT (camera-motion compensation): clinch_clip 0% missing, Trim 44% -> 8%. -> DECISIONS.md 006.
  - analyze now prints distinct track IDs seen (inflated by bystanders; missing % is the better signal).
  - Still broken: full occlusion + pan in Recording 122938 -> new ID, never re-linked.
  - Next: decide on ID re-linking (option A), then per-frame keypoints JSON.

## 2026-09-28 / 29
  - Added re-linking (FighterFollower), then a colour check; both fixed the 10 s clips but failed on full videos.
  - Full videos needed streaming frames (40–53 GB RAM otherwise): VideoStream, --duration defaults to whole video.
  - Found the real problem by comparing against the preview picks: on clinch_clip and Trim the skeleton was
    following the *opponent* for long stretches (BoT-SORT swaps IDs in clinches). Missing % hid this.
  - Anchor-fingerprint attempt made it worse (clinch 89% missing); reverted.
  - My idea: compare the fighters' kit at the start and use whatever differs most (often headgear).
    Kit identification (headgear/shirt/trunks, fighter-vs-opponent scoring) -> DECISIONS.md 007.
  - Results: 104715 5%, clinch 8%, 122938 20%, Trim 26% missing; skeleton stays on the right fighter,
    only brief jumps inside clinches. Outputs in outputs/analyze/.
  - Next: commit, then per-frame keypoints JSON with a clinch flag.
