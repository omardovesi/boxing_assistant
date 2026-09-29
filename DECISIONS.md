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

## 006 — Person tracker: BoT-SORT
Date: 2026-09-25
Status: resolved (kept; identity is now checked on top of it, see 007).
Alternatives: ByteTrack (first choice, assumed a tripod camera); re-linking
lost IDs in our own code (still possible on top); limiting testing to
tripod footage.
Why: most of my test clips are handheld. ByteTrack predicts box motion
assuming a still camera, so pans broke matching and the fighter got new IDs.
BoT-SORT (built into ultralytics, default botsort.yaml, ReID off) adds
camera-motion compensation (sparse optical flow). Same 10s segments, share
of frames where the chosen fighter's ID was missing:

| Clip | ByteTrack | BoT-SORT |
|---|---|---|
| Recording 104715 | 9% | 6% |
| Recording 122938 | 3% | 12% |
| VIDEO-…-Trim | 44% | 8% |
| clinch_clip | 64% | 0% (checked by eye: no swap to opponent) |

Speed cost: roughly 10–35% slower (e.g. 23 → 15 fps on 122938, GPU).
Known failure: in 122938 the fighter walks fully behind the opponent
during a pan (frame ~266) and comes back ~15 frames later with a new ID
that is never re-linked. Candidate fix: our own re-linking (option A).

Lesson learned later (2026-09-29): "ID missing %" can't see swaps. On the
full clinch_clip and Trim videos the chosen ID was often *present but on the
opponent*: BoT-SORT swaps the two fighters' IDs in clinches. The 10 s
"0% missing" above hid that. Always check by eye against the person picked
in the preview, not just that the skeleton stays on one person.

## 007 — Identify the fighter by their kit (headgear / shirt / trunks)
Date: 2026-09-29
Status: resolved.
Problem: the tracker loses the fighter (new ID after occlusion, camera pans,
scene cuts) and swaps IDs with the opponent in clinches, so following one
track ID puts the skeleton on the wrong person.
Chosen: `FighterFollower` (pipeline/tracking.py) + kit comparison
(pipeline/appearance.py):
- Kit regions from pose keypoints: headgear (box above/around the head, not
  the face), shirt (shoulders→hips), trunks (hips→knees). Each is a colour
  histogram: hue-saturation bins for coloured pixels + 4 brightness bins for
  colourless ones, so white vs black clothing counts.
- Calibration: first 30 frames where the fighter and the opponent (nearest
  person of similar size, or `--opponent ID`) don't overlap. Per region,
  separation = fighter-vs-opponent distance − fighter's own spread; regions
  under 0.10 are dropped, the rest weighted by separation. The CLI prints it.
- Then every frame, a person counts as the fighter when their kit is closer to
  the fighter's than the opponent's by ≥ 0.10, within 0.45 of the fighter's,
  and ≥ 0.7× the opponent's height (keeps bystanders out). Fighter ID gone →
  relink to that person. Someone else counts as the fighter and the followed
  person doesn't for 5 frames in a row → swap. Followed person looks like the
  opponent for 5 frames → no skeleton rather than a wrong one.
- Both kit prototypes adapt slowly (α 0.05), only from confident,
  non-overlapping frames (lighting changes, scene cuts).
- Before calibration (or without frames): old rule, relink only to a *new* ID
  nearby of similar size, give up after 90 frames.

Results, full videos (fighter missing; before → after):

| Clip | Deciding kit | Missing | Swaps corrected |
|---|---|---|---|
| Recording 104715 | headgear 46%, trunks 37%, shirt 18% | 10% → 5% | 1 |
| clinch_clip | headgear 49%, shirt 31%, trunks 20% | 34%* → 8% | 5 |
| Recording 122938 | headgear 47%, trunks 30%, shirt 24% | 27% → 20% | 0 |
| VIDEO-…-Trim | trunks 39%, shirt 38%, headgear 23% | 48%* → 26% | 0 |

\* before, much of the "found" time was actually on the opponent.
Checked by eye (12 frames per video, against the preview pick): all on the
right fighter; remaining "missing" frames were fighter out of view / camera on
the floor. Reviewed by me on the full videos: only brief jumps to the opponent
inside clinches, recovered right after.

Alternatives tried and dropped:
1. Relink by position/size only: jumped to a bystander (104715, frame 281).
2. One colour fingerprint + fixed threshold 0.40: fingerprint ignored
   white/black and the "head" crop was mostly face skin, so it barely
   separated the fighters; permanent rejection + giving up after 90 frames
   lost the fighter for most of long videos.
3. Frozen "anchor" fingerprint + per-frame swap check: made clinch_clip worse
   (34% → 89% missing) and its one Trim "swap" went *onto* the opponent.
4. Not tried yet: BoT-SORT's built-in ReID (learned appearance features).
   Worth comparing if kit identification fails, e.g. same-coloured kit.
Known limits: both fighters in the same kit; the fighter never apart from the
opponent early on (no calibration → ID-only); brief wrong frames in clinches.
Also: frames are now streamed (pipeline/common/video_io.py VideoStream), since
full 1440p clips decoded at once need 40–53 GB RAM; `--duration` defaults to
the whole video. Skeleton is drawn in one colour (cyan) whatever the track ID.
