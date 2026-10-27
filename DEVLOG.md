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

## 2026-09-30
  - Added the keypoints JSON: one entry per frame next to the annotated video, so metrics can be built
    without re-running the pose model. -> DECISIONS.md 008.
  - Tried blocking kit switches while boxes overlap (branch clinch-switch-fix). Fixed two wrong switches
    on clinch_clip but blocked a correct one at frame 1180 and added 156 missing frames. Not merged.
  - Takeaway: inside a clinch I can't reliably tell the fighters apart at all, so stop trying.

## 2026-10-06 / 09
  - Clinches: the fighter's identity is no longer assigned while the boxes overlap (IoU >= 0.3). After
    the clinch the fighter is picked out again by kit alone. -> DECISIONS.md 009.
  - Fiddly part was deciding when a clinch is over: one visible person usually means someone is hidden,
    not that they split. Needed 3 frames of two people seen apart, or a 1 s timeout. Also had to keep the
    "nearby" range tight so the coach at ringside wasn't counted as the second fighter.
  - clinch_clip: drawn 61%, held back for clinch 33%, missing otherwise 6%.
  - Added frame weights (0-1 trust score per frame from overlap x body confidence) plus the helpers every
    metric will use: weighted averages, event counts, coverage. -> DECISIONS.md 010.
  - clinch_clip weights: 46% full, 14% partial, 40% zero; 53% usable for counting events.

## 2026-10-13
  - Logging only: the JSON lists each clinch (start, end, ended "apart" or by "timeout"). clinch_clip:
    49 clinches, 42 apart, 7 timeout. Tracking output unchanged.
  - Docs caught up (DECISIONS 008-010, README outputs section).
  - Next: re-run the other 3 clips with clinch handling, get the branches onto main, then start
    Phase 2 with guard height.

## 2026-10-22
  - Phase 2 started (plan: baseline per fighter, punches, guard drops "reported both ways").
  - Step 1: pipeline/explore.py plots guard height (wrist above shoulder line) and arm extension
    (shoulder->wrist) per hand, in torso lengths, with weight-0 frames grayed out. Ran on all 4 clips.
  - Typical values on usable frames: guard median -0.2 to -0.5 (wrist *below* the shoulder line),
    extension median ~0.5, p90 ~0.8.
  - Checked by eye on 104715 (60-75 s): a tight guard with gloves at the chin still reads -0.3 to -0.7,
    because the wrist point sits low inside the glove. So "shoulder line" is not "hands up"; a fixed
    standard needs calibrating, or a different reference (nose is confident in 77-90% of usable frames,
    shoulders 99-100%).
  - Noise: about +-0.1 frame to frame, plus 1-2 frame spikes where the wrist point lands in the wrong
    spot (64.4 s looked like a punch, wasn't). Needs smoothing that ignores single-frame jumps.
  - Camera angle: fighter is side-on only ~25-35% of usable frames (shoulder width vs torso). Punches
    thrown toward the camera barely change 2D arm length, so extension alone will miss many punches.
  - Next: decide guard reference + punch signal, then step 2 (body.py helpers with tests).

## 2026-10-25
  - Labelled the kit: 40 guard frames (per hand up/down/unsure, plus "punch" for a hand mid-punch) and
    every punch in both clips (43). Copied to labels/phase2/ so they're in git.
  - pipeline/labels.py picks the fixed guard threshold from the labels: -0.34 torso lengths, balanced
    accuracy 0.83 (104715 0.89, clinch_clip 0.74). Fitted on either clip alone it lands at -0.33/-0.35,
    so it holds across clips. -> DECISIONS.md 011.
  - Checked the wrong frames by eye: labels right (one changed to unsure, hand hidden). G010/G021 look
    like the model swapping the wrists; G016/G022 unexplained. Rest are borderline.
  - Next: compare the punch signals against the punch labels (DECISIONS 012).

## 2026-10-27
  - Guard misses looked into: no left/right wrist swap behind G010 or G021 (real swaps: 0.1-0.8% of
    picture-to-picture steps). G010: the right wrist point jumped onto the raised glove for one picture
    that is shown twice (repeated frame), so the 3-frame median keeps it. G016: fighter_2's JSON
    follows fighter_1 there (10 frames in 104715). G022: back view and blur, left wrist put at the
    waist; torso and lean are normal.
  - Punch signals scored on the 68 labelled punches vs 186 no-punch windows (all windows 1.6 s):
    wrist speed best (AUC 0.83), arm extension 0.74, elbow opening 0.68. Side view easiest; extension
    fails on back views. Wrist-speed threshold doesn't carry over between clips yet (clinch_clip has
    only 12 no-punch windows).
