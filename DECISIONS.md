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

Re-run on 2026-10-20 (current main: kit identification plus the clinch
handling from 009), same clips and fighters:

| Clip | Skeleton drawn | Clinch (held back) | Missing otherwise |
|---|---|---|---|
| Recording 104715 | 88% | 10% | 2% |
| clinch_clip | 61% | 33% | 6% |
| Recording 122938 | 71% | 10% | 19% |
| VIDEO-…-Trim | 64% | 11% | 25% |

"Missing otherwise" is the number to compare with the table above. It went
down a little everywhere because clinch frames are now counted separately.
The brief jumps to the opponent inside clinches are gone, since nobody is
drawn during a clinch (009). Checked by eye: 9 frames per clip spread over
the whole video, all on the right fighter. What's still missing is the same
as before: the fighter behind the opponent in 122938, and the phone camera
pointing at the floor in Trim.

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
Note (2026-10-28): both fighters can end up on the same person. In 104715
(frames 3835–3850), fighter_2 is hidden behind fighter_1, and fighter_2.json
switches to fighter_1's track (1870) in 3835–3839 and 3846–3850, with
fighter_2 missing in between. Both files then hold the same skeleton, and
the overlap is 0 (only one person is visible), so the frames get full
weight. Found through guard frame G016 (DECISIONS 011). Only matters when
two fighters are tracked in one clip (the labelling kit, not the app, which
analyses one fighter). Evaluation code leaves out frames where both files
follow the same track.
Also: frames are now streamed (pipeline/common/video_io.py VideoStream), since
full 1440p clips decoded at once need 40–53 GB RAM; `--duration` defaults to
the whole video. Skeleton is drawn in one colour (cyan) whatever the track ID.

## 008 — Save the fighter's keypoints to a JSON file
Date: 2026-09-30
Status: resolved.
What: every run of `--fighter N` now writes `fighter_N.json` next to
`fighter_N.mp4`. It has one entry per video frame, so frame numbers and
times always match the video. Each entry says whether the fighter was found
and, if so, their 17 body points (x, y, and how confident the model is),
their box, the track ID being followed, and how much their box overlaps the
opponent's.
Alternatives: only save the annotated video; save pixel coordinates.
Why: Phase 2 metrics need the numbers, and re-running the pose model every
time I try a new metric is slow. Coordinates stay as fractions of the frame
(0–1, as the model gives them) so the file doesn't depend on the video's
resolution; width and height are saved at the top to turn them into pixels.
Frames where the fighter is missing are kept (marked `"found": false`)
rather than dropped, so nothing gets out of step with the video.

## 009 — In a clinch, stop guessing who the fighter is
Date: 2026-10-09
Status: resolved.
Problem: in a clinch the two fighters are tangled together. Two things I
relied on to tell them apart both break down at once:
- the tracker often swaps their IDs, and
- the kit check (007) gets confused, because the crop around each fighter
  also contains a big chunk of the other fighter's kit.
So the skeleton kept jumping onto the opponent mid-clinch.
Chosen: during a clinch, don't draw anyone. The fighter is marked missing
with `"clinch": true` in the JSON, and when the clinch is over they are
picked out again by kit alone, ignoring which ID they had before.
- A clinch starts when the fighter's box overlaps the nearest person of
  similar size by 0.3 or more. (Overlap here is IoU: the shared area divided
  by the total area of both boxes. 0 = apart, 1 = the same box.)
- It ends after 3 frames where two people are seen close to the fighter's
  spot and not overlapping. If only one person is visible there, that
  usually means one fighter is hidden behind the other, so it doesn't count
  as "apart".
- Safety net: if only one person is visible for 1 second (30 frames), the
  clinch ends anyway, so a lost partner can't freeze it forever.
- "Close" means within half the fighter's height, tight enough that a coach
  standing at ringside doesn't count as the second fighter.
Alternative tried first (branch `clinch-switch-fix`, not merged): keep
following, but don't let the kit check switch people while the boxes overlap
by 0.1 or more. On clinch_clip it removed two wrong switches (frames 2554
and 2604) but also blocked a correct one at frame 1180, which left the
skeleton on the opponent for about 60 frames, and it added 156 missing
frames. Overlap alone couldn't tell the good switches from the bad ones.
Skipping clinches entirely is simpler and more honest.
Result on clinch_clip (3908 frames, run of 2026-10-09): the fighter is drawn
in 61% of frames, held back for a clinch in 33%, and missing for other
reasons (out of view etc.) in 6%. 49 clinches: 42 ended with the fighters
seen apart, 7 by the 1-second timeout.
Trade-off: no skeleton during clinches, so nothing that happens in a clinch
(short punches, holding) can be measured. Fine for now: the planned metrics
are about open exchanges. The other three clips haven't been re-run since
this change.
Logging (2026-10-13): the JSON now also lists every clinch with its start,
end, how it ended (`"apart"` or `"timeout"`) and how many frames passed
after the last real overlap. That changes nothing about the tracking; it's
there so I can check whether the exit rules end clinches too early or too
late.
Note (2026-10-20): PR #4 (clinch-switch-fix, commit 543e715) was merged into
main by mistake on GitHub; this merge undoes it, keeping docs-update's
tracking.py and tests. If ID swaps come back in near-clinches, 543e715 is
where the rule and its four tests live.

## 010 — Give each frame a trust score ("frame weight") for metrics
Date: 2026-10-09
Status: resolved (numbers may be tuned once real metrics exist).
Problem: not every frame is equally reliable. Some are borderline clinches,
some have a blurry or half-hidden body. Treating them all the same would let
bad frames drag metrics around, while throwing them away completely is a
hard cut that hides how much evidence a number really has.
Chosen: every frame gets a weight from 0 (ignore it) to 1 (fully trust it):

    frame_weight = overlap factor × confidence factor

- Overlap factor: 1 when the boxes overlap by 0.10 or less, sliding down to 0
  at 0.30 (where a clinch starts anyway).
- Confidence factor: based on the average confidence of the 12 body points
  (shoulders down to ankles; face points are left out because headgear makes
  them unreliable). 0 at 0.30 or below, sliding up to 1 at 0.55 or above.
  0.55 is roughly what the worst 10% of clean frames score on clinch_clip.
- No skeleton → weight 0.
How metrics use it (pipeline/metrics.py):
- Averages (e.g. guard height) are weighted, so shaky frames count less.
- Counts (e.g. punches) only use frames with weight 0.5 or more, because a
  punch either happened or it didn't, and half-counting one makes no sense.
- Every result also reports "coverage": the total weight divided by the
  number of frames. A coverage of 0.2 means "this number rests on thin
  evidence", and the app can say so instead of hiding it.
All the limits live in pipeline/config.py. The JSON stores the raw overlap
and confidences, so changing a limit works on old files without re-running
the video. On clinch_clip: 46% of frames have full weight, 14% are in
between, 40% are zero (mostly clinches), and 53% are good enough for counts.
`python -m pipeline.metrics <json> --frame N` shows the breakdown for any
frame.
Alternatives: a hard yes/no filter (simpler, but throws away information
and makes metrics jump when a frame sits right at the cut-off); weighting by
confidence only (misses borderline clinches where the model is confidently
wrong).

## 011 — Guard height: measured from the shoulders, judged two ways
Date: 2026-10-22
Status: accepted. Fixed threshold picked 2026-10-25 (see the end).
What: guard height = how far the wrist is above the shoulder line, in torso
lengths (mid-shoulders to mid-hips), per hand. Torso lengths make it
independent of how far the fighter is from the camera.
Why the shoulders and not the nose/chin: shoulders are found in 99–100% of
usable frames, the nose only in 77–90%. Up/down also barely changes with
which way the fighter faces the camera.
What the data showed first (pipeline/explore.py, all four clips): the wrist
point sits low inside the glove, so a tight guard with the gloves at the chin
still reads 0.3–0.7 torso lengths *below* the shoulder line. "Above the
shoulder line" therefore does not mean "hands up".
Judged two ways (my call, see the Phase 2 plan):
1. Against the fighter's own normal stance (median per hand over the clip):
   "guard drop" events fire when a hand goes clearly below *their* normal.
   A deliberate low guard (e.g. Philly shell) doesn't fire all round.
2. Against one fixed standard for everyone: share of time each hand is below
   it, as a single "exposure" number. The value isn't guessed: I'll label
   30–40 frames as guard up / guard down across at least 2 clips, including
   borderline ones, and the threshold is picked from those.
Noise: the wrist point jitters about ±0.1 frame to frame and sometimes jumps
for a single frame. Example, 104715 at 64.43 s (frame 1933): the right wrist
lands below its own elbow for one frame, confidence still 0.88. It is *not* a
left/right swap (the left wrist didn't move). Fix: a 3-frame median filter.
Its length is set in seconds in pipeline/config.py (0.1 s = 3 frames at
30 fps) so it means the same at other frame rates, with a test that a short
jab survives it.
Also found: 12–33% of frames in my clips are repeats of the previous frame
(lower real frame rate than the 30 fps the file says). Anything measuring
speed uses a short time span, not frame-to-frame differences.
The fixed threshold (2026-10-25): **-0.34 torso lengths** (below = down),
`GUARD_EXPOSURE_THRESHOLD` in pipeline/config.py.
- Labels: 40 frames from the labelling kit (20 from 104715 with both fighters,
  20 from clinch_clip with fighter_1), judged per hand without seeing the
  values. Of the 80 hands, 63 are usable (42 down, 21 up); 15 are left out
  as `unsure` or `punch` (a hand mid-punch says nothing about the guard) and
  2 had no wrist point. Kept in labels/phase2/.
- Picked by `python -m pipeline.labels guard`, on the smoothed value the
  metric sees: the midpoint between labelled values with the best *balanced
  accuracy* (average of the share of downs caught and the share of ups
  caught), so the 2:1 majority of downs doesn't pull it. On a tie, the
  midpoint in the widest gap between values wins (my choice).
- Result: downs 34/42, ups 18/21, balanced accuracy 0.83. 104715: 0.89,
  clinch_clip: 0.74. Left hand 0.77, right hand 0.88.
- Cross-clip check: fitted on 104715 alone it comes out at -0.345 and scores
  0.74 on clinch_clip; fitted on clinch_clip alone it's -0.332 and scores
  0.82 on 104715. The threshold barely moves, so it looks like a real
  standard rather than a fit to one clip. clinch_clip is harder: many of its
  hands sit between -0.35 and -0.25, where up and down overlap.
- G034 right was relabelled from `up` to `unsure` after review: the hand is
  hidden, so it's genuinely ambiguous. The threshold didn't change (-0.344
  before and after); balanced accuracy went from 0.81 to 0.83.
- Mistakes checked by eye (all other labels confirmed correct):
  - G010 and G021: not a left/right wrist swap. Checked 2026-10-27: real
    swaps (trading the two wrist points fits the previous picture much
    better than keeping them) happen in only 0.1-0.8% of picture-to-picture
    steps, and none within 10 frames of either. G010: the right wrist point
    jumps onto the raised glove for one picture (-0.57 -> +0.02 and back),
    and that picture is repeated (frames 2397-2398), so the 3-frame median
    filter keeps it. G021: inconclusive; back view, right after 3 frames
    without the fighter, right wrist only 45% confident.
  - G016: tracking error, not a measurement one. fighter_2.json follows
    fighter_1 there (see DECISIONS 007, note of 2026-10-28).
  - G022: torso 18% shorter than the clip's typical one and lean normal
    (8°), so neither explains it: rescaled it still reads about -0.88. It's a
    blurred back view and the model put the left wrist at the waist with 82%
    confidence.
  - The rest are within about 0.25 of the threshold: borderline frames.
- Known limit: 63 hands from 3 fighters in 2 clips. Phase 3 re-checks it on
  more labels.
Update (2026-10-28): guard height is now median-filtered over distinct
pictures, not frames, so a wrong picture shown twice (G010) is removed too.
Repeated pictures come from the video's pixels (the labelling kit's cache);
the keypoints of a repeat aren't exactly equal (compression noise up to
0.001), so they can't be used to find them. The threshold moved from -0.344
to -0.350 (now **-0.35**), balanced accuracy 0.83 -> 0.82; cross-clip fits
-0.350 / -0.343 (closer than before), the clinch_clip fit scores 0.87 on
104715 (was 0.82). G010 right: +0.02 -> -0.29. Punch signals keep the
per-frame filter: over pictures it flattens a peak lasting one picture and
lost 2 of 14 front-view punches.
To do at the next re-run: export.py writes a per-frame `repeat` flag (with
the opponent's box, 012), so the app doesn't need the labelling kit's cache.
Guard-drop events (2026-10-28, pipeline/guard.py), the "against their own
normal" half of this decision:
- A drop = a hand at least 0.3 torso lengths below that fighter's usual
  guard height for that hand (median over trusted frames) for 0.3 s or
  more. 0.3 is about 3x the wrist jitter (my choice, not fitted).
- Not during that hand's punches or the 0.5 s after each (the hand is still
  coming back). Only the same hand's punches count: a rear hand dropping
  while the lead hand jabs is exactly what a coach wants to see.
- Only trusted frames (weight >= 0.5); a missing or untrusted frame ends a
  drop. Guard height filtered over distinct pictures, as for the threshold.
- Usual guard is the median over all trusted frames; punches aren't taken
  out first (they're a few percent of frames, the median barely moves).
- Each event: hand, start, end, duration, usual, lowest, below_usual.
Result (`python -m pipeline.labels drops`): 30 drops in the two clips,
3.4-7.1 per usable minute per fighter, median 0.4-0.6 s long. Usual guard
ranges from -0.30 to -0.63, so fighters differ a lot (104715 fighter_1
carries the left hand low). None of the 22 hands labelled `up` falls in a
drop; 2 of 43 labelled `down` do, which is expected, as most `down` labels
are at that fighter's usual height (the fixed standard catches those).
By-eye check: 10 random drops in outputs/check/guard_drops_check.csv, next
to the 20 punch detections.

## 012 — Punch detection: several signals, record the camera angle
Date: 2026-10-22
Status: accepted; thresholds come later from the data.
Problem: the fighter is side-on to the camera in only ~25–35% of usable
frames. A punch thrown towards the camera hardly changes the arm's length on
screen, so arm extension alone would miss many punches.
Chosen signals, combined and compared on real punches in step 5:
- arm extension beyond that hand's own resting extension (handles long
  guards: an arm held out isn't a punch, a jab from it still is)
- wrist speed relative to the shoulder (so stepping in isn't a punch)
- how much the elbow straightens
- wrist velocity towards the opponent. Needs the opponent's position in
  every frame, which the JSON doesn't store yet (only the overlap and the
  opponent's ID): export.py will also write the opponent's box, and the
  clips get re-run.
Every punch event also records a facing estimate, shoulder width / torso
length (roughly: under 0.35 side-on, 0.35–0.6 angled, 0.6+ facing the
camera), so accuracy can be reported per view in Phase 3.
Not in Phase 2: special handling for punches thrown straight at the camera.
They'll be undercounted; Phase 3 measures by how much, per view.
Results (2026-10-27/28, `python -m pipeline.labels punch` and `events`):
- Labels: every punch in both clips, 75 rows; 68 used (7 by clinch_clip's
  red-headgear fighter, who has no keypoints JSON). 60 of the 75 times are
  whole seconds, so every window (punch or no punch) is 1.6 s long.
  No-punch windows keep 1 s from any punch by that fighter.
- Per window, best single signal: wrist speed, AUC 0.83 (extension 0.74,
  elbow opening 0.68). Side views easiest; extension catches 4/13 back-view
  punches.
- Tried and dropped: wrist speed divided by the fighter's own usual speed
  (median, or 90th percentile, per hand over trusted frames), to make one
  threshold fit every fighter. It did worse (AUC 0.70 vs 0.83, catching
  36/67 instead of 57/67) and the cross-clip gap didn't close (7.7 vs 2.9).
  Why: "usual speed" mostly measures how busy a fighter is, not how fast
  their punches are. The clinch_clip fighter moves a lot (median 1.56 torso
  lengths/s vs about 0.9-1.0 in 104715), so dividing by it shrinks their
  punches below everyone else's.
- Speed + extension (both above their thresholds): same punches caught,
  48 -> 41 false windows, no back/front punches lost. Kept.
- As a detector (pipeline/punches.py, speed > 5.2 torso lengths/s and
  extension > 0.11 within 0.3 s, stretches < 0.2 s apart merged): finds
  49/68 labelled punches but fires 136 more times (25 per usable minute per
  fighter). Higher thresholds trade recall for false alarms (8.0 / 0.11:
  25/68 found, 5 per minute); no setting is good. What the extra detections
  are (unlabelled punches, feints, parries, guard resets) is the by-eye
  check's job: outputs/check/punch_events_check.csv, 20 random detections.
- Cross-clip: thresholds don't carry over well (speed fitted on 104715:
  7.9, on clinch_clip: 4.3). clinch_clip has only 12 no-punch windows.
- Back vs front view: nose + eye confidence separates them (median 0.22 on
  back-view punches, 0.82 front, 0.70 side; AUC 0.94 front vs back). Every
  punch event records it as `face_confidence`, next to `facing`.
Design choices (all limits in pipeline/config.py):
- Punch = wrist speed above 5.2 torso lengths/s and, within 0.3 s of that
  fast stretch, arm extension above 0.11 beyond the hand's usual length.
  Both values are the best-balanced-accuracy fits on all 68 punches.
- Wrist speed = distance the wrist moves relative to its shoulder over
  0.1 s; elbow opening = degrees gained over 0.2 s (spans in seconds,
  because of repeated frames).
- Fast stretches less than 0.2 s apart are one punch (out and back).
- Only frames with weight >= 0.5 can be part of a punch (DECISIONS 010).
- Punch signals use the median filter over frames, not pictures (011
  update): over pictures it flattens one-picture peaks.
- Evaluation: every window 1.6 s (whole-second label times); no-punch
  windows keep 1 s from that fighter's punches; a detection matches a label
  when its peak is inside the label's window, each used once.
- By-eye check: a fixed random sample (seed 0) of 20 detections, classified
  as punch / feint / parry / guard / other, since "is it correct?" alone
  wouldn't say what the false alarms are.
- No more tuning in Phase 2. Phase 3 improves accuracy with exact label
  times (2 decimals), the opponent's box (velocity towards the opponent)
  and the per-frame repeat flag.
Re-run with exact label times (2026-10-28): all 76 punches re-timed in mpv
to the frame (mm:ss.mmm); 69 used.
- Short windows: the arm's longest point lands a median 0.02 s from the
  labelled time, so the labels are on time. The wrist-speed peak lands a
  median 0.13 s after (speed looks back 0.1 s, and the return is often fast
  too); 80% within -0.21..+0.45 s. Windows are now 0.25 s before to 0.35 s
  after the label (0.6 s, was 1.6 s), and no-punch windows are 0.6 s too.
  Whole-second times are no longer accepted.
- Combinations: each punch has its own window and a detection matches at
  most one label. Of 13 same-hand pairs under 1 s apart, 5 are both found
  separately; no detection was counted for two punches.
- Per window: wrist speed AUC 0.85 (was 0.83), extension 0.73, elbow
  opening 0.71. Speed + extension fitted jointly: speed > 4.53, extension
  > -0.03; extension only removes 8 of 118 false windows now.
- Cross-clip: speed fitted on 104715 5.66, on clinch_clip 4.39 (gap was
  7.9 vs 4.3). clinch_clip now has 71 no-punch windows (was 12).
- Decision: the detector keeps 5.2 / 0.11. The re-picked 4.53 / -0.03 finds
  51/69 punches but fires 222 times where no punch is labelled (41 per
  usable minute); 5.2 / 0.11 finds 42/69 with 143 (26 per minute). Picking
  per window by balanced accuracy over-rewards the rare punch windows; a
  detector should be tuned on events (Phase 3). 5.2 / 0.11 is also what the
  by-eye sample was drawn from.

## 013 — Metrics go in their own file, with coverage on every number
Date: 2026-10-28
Status: accepted.
What: `python -m pipeline.summary fighter_N.json` writes
`fighter_N_metrics.json` next to the keypoints file: a summary (guard per
hand: average height, usual height, exposure below -0.35, drop count, height
per minute; punch counts per hand and per usable minute; clinch count;
the thresholds used) and every punch and guard-drop event sorted by time.
Why a separate file: the keypoints JSON stays the raw record of the video.
Metrics are recomputed in seconds from it whenever a threshold changes,
without re-running the pose model, and the `settings` block says which
thresholds produced a file.
Choices (mine):
- Each event gets a `time` to sort on: the peak for a punch, the start for a
  guard drop.
- Averages and exposure are weighted by frame weight and carry coverage
  (DECISIONS 010); counts use trusted frames only (weight >= 0.5).
- Height per minute is per clock minute of the video, so fatigue shows as
  a trend; minutes with little trusted footage show low coverage.
- Repeated pictures: the JSON's own per-frame `repeat` flags if present
  (planned for the next re-run), else `--repeats`, else filter over frames;
  `guard_filter` says which was used.
- Frames where two fighters' files follow the same person are not left out
  here (that's only an issue when two fighters are tracked in one clip).
Known limit: punch counts inherit the detector's false alarms (012): on
clinch_clip it reports 78 punches where 32 are labelled.
