# Boxing Coach AI — Spec

## Overview
A web app where a boxer uploads sparring footage, selects which fighter
to analyze, and gets feedback on offense and defense synced to video
playback. Sessions are saved so the app can show progress over time:
what improved and what still needs work.

## Target user
Amateur boxers and their coaches who record sparring on a phone and
want objective feedback between sessions.

## Core user flow (final product)
1. User uploads a sparring video.
2. User clicks on the fighter to analyze in the first frame.
3. App processes the video in the background and shows progress.
4. User watches the video with a skeleton overlay and timestamped
   feedback appearing as it plays. A toggle filters feedback to
   offense or defense.
5. Clicking a feedback item jumps to that moment in the video.
6. A progress dashboard shows how metrics change across sessions,
   with a written summary of improvements and weaknesses.

## Recording requirements
The analysis assumes:
- Single camera, roughly side-on to the fighters. Tripod is best;
  handheld with pans is OK (BoT-SORT compensates for camera motion,
  see DECISIONS.md 006)
- Both fighters fully in frame most of the time
- Decent lighting, 720p or higher, 30fps or higher
- Clips of 1–10 minutes

## Phases

### Phase 1 — CV pipeline CLI (current)
A command-line tool that runs on a local video file.

Done when:
- `python -m pipeline.analyze VIDEO --fighter ID` runs end to end
- It detects and tracks each person with a consistent ID
- A preview mode shows the first frame with numbered fighters so the
  user can choose an ID
- It outputs an annotated video with the selected fighter's skeleton
- It outputs a JSON file of per-frame keypoints for that fighter
- It works on at least 3 of my test clips, including one with clinching
- Known tracking failures are documented in DECISIONS.md

### Phase 2 — First metrics
Done when:
- Guard height and punch count are computed from keypoints
- Output JSON includes timestamped events (e.g. "left hand dropped")
- Each metric function has unit tests
- I've checked at least 20 events against the video by eye

### Phase 3 — Evaluation
- Hand-label 100–200 punches and guard drops across my clips
- Report accuracy of each metric in the README

### Phase 4 — Backend and database
- FastAPI, Postgres, background job queue, cloud video storage
- Upload a video, run the pipeline as a job, store metrics and events

### Phase 5 — Frontend
- Upload, click-to-select fighter, playback with overlay and synced
  feedback, offense/defense toggle, clickable events

### Phase 6 — LLM coaching
- LLM turns measured metrics and events into coaching feedback
- Every piece of feedback must reference a metric or timestamp

### Phase 7 — Progress tracking
- User accounts, session history, per-metric trends normalized per
  round/minute, written progress summary

## Metrics (planned)
Defense: guard height, return-to-guard time after punching, head
movement frequency, chin position, stance width.
Offense: punches per round, punch type (jab/cross/hook/uppercut),
combination length, lead vs rear hand ratio.
Conditioning: how metrics change round by round.

## Out of scope for now
- Live webcam analysis
- Detecting whether punches land
- Multi-camera footage
- Mobile app
- Analyzing both fighters at once

## Open questions
- ~~Which pose model works best on fast punches?~~ Resolved:
  YOLOv8-pose (DECISIONS.md 005)
- ~~How to handle tracking ID swaps during clinches?~~ Resolved:
  identify the fighter by kit on top of BoT-SORT (DECISIONS.md 006–007)
- How to detect round boundaries (manual input vs automatic)?