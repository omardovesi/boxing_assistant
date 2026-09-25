# Boxing Coach AI

An app that analyzes sparring footage of a selected fighter and gives
offense/defense feedback, with progress tracking across sessions.
See SPEC.md for features and phases. See DECISIONS.md for past choices.

## Current phase
Phase 1: a command-line tool in /pipeline that takes a video, tracks
one selected fighter, and outputs an annotated video plus a JSON of
per-frame keypoints. No web app yet.

## Structure
- pipeline/  CV code (Python). Must stay independent of backend/frontend.
- backend/   FastAPI (not started)
- frontend/  Next.js (not started)
- data/      local test videos, gitignored, never commit

## Stack
- Python 3.11, virtual environment in .venv
- Pose model: TBD (see DECISIONS.md)

## How I want you to work
- I am learning. Explain what you're doing and why, briefly, before
  larger changes.
- For anything beyond a small change, propose a plan and wait for my
  approval before writing code.
- Keep changes small and focused on one task.
- Don't add new dependencies without telling me why.
- Write unit tests for metric and data-processing functions.
- Never commit videos, model weights, or .env files.
- Work on a branch, not main. Don't merge or push without asking me.
- When you make a design choice I didn't specify, tell me so I can
  record it in DECISIONS.md.

## Commands
- Tests: `.venv\Scripts\python -m pytest tests`
- Pick a fighter (saves outputs/analyze/<clip>/preview.jpg with numbered people):
  `.venv\Scripts\python -m pipeline.analyze "data/clip.mp4" --preview`
- Draw one fighter's skeleton (use the same --start as the preview):
  `.venv\Scripts\python -m pipeline.analyze "data/clip.mp4" --fighter 1 --duration 10`