# SetuSign — Project Memory

> This file is the single source of truth for goals, rules, decisions, and current phase.
> Updated after every phase milestone.

## Product Summary

SetuSign is an offline, two-way Indian Sign Language (ISL) communicator for service counters
(clinic reception, bank, government office), running entirely on-device on Snapdragon-powered HP PCs
(Hexagon NPU). Individual submission to Qualcomm's Snapdragon® AI Lab Build & Present Challenge 2026.

### Directions
- **Sign → text/speech**: hand + pose keypoints → sign recognizer → gloss-to-sentence LLM → on-screen text + TTS.
- **Speech → captions**: Whisper → large captions for the deaf user.
- **On-device personalization**: freeze encoder, fine-tune lightweight classifier head locally (3–5 examples per word).
- **Confidence gate**: below threshold, show top-3 candidates for one-tap confirm.
- **Privacy**: skeleton keypoints only; never store or upload video.

### Scope
- Isolated signs only.
- Vocabulary: INCLUDE-50 (AI4Bharat) + ~20 counter-mode phrases.
- English output first.
- No sign-avatar generation.

---

## Hard Rules

1. **Honesty over appearance.** Never simulate, mock, or hardcode metrics, NPU utilization, or inference results.
   Every number in logs, the README, or the UI must come from a real run and be reproducible by a script in the repo.
2. **Always log the execution provider** actually used (QNN/NPU vs CPU) for every model.
   If QNN is unavailable, fall back to CPU and say so loudly. Never claim NPU execution you haven't verified.
3. **Verify, don't assume.** Before using Qualcomm AI Hub, qai-hub-models, onnxruntime-qnn, or any model,
   check the current official docs and package metadata for supported Python versions, CPU architectures
   (ARM64 vs x64/emulation), and APIs. Document findings and cite sources.
   Prefer native ARM64 runtime path; if a tool only works under x64 emulation, isolate it and document why.
4. **Secrets**: read the AI Hub API token from env var `QAI_HUB_API_TOKEN`. Never print it, commit it, or write it to a file.
   `.env` is in `.gitignore`.
5. **Ask before any download over 1 GB** and before installing anything system-wide.
   The full INCLUDE dataset is ~57 GB; download only what INCLUDE-50 needs, using the official train/test split files.
6. **Small, meaningful git commits** with descriptive messages after each working unit.
   No giant "initial commit" dumps. Never rewrite history.
7. **Attribute third-party code, datasets, and models** in README and THIRD_PARTY.md.
8. **Stop at the end of each phase**, report results, and wait for go-ahead.

---

## Phase Plan

| Phase | Description | Priority |
|-------|-------------|----------|
| 0 | Scaffold and environment check | Must |
| 1 | Data and baseline | Must |
| 2 | Hand/pose landmark models via AI Hub; ONNX Runtime QNN EP | Must |
| 3 | Own transformer recognizer; quantize via AI Hub | Must |
| 4 | Signer-independent test tooling | Must |
| 5 | Personalization + confidence gate | Must |
| 6 | Gloss-to-sentence with small on-device LLM | Should |
| 7 | Whisper speech→captions and TTS | Should (cut first) |
| 8 | Concurrency + power benchmark | Must |
| 9 | Accessible desktop UI, installer, final docs | Must |

---

## Decisions Log

### 2026-09-30 — Phase 0 scaffold
- **Dev machine**: Intel i5-1335U (x86-64), no Snapdragon NPU. All code uses CPU fallback; NPU testing deferred to Snapdragon device.
- **Python version**: 3.12 chosen over 3.14 because MediaPipe only supports ≤3.12. ONNX Runtime works with both.
- **Repo root**: `G:\snap\setusign`
- **Downloads**: all large data/model downloads go to `G:\` drive.

---

## Current Phase

**Phase 0 — Scaffold and environment check** (in progress)
