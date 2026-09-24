# SetuSign

> Offline, two-way Indian Sign Language communicator for service counters,
> optimized for Snapdragon-powered HP PCs (Hexagon NPU).

**Qualcomm Snapdragon® AI Lab Build & Present Challenge 2026** — Individual submission

---

## What It Does

| Direction | Pipeline |
|-----------|----------|
| **Sign → Text/Speech** | Camera → hand/pose keypoints → sign recognizer → gloss-to-sentence LLM → on-screen text + TTS |
| **Speech → Captions** | Microphone → Whisper ASR → large on-screen captions for the deaf user |

### Key Features
- Runs **entirely on-device** — no cloud, no internet required.
- **Privacy-first**: processes skeleton keypoints only; never stores or uploads video.
- **On-device personalization**: a new user enrolls by signing each word 3–5 times; only a lightweight classifier head is fine-tuned.
- **Confidence gate**: below threshold, shows top-3 candidates for one-tap confirmation.
- Vocabulary: INCLUDE-50 (AI4Bharat, 50 ISL signs) + ~20 counter-mode phrases.

---

## Project Status

| Phase | Status |
|-------|--------|
| 0 — Scaffold & env check | 🔄 In progress |
| 1 — Data & baseline | ⬜ Not started |
| 2 — Landmark models via AI Hub | ⬜ Not started |
| 3 — Transformer recognizer | ⬜ Not started |
| 4 — Signer-independent testing | ⬜ Not started |
| 5 — Personalization + confidence | ⬜ Not started |
| 6 — Gloss-to-sentence LLM | ⬜ Not started |
| 7 — Whisper + TTS | ⬜ Not started |
| 8 — Concurrency & power bench | ⬜ Not started |
| 9 — UI, installer, final docs | ⬜ Not started |

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd setusign

# 2. Create Python 3.12 venv
py -3.12 -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Check environment
python env/check_env.py
```

---

## Repository Structure

```
setusign/
├── CLAUDE.md          # Project memory: goals, rules, decisions
├── README.md          # This file
├── THIRD_PARTY.md     # Attribution for third-party code/data/models
├── requirements.txt   # Python dependencies
├── env/               # Environment setup notes + check scripts
├── data/              # Gitignored raw data; manifests committed
├── models/            # Export/compile scripts; artifacts gitignored
├── src/setusign/      # App code
│   ├── keypoints/     # Hand/pose landmark extraction
│   ├── recognizer/    # Sign recognition model
│   ├── personalization/ # On-device fine-tuning
│   ├── llm/           # Gloss-to-sentence
│   ├── asr/           # Whisper speech-to-text
│   ├── tts/           # Text-to-speech
│   └── ui/            # Desktop UI
├── bench/             # Reproducible benchmarks
│   └── results/       # JSON results from benchmark runs
├── tests/             # Unit and integration tests
└── docs/              # Architecture, benchmarks, limitations
```

---

## Hardware Requirements

- **Target**: Snapdragon X Elite / X Plus powered HP PC with Hexagon NPU
- **Development**: Any Windows PC with Python 3.12 (CPU fallback)
- **Camera**: USB or built-in webcam
- **Microphone**: Built-in or external

---

## Third-Party Attribution

See [THIRD_PARTY.md](THIRD_PARTY.md) for full attribution of datasets, models, and libraries.

---

## License

TBD
