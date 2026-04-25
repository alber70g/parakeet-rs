# Experiment Log

## Experiment 0 — Baseline CPU
Branch: research/meeting-speed / Type: real / Parent: none / Commit: 4bfadd6
Hypothesis: Measure raw CPU performance to set baseline.
Changes: none
Result: TIMEOUT at 5 min — Sortformer diarization alone exceeds budget on 336s audio. CPU is completely non-viable.
Duration: >300s / Status: crash (timeout)
Insight: Entire win must come from GPU. Sortformer on CPU is O(audio_len²) in attention — 336s of audio is too large.
Next: Get onnxruntime.dll (CUDA) working via load-dynamic + ORT_DYLIB_PATH.
