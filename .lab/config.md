# Lab Config

## Objective
Optimize meeting.rs pipeline speed: diarization + TDT transcription + speaker snippet extraction.
Must still produce speaker_N.wav snippets and a speaker-attributed transcript.

## Primary Metric
Wall-clock seconds — the `✓ Done in Xs` value printed by the binary.
**Direction**: lower is better.
**Target**: ≤ 16.8s (5% of 336.3s audio)

## Baseline
TBD (experiment #0)

## Best
TBD

## Test Audio
C:/Users/Albert/Downloads/meeting.wav  (336.3s, 16kHz mono)

## Run Command
```
export PATH="$HOME/.cargo/bin:$PATH" && cargo run --release --example meeting --features "sortformer cuda" -- C:/Users/Albert/Downloads/meeting.wav --identify --samples-dir ./speaker_samples 2>&1 | tail -5
```
Extract time: grep `✓ Done in` from output.

## Scope
- examples/meeting.rs (preferred)
- Cargo.toml (features)
- src/ if necessary

## Constraints
- Output: must produce speaker_N.wav per speaker + transcript printed to stdout
- Transcript quality irrelevant — speed only
- No external tools beyond what's in Cargo.toml

## Wall-clock Budget Per Experiment
5 minutes (kill + log as TIMEOUT if exceeded)

## Termination
Stop when primary metric ≤ 16.8s
