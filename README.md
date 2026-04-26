# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast speech recognition with NVIDIA's Parakeet models via ONNX Runtime.

Note: CoreML is unstable with this model. For Apple, use WebGPU EP (uses metal under the hood,dont confuse by its name :-). it's a native GPU standard, not only web) or CPU. But even CPU alone is significantly faster on my Mac M3 16GB compared to Whisper metal! :-)

## Models

**CTC (English-only)**:
```rust
use parakeet_rs::{Parakeet, Transcriber, TimestampMode};

let mut parakeet = Parakeet::from_pretrained(".", None)?;

// Load and transcribe audio (see examples/raw.rs for full example)
let result = parakeet.transcribe_samples(audio, 1600, 1, Some(TimestampMode::Words))?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**TDT (Multilingual)**: 25 languages with auto-detection
```rust
use parakeet_rs::{ParakeetTDT, Transcriber, TimestampMode};

let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;
let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Sentences))?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**EOU (Streaming)**: Real-time ASR with end-of-utterance detection
```rust
use parakeet_rs::ParakeetEOU;

let mut parakeet = ParakeetEOU::from_pretrained("./eou", None)?;

// Prepare your audio (Vec<f32>, 16kHz mono, normalized)
let audio: Vec<f32> = /* your audio samples */;

// Process in 160ms chunks for streaming
const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = parakeet.transcribe(chunk, false)?;
    print!("{}", text);
}
```

**Nemotron (Streaming)**: Cache-aware streaming ASR with punctuation
```rust
use parakeet_rs::Nemotron;

let mut model = Nemotron::from_pretrained("./nemotron", None)?;

// Process in 560ms chunks for streaming
const CHUNK_SIZE: usize = 8960; // 560ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = model.transcribe_chunk(chunk)?;
    print!("{}", text);
}
```

**Cohere Transcribe (Offline Multilingual)**: 14 languages, punctuation & ITN toggles (yes, "parakeets🦜" talk about more than just NVIDIA right?? :-P)
```toml
parakeet-rs = { version = "0.3", features = ["cohere"] }
```
```rust
use parakeet_rs::CohereASR;

let mut model = CohereASR::from_pretrained("./cohere", None)?;

// audio: Vec<f32>, 16kHz mono (long-form supported)
let text = model.transcribe_audio(&audio, "en", true, false)?; // lang, pnc, itn
println!("{}", text);
```
See `examples/cohere.rs` for a runnable demo.

**Multitalker (Streaming Multi-Speaker ASR)**: Speaker-attributed transcription
```toml
parakeet-rs = { version = "0.3", features = ["multitalker"] }
```
```rust
use parakeet_rs::MultitalkerASR;

let mut model = MultitalkerASR::from_pretrained(
    "./multitalker",             // encoder, decoder, tokenizer
    "sortformer.onnx",           // Sortformer v2 for diarization
    None,
)?;

for chunk in audio.chunks(17920) {  // ~1.12s at 16kHz
    let results = model.transcribe_chunk(chunk)?;
    for r in &results {
        println!("[Speaker {}] {}", r.speaker_id, r.text);
    }
}
```
See `examples/multitalker.rs` for full usage with latency modes.

**Sortformer v2 & v2.1 (Speaker Diarization)**: Streaming 4-speaker diarization
```toml
parakeet-rs = { version = "0.3", features = ["sortformer"] }
```
```rust
use parakeet_rs::sortformer::{Sortformer, DiarizationConfig};

let mut sortformer = Sortformer::with_config(
    "diar_streaming_sortformer_4spk-v2.onnx", // or v2.1.onnx
    None,
    DiarizationConfig::callhome(),  // or dihard3(),custom()
)?;
let segments = sortformer.diarize(audio, 16000, 1)?;
for seg in segments {
    println!("Speaker {} [{:.2}s - {:.2}s]", seg.speaker_id,
        seg.start as f64 / 16_000.0, seg.end as f64 / 16_000.0);
}

// For streaming/real-time use, diarize_chunk() preserves state across calls:
let segments = sortformer.diarize_chunk(&audio_chunk_16k_mono)?;
```
See `examples/diarization.rs` for combining with TDT transcription.

See `examples/streaming_diarization.rs` for `diarize_chunk` usage example.

See `scripts/export_diar_sortformer.py` for exporting the model with custom streaming parameters.

## Setup

**CTC**: Download from [HuggingFace](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx): `model.onnx`, `model.onnx_data`, `tokenizer.json`

**TDT**: Download from [HuggingFace](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx): `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt`

**EOU**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx): `encoder.onnx`, `decoder_joint.onnx`, `tokenizer.json`

**Nemotron**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b): `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model` (*[int8](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8) / [int4](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int4)*)

**Unified**: Download from [HuggingFace](https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx): `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model`

**Multitalker**: Download from [HuggingFace](https://huggingface.co/smcleod/multitalker-parakeet-streaming-0.6b-v1-onnx-int8/tree/main): `encoder.int8.onnx`, `decoder_joint.int8.onnx`, `tokenizer.model` (also needs a Sortformer model for diarization)

**Cohere Transcribe**: Download from [HuggingFace](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX): `encoder_model.onnx` (+ `.onnx_data*`), `decoder_model_merged.onnx` (+ `.onnx_data`), `tokenizer.json` (FP32, FP16, INT8, INT4 variants available)

**Diarization (Sortformer v2 & v2.1)**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main): `diar_streaming_sortformer_4spk-v2.onnx` or `v2.1.onnx`.

Quantized versions available (int8). All files must be in the same directory.

GPU support (auto-falls back to CPU if fails):
```toml
parakeet-rs = { version = "0.3", features = ["cuda"] }  # or tensorrt, webgpu, directml, migraphx or other ort supported EPs (check cargo features)
```

```rust
use parakeet_rs::{Parakeet, ExecutionConfig, ExecutionProvider};

let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
let mut parakeet = Parakeet::from_pretrained(".", Some(config))?;
```

Advanced session configuration via [ort SessionBuilder](https://docs.rs/ort/latest/ort/session/builder/struct.SessionBuilder.html):
```rust
let config = ExecutionConfig::new()
    .with_custom_configure(|builder| builder.with_memory_pattern(false));
```

## Features

- [CTC: English with punctuation & capitalization](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [TDT: Multilingual (auto lang detection)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [EOU: Streaming ASR with end-of-utterance detection](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- [Nemotron: Cache aware streaming ASR (600M params,EN only)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Unified: Offline + buffered streaming RNNT ASR (600M params, EN only)](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
- [Multitalker: Streaming multi-speaker ASR with speaker-kernel injection](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) ([ONNX int8](https://huggingface.co/smcleod/multitalker-parakeet-streaming-0.6b-v1-onnx-int8))
- [Cohere Transcribe: Offline multilingual ASR (14 languages, long-form supported)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) ([ONNX](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX))
- [Sortformer v2 & v2.1: Streaming speaker diarization (up to 4 speakers)](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) NOTE: you can also download v2.1 model same way.
- Token-level timestamps (CTC, TDT)

## Meeting Pipeline (Diarization + Transcription)

`examples/meeting.rs` and `examples/meeting_worker.rs` combine Sortformer diarization with TDT transcription into a full meeting pipeline. Both run diarization and transcription in parallel on GPU.

**Benchmarks** (RTX 2070 Super, CUDA EP, ORT 1.25):

| Audio | Mode | Time | % of duration |
|-------|------|------|---------------|
| 5m 36s | standalone cold start | 6.4s | 1.9% |
| 5m 36s | worker — first run | 4.0s | 1.2% |
| 5m 36s | worker — cached diar | 2.4s | 0.7% |
| 39m 41s | standalone cold start | 24.2s | 1.0% |
| 39m 41s | worker — first run | 22.2s | 0.9% |
| 39m 41s | worker — cached diar | 17.2s | 0.7% |

### Standalone (`examples/meeting.rs`)

Loads models fresh each run. Best for one-off transcriptions.

```bash
cargo build --release --example meeting --features "sortformer load-dynamic cuda"

# Run (audio must be 16kHz mono WAV)
ORT_DYLIB_PATH="/path/to/onnxruntime.dll" \
./target/release/examples/meeting path/to/recording.wav \
  --samples-dir ./speaker_samples --cuda
```

Output: timestamped transcript with speaker labels. On first run use `--identify` to extract per-speaker audio clips, then fill in `speakers.json` to get named output on subsequent runs.

### Worker (`examples/meeting_worker.rs`)

Persistent daemon that loads both models once (~5.5s) and keeps them in GPU memory. Amortizes model load across all requests. Repeat runs of the same file skip diarization entirely (cache hit).

```bash
cargo build --release --example meeting_worker --features "sortformer load-dynamic cuda"

# Start daemon
./target/release/examples/meeting_worker.exe --cuda --cache-dir .diar_cache

# Send requests as JSON lines on stdin:
{"audio":"recording.wav","samples_dir":"./speaker_samples","cache_dir":".diar_cache"}

# Quit:
{"quit":true}
```

Response JSON:
```json
{"status":"ok","elapsed_s":2.37,"diar_source":"cache","speakers":[0,1,2,3]}
```

**Diarization cache**: results are stored as `<cache_dir>/<metadata-hash>.diar.json`. The cache key is derived from the file's mtime + size (no content read needed). Speaker audio clips are saved as `<cache_dir>/speakers/<hash>_<spk_id>.emb` (raw f32, 5s per speaker) for cross-file identity transfer.

**GPU-optimal parameters** (tuned empirically):
- `chunk_len=80, fifo_len=80` — fewer ONNX kernel launches vs smaller chunks
- `spkcache_len=40` — quality floor; below 30 misses speakers in long 4-speaker meetings
- No CUDA stagger between diar+TDT threads — JIT already done at model load

### Web Service (`server/`)

Bun.js + TypeScript HTTP server wrapping the worker. Accepts **any audio format ffmpeg understands** — MP3, M4A, OGG, FLAC, WAV, etc. Handles conversion, caching, and worker IPC automatically.

**Setup:**
```bash
cd server && bun install
```

**Start (from project root):**
```bash
export ORT_DYLIB_PATH="/c/Python312/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
export PATH="/c/Users/$USER/AppData/Local/uv/cache/archive-v0/<hash>/torch/lib:$PATH"

bun run server/index.ts
# [server] listening on http://localhost:3000
# [worker] ready (load=5.5s)
```

**Endpoints:**

`GET /health`
```bash
curl http://localhost:3000/health
# {"status":"ok","worker":"...meeting_worker.exe"}
```

`POST /transcribe` — file upload (any format):
```bash
# MP3, M4A, OGG, FLAC — anything ffmpeg reads
curl -X POST http://localhost:3000/transcribe \
  -F "audio=@meeting.mp3"

# Second request for the same file: ~100ms (full cache hit)
curl -X POST http://localhost:3000/transcribe \
  -F "audio=@meeting.mp3"
```

`POST /transcribe` — JSON body with local file path:
```bash
curl -X POST http://localhost:3000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio": "/abs/path/to/meeting.mp3"}'
```

Optional JSON fields: `cache_dir`, `samples_dir`, `speakers` (path to speakers.json), `output` (save transcript to file), `identify` (extract speaker clips).

**Response:**
```json
{
  "status": "ok",
  "elapsed_s": 17.1,
  "diar_source": "cache",
  "transcript_source": "live",
  "speakers": [0, 1, 2, 3],
  "transcript": "[00:00 – 00:04] Speaker 0:\n  Hello everyone.\n..."
}
```
On a full cache hit `transcript_source` is `"cache"` and the response returns in ~100ms regardless of audio length.

**Three-layer cache** (all keyed by SHA-256 of the audio content):

| Layer | Path | Saves |
|-------|------|-------|
| WAV conversion | `.diar_cache/wav/<hash>.wav` | ffmpeg re-conversion |
| Diarization | `.diar_cache/<meta-hash>.diar.json` | Sortformer inference (~4–20s) |
| Full transcript | `.diar_cache/transcript/<hash>.json` | Everything — TDT + merge |

**Speaker endpoints:**

`GET /speakers/:hash` — list clips for an audio file:
```bash
curl http://localhost:3000/speakers/<hash>
# {"hash":"6c46...","clips":[{"id":0,"url":"/speakers/<hash>/0","file":"speaker_0.wav"},...]}`
```

`GET /speakers/:hash/:id` — download a speaker clip (5s WAV):
```bash
curl -o speaker_0.wav http://localhost:3000/speakers/<hash>/0
```

Clips are scoped per audio file under `speaker_samples/<hash>/speaker_N.wav` so different meetings never overwrite each other. The hash comes from the `POST /transcribe` response or the `/speakers/:hash` listing.

**Speaker identification flow:**
```bash
# Step 1 — extract speaker clips (bypasses transcript cache to regenerate clips)
curl -X POST http://localhost:3000/transcribe \
  -F "audio=@meeting.mp3" \
  -F "identify=true"
# Response includes "speaker_clips": {"0": "...speaker_0.wav", "1": "...", ...}
# and the hash needed for the /speakers endpoints

# Step 2 — list and download clips to identify each speaker
HASH="<hash from response>"
curl http://localhost:3000/speakers/$HASH
curl -o alice.wav http://localhost:3000/speakers/$HASH/0
curl -o bob.wav   http://localhost:3000/speakers/$HASH/1

# Step 3 — fill in names and re-transcribe (instant from cache)
cat > speakers.json << 'EOF'
{"0": "Alice", "1": "Bob", "2": "Carol", "3": "Dave"}
EOF

curl -X POST http://localhost:3000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio":"/abs/path/meeting.mp3","speakers":"speakers.json"}'
```

**Environment variables** (all have defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `3000` | HTTP port |
| `WORKER_EXE` | `target/release/examples/meeting_worker.exe` | Worker binary path |
| `CACHE_DIR` | `.diar_cache` | Root cache directory |
| `SAMPLES_DIR` | `speaker_samples` | Speaker clip output directory |
| `FFMPEG` | `ffmpeg` | ffmpeg binary path |
| `NO_CUDA` | unset | Set to `1` to disable CUDA |
| `WORKER_TIMEOUT_MS` | `300000` | Per-request timeout (ms) |
| `ORT_DYLIB_PATH` | Python onnxruntime-gpu DLL | Path to ORT shared library |
| `CUDA_LIBS` | PyTorch uv cache torch/lib | Directory containing cuDNN/cuBLAS DLLs |

### CUDA Setup (Windows)

CUDA EP requires ORT 1.25 from the Python `onnxruntime-gpu` package and cuDNN 9 + cuBLAS 12 from the PyTorch uv cache:

```bash
export ORT_DYLIB_PATH="/c/Python312/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
export PATH="/c/Users/$USER/AppData/Local/uv/cache/archive-v0/<hash>/torch/lib:$PATH"

cargo build --release --example meeting --features "sortformer load-dynamic cuda"
```

ORT 1.19 does **not** work with CUDA EP for dynamic-shape Sortformer graphs — use ORT 1.25 via the Python package.

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float) for the Rust API and standalone/worker examples. The web service accepts any format and converts automatically via ffmpeg.
- CTC/TDT models have ~4-5 minute audio length limit. For longer files, use streaming models or split into chunks.

## License

Code: MIT OR Apache-2.0

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) by NVIDIA. This library does not distribute the models.
