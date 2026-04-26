/*
Meeting Worker — Persistent Model Server

Loads Sortformer + TDT once, keeps them in memory across multiple audio files.
Read requests from stdin (one JSON line per request), prints JSON results to stdout.

This eliminates the ~3-4s model load + CUDA JIT cost paid on every invocation of
the standalone `meeting` example.

USAGE
─────
# Build
cargo build --release --example meeting_worker --features "sortformer load-dynamic cuda"

# Start server
./target/release/examples/meeting_worker.exe --cuda

# Send requests (one JSON per line):
{"audio":"recording.wav","samples_dir":"./speaker_samples"}
{"audio":"meeting2.wav","samples_dir":"./speaker_samples","cache_dir":"./.diar_cache"}
{"audio":"meeting2.wav","samples_dir":"./speaker_samples","cache_dir":"./.diar_cache"}  <- cached!

# Quit
{"quit":true}

CACHING
───────
--cache-dir: saves diarization result as <sha256_of_audio_bytes>.diar.json
  Re-running the same audio file skips Sortformer entirely (saves ~4s).

SPEAKER EMBEDDINGS
──────────────────
Save speaker cache embeddings after each run to <cache_dir>/speakers/<run_hash>_spk<N>.emb
These are raw f32 arrays (spkcache_len * EMB_DIM floats) that can be loaded back to
pre-populate Sortformer's speaker cache for a new recording (speaker identity transfer).

Note: The speaker embedding reuse API is exposed but experimental — Sortformer resets
its state on each diarize() call, so pre-loading embeddings requires a custom diarize
path (see diarize_with_prior_speakers below).
*/

#[cfg(feature = "sortformer")]
use hound::{WavSpec, WavWriter};
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};
#[cfg(feature = "sortformer")]
use std::collections::HashMap;
#[cfg(feature = "sortformer")]
use std::io::{BufRead, Write};
#[cfg(feature = "sortformer")]
use std::path::{Path, PathBuf};
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[cfg(feature = "sortformer")]
const TDT_CHUNK_SAMPLES: usize = 16_000 * 30;
#[cfg(feature = "sortformer")]
const MIN_SPEAKER_SAMPLE_SECS: f32 = 1.0;
#[cfg(feature = "sortformer")]
const SPEAKER_SAMPLE_SECS: f32 = 5.0;

// ── Cached diarization result (serialized to JSON) ────────────────────────────

#[cfg(feature = "sortformer")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct CachedSegment {
    start: u64,
    end: u64,
    speaker_id: usize,
}

#[cfg(feature = "sortformer")]
impl From<&parakeet_rs::sortformer::SpeakerSegment> for CachedSegment {
    fn from(s: &parakeet_rs::sortformer::SpeakerSegment) -> Self {
        Self { start: s.start, end: s.end, speaker_id: s.speaker_id }
    }
}

#[cfg(feature = "sortformer")]
impl From<&CachedSegment> for parakeet_rs::sortformer::SpeakerSegment {
    fn from(s: &CachedSegment) -> Self {
        parakeet_rs::sortformer::SpeakerSegment { start: s.start, end: s.end, speaker_id: s.speaker_id }
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
#[derive(serde::Deserialize, Default)]
struct Request {
    audio: Option<String>,
    tdt_dir: Option<String>,
    sortformer_model: Option<String>,
    speakers: Option<String>,
    output: Option<String>,
    samples_dir: Option<String>,
    cache_dir: Option<String>,
    identify: Option<bool>,
    quit: Option<bool>,
}

#[cfg(feature = "sortformer")]
#[derive(serde::Serialize)]
struct Response {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_s: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diar_source: Option<String>, // "live" or "cache"
    #[serde(skip_serializing_if = "Option::is_none")]
    speakers: Option<Vec<usize>>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
fn load_wav(path: &str) -> Result<(Vec<f32>, u32, u16), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok((audio, spec.sample_rate, spec.channels))
}

#[cfg(feature = "sortformer")]
fn to_mono(audio: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels == 1 { return audio; }
    audio.chunks(channels as usize).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
}

/// Simple non-cryptographic hash of file contents for cache key.
/// Uses FNV-1a 64-bit for speed — fast enough for 336s audio (5MB).
#[cfg(feature = "sortformer")]
fn file_hash(path: &str) -> std::io::Result<String> {
    let data = std::fs::read(path)?;
    let mut hash: u64 = 14695981039346656037u64; // FNV offset basis
    for &byte in &data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211u64); // FNV prime
    }
    Ok(format!("{:016x}", hash))
}

#[cfg(feature = "sortformer")]
fn format_time(secs: f32) -> String {
    let total = secs as u32;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

#[cfg(feature = "sortformer")]
fn save_speaker_sample(
    audio: &[f32], start_sample: usize, end_sample: usize, path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let spec = WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 32,
                         sample_format: hound::SampleFormat::Float };
    let mut writer = WavWriter::create(path, spec)?;
    let end = end_sample.min(audio.len());
    for &s in &audio[start_sample..end] { writer.write_sample(s)?; }
    writer.finalize()?;
    Ok(())
}

/// Save raw f32 speaker embedding (spkcache flattened) to a .emb file.
/// Format: 4-byte little-endian u32 count, then count * 4-byte f32 values.
#[cfg(feature = "sortformer")]
fn save_speaker_embedding(data: &[f32], path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let mut f = std::fs::File::create(path)?;
    let count = data.len() as u32;
    f.write_all(&count.to_le_bytes())?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Load a .emb file back to a Vec<f32>.
#[cfg(feature = "sortformer")]
#[allow(dead_code)]
fn load_speaker_embedding(path: &Path) -> std::io::Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut count_bytes = [0u8; 4];
    f.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;
    let mut data = vec![0.0f32; count];
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf)?;
    for i in 0..count {
        data[i] = f32::from_le_bytes([buf[4*i], buf[4*i+1], buf[4*i+2], buf[4*i+3]]);
    }
    Ok(data)
}

// ── Build execution configs ───────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
fn build_tdt_config(use_cuda: bool) -> ExecutionConfig {
    #[cfg(feature = "cuda")]
    if use_cuda {
        return ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cuda)
            .with_intra_threads(1);
    }
    let _ = use_cuda;
    ExecutionConfig::new().with_intra_threads(4).with_inter_threads(0)
}

#[cfg(feature = "sortformer")]
fn build_sortformer_config(use_cuda: bool) -> ExecutionConfig {
    #[cfg(feature = "cuda")]
    if use_cuda {
        return ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cuda)
            .with_intra_threads(1);
    }
    let _ = use_cuda;
    ExecutionConfig::new().with_intra_threads(2).with_inter_threads(0)
}

// ── Process one audio request ─────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
#[allow(clippy::too_many_arguments)]
fn process_audio(
    audio_path: &str,
    samples_dir: &str,
    speakers_file: Option<&str>,
    output_file: Option<&str>,
    identify_mode: bool,
    cache_dir: Option<&str>,
    sortformer: &mut Sortformer,
    tdt: &mut ParakeetTDT,
    use_cuda: bool,
) -> Result<(f32, String, Vec<usize>), Box<dyn std::error::Error>> {
    let req_start = Instant::now();

    // ── Load audio ────────────────────────────────────────────────────────────
    let (raw_audio, sample_rate, channels) = load_wav(audio_path)?;
    let audio = to_mono(raw_audio, channels);

    if sample_rate != 16_000 {
        return Err(format!("Audio must be 16kHz, got {}Hz", sample_rate).into());
    }

    // ── Diarization cache lookup ───────────────────────────────────────────────
    let (speaker_segments, diar_source) = if let Some(cache) = cache_dir {
        let hash = file_hash(audio_path)?;
        let cache_path = PathBuf::from(cache).join(format!("{}.diar.json", hash));

        if cache_path.exists() {
            let json = std::fs::read_to_string(&cache_path)?;
            let cached: Vec<CachedSegment> = serde_json::from_str(&json)?;
            let segs: Vec<parakeet_rs::sortformer::SpeakerSegment> =
                cached.iter().map(|s| s.into()).collect();
            (segs, "cache".to_string())
        } else {
            // Run diarization and cache result
            let segs = sortformer.diarize(audio.clone(), 16_000, 1)?;
            std::fs::create_dir_all(cache)?;
            let cached: Vec<CachedSegment> = segs.iter().map(|s| s.into()).collect();
            std::fs::write(&cache_path, serde_json::to_string(&cached)?)?;

            // Save per-speaker embeddings
            save_speaker_embeddings_from_segs(&segs, &audio, cache, &hash);

            (segs, "live".to_string())
        }
    } else {
        let segs = sortformer.diarize(audio.clone(), 16_000, 1)?;
        (segs, "live".to_string())
    };

    // ── Unique speakers ───────────────────────────────────────────────────────
    let mut unique_speakers: Vec<usize> = {
        let mut ids: Vec<usize> = speaker_segments.iter().map(|s| s.speaker_id).collect();
        ids.sort_unstable(); ids.dedup(); ids
    };
    unique_speakers.sort_unstable();

    // ── Speaker name mapping ───────────────────────────────────────────────────
    let mut speaker_names: HashMap<usize, String> = HashMap::new();
    if let Some(sf) = speakers_file {
        if Path::new(sf).exists() {
            let json = std::fs::read_to_string(sf)?;
            let raw: HashMap<String, String> = serde_json::from_str(&json)?;
            for (k, v) in raw {
                if let Ok(id) = k.parse::<usize>() { speaker_names.insert(id, v); }
            }
        }
    }

    // ── Identify mode ─────────────────────────────────────────────────────────
    if identify_mode {
        for &spk in &unique_speakers {
            let name = speaker_names.get(&spk).cloned()
                .unwrap_or_else(|| format!("Speaker {}", spk));
            let best = speaker_segments.iter()
                .filter(|s| s.speaker_id == spk)
                .max_by_key(|s| s.end - s.start);
            if let Some(seg) = best {
                let seg_start_s = seg.start as f32 / 16_000.0;
                let seg_end_s = seg.end as f32 / 16_000.0;
                let seg_len = seg_end_s - seg_start_s;
                if seg_len < MIN_SPEAKER_SAMPLE_SECS { continue; }
                let sample_end_s = (seg_start_s + SPEAKER_SAMPLE_SECS).min(seg_end_s);
                let start_idx = seg.start as usize;
                let end_idx = (sample_end_s * 16_000.0) as usize;
                let sample_path = PathBuf::from(samples_dir)
                    .join(format!("speaker_{}.wav", spk));
                save_speaker_sample(&audio, start_idx, end_idx, &sample_path)?;
                println!("  Speaker {} ({}): [{} – {}] → {}",
                    spk, name, format_time(seg_start_s), format_time(sample_end_s),
                    sample_path.display());
            }
        }
        return Ok((req_start.elapsed().as_secs_f32(), diar_source, unique_speakers));
    }

    // ── Transcription ─────────────────────────────────────────────────────────
    let total_samples = audio.len();
    let mut timed_sentences: Vec<(f32, f32, String)> = Vec::new();
    let mut offset = 0usize;
    let _ = use_cuda;
    while offset < total_samples {
        let end = (offset + TDT_CHUNK_SAMPLES).min(total_samples);
        let chunk = &audio[offset..end];
        let time_offset = offset as f32 / 16_000.0;
        let result = tdt.transcribe_samples(chunk.to_vec(), 16_000, 1, Some(TimestampMode::Sentences))?;
        for t in result.tokens {
            timed_sentences.push((t.start + time_offset, t.end + time_offset, t.text));
        }
        offset = end;
    }

    // ── Merge transcript + diarization ────────────────────────────────────────
    let mut segments: Vec<(f32, f32, usize, String)> = timed_sentences.into_iter()
        .filter(|(_, _, text)| !text.trim().is_empty())
        .map(|(start, end, text)| {
            let speaker_id = speaker_segments.iter()
                .filter_map(|s| {
                    let s_start = s.start as f32 / 16_000.0;
                    let s_end = s.end as f32 / 16_000.0;
                    let overlap = (end.min(s_end) - start.max(s_start)).max(0.0);
                    if overlap > 0.0 { Some((s.speaker_id, overlap)) } else { None }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(id, _)| id)
                .unwrap_or(usize::MAX);
            (start, end, speaker_id, text)
        })
        .collect();

    // Merge consecutive same-speaker segments
    let mut merged: Vec<(f32, f32, usize, String)> = Vec::new();
    for seg in segments.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.2 == seg.2 && seg.0 - last.1 < 2.0 {
                last.1 = seg.1;
                last.3.push(' ');
                last.3.push_str(seg.3.trim());
                continue;
            }
        }
        merged.push(seg);
    }

    // ── Print transcript ──────────────────────────────────────────────────────
    println!("=== Transcript: {} ===\n", audio_path);
    for (start, end, spk_id, text) in &merged {
        let name = if *spk_id == usize::MAX { "UNKNOWN".to_string() } else {
            speaker_names.get(spk_id).cloned()
                .unwrap_or_else(|| format!("Speaker {}", spk_id))
        };
        println!("[{} – {}] {}:", format_time(*start), format_time(*end), name);
        println!("  {}\n", text.trim());
    }

    // ── Write to file ─────────────────────────────────────────────────────────
    if let Some(out) = output_file {
        use std::io::Write;
        let mut f = std::fs::File::create(out)?;
        for (start, end, spk_id, text) in &merged {
            let name = if *spk_id == usize::MAX { "UNKNOWN".to_string() } else {
                speaker_names.get(spk_id).cloned()
                    .unwrap_or_else(|| format!("Speaker {}", spk_id))
            };
            writeln!(f, "[{} – {}] {}:", format_time(*start), format_time(*end), name)?;
            writeln!(f, "  {}\n", text.trim())?;
        }
    }

    Ok((req_start.elapsed().as_secs_f32(), diar_source, unique_speakers))
}

/// Save per-speaker audio snippets as .emb reference data for identity transfer.
/// Each speaker's 5s clip is saved as raw f32 audio for potential future embedding.
#[cfg(feature = "sortformer")]
fn save_speaker_embeddings_from_segs(
    segs: &[parakeet_rs::sortformer::SpeakerSegment],
    audio: &[f32],
    cache_dir: &str,
    run_hash: &str,
) {
    let emb_dir = PathBuf::from(cache_dir).join("speakers");
    let _ = std::fs::create_dir_all(&emb_dir);

    let mut by_speaker: HashMap<usize, Vec<&parakeet_rs::sortformer::SpeakerSegment>> = HashMap::new();
    for s in segs { by_speaker.entry(s.speaker_id).or_default().push(s); }

    for (&spk_id, spk_segs) in &by_speaker {
        // Find longest segment
        let best = spk_segs.iter().max_by_key(|s| s.end - s.start);
        if let Some(seg) = best {
            let start = seg.start as usize;
            let end = ((seg.start as f32 / 16_000.0 + 5.0) * 16_000.0) as usize;
            let end = end.min(seg.end as usize).min(audio.len());
            if end > start + 16_000 { // at least 1s
                let clip = &audio[start..end];
                let path = emb_dir.join(format!("{}_{}.emb", run_hash, spk_id));
                let _ = save_speaker_embedding(clip, &path);
            }
        }
    }
}

// ── CLI args ──────────────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
struct ServerArgs {
    tdt_dir: String,
    sortformer_model: String,
    use_cuda: bool,
    cache_dir: Option<String>,
}

#[cfg(feature = "sortformer")]
fn parse_server_args() -> Result<ServerArgs, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = ServerArgs {
        tdt_dir: "./tdt".to_string(),
        sortformer_model: "diar_streaming_sortformer_4spk-v2.onnx".to_string(),
        use_cuda: false,
        cache_dir: None,
    };
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--tdt" => { i += 1; args.tdt_dir = raw[i].clone(); }
            "--model" => { i += 1; args.sortformer_model = raw[i].clone(); }
            "--cuda" => { args.use_cuda = true; }
            "--cache-dir" => { i += 1; args.cache_dir = Some(raw[i].clone()); }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        i += 1;
    }
    Ok(args)
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("This example requires the 'sortformer' feature.");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let server_args = parse_server_args().map_err(|e| { eprintln!("{}", e); e })?;

        let startup = Instant::now();
        eprintln!("[worker] Loading models...");

        // Load Sortformer
        let mut sortformer = Sortformer::with_config(
            &server_args.sortformer_model,
            Some(build_sortformer_config(server_args.use_cuda)),
            DiarizationConfig::callhome(),
        )?;
        sortformer.chunk_len = 31;
        sortformer.fifo_len = 31;
        sortformer.spkcache_len = 47;

        // Load TDT (500ms stagger when CUDA to avoid JIT collision)
        if server_args.use_cuda {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        let mut tdt = ParakeetTDT::from_pretrained(&server_args.tdt_dir,
            Some(build_tdt_config(server_args.use_cuda)))?;

        eprintln!("[worker] Models loaded in {:.1}s. Ready.", startup.elapsed().as_secs_f32());

        // Print ready signal so callers can detect startup completion
        println!("{{\"status\":\"ready\",\"load_s\":{:.2}}}", startup.elapsed().as_secs_f32());
        std::io::stdout().flush()?;

        // ── Request loop ──────────────────────────────────────────────────────
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let req: Request = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    let resp = Response {
                        status: "error".to_string(),
                        error: Some(format!("JSON parse error: {}", e)),
                        elapsed_s: None, diar_source: None, speakers: None,
                    };
                    println!("{}", serde_json::to_string(&resp)?);
                    std::io::stdout().flush()?;
                    continue;
                }
            };

            if req.quit.unwrap_or(false) {
                eprintln!("[worker] Quit requested.");
                break;
            }

            let audio_path = match &req.audio {
                Some(p) => p.clone(),
                None => {
                    let resp = Response {
                        status: "error".to_string(),
                        error: Some("Missing 'audio' field".to_string()),
                        elapsed_s: None, diar_source: None, speakers: None,
                    };
                    println!("{}", serde_json::to_string(&resp)?);
                    std::io::stdout().flush()?;
                    continue;
                }
            };

            let samples_dir = req.samples_dir.as_deref().unwrap_or("./speaker_samples");
            let cache_dir_req = req.cache_dir.as_deref()
                .or(server_args.cache_dir.as_deref());
            let identify = req.identify.unwrap_or(false);

            match process_audio(
                &audio_path,
                samples_dir,
                req.speakers.as_deref(),
                req.output.as_deref(),
                identify,
                cache_dir_req,
                &mut sortformer,
                &mut tdt,
                server_args.use_cuda,
            ) {
                Ok((elapsed, diar_source, speakers)) => {
                    eprintln!("[worker] ✓ Done in {:.1}s (diar: {})", elapsed, diar_source);
                    let resp = Response {
                        status: "ok".to_string(),
                        error: None,
                        elapsed_s: Some(elapsed),
                        diar_source: Some(diar_source),
                        speakers: Some(speakers),
                    };
                    println!("{}", serde_json::to_string(&resp)?);
                }
                Err(e) => {
                    eprintln!("[worker] Error: {}", e);
                    let resp = Response {
                        status: "error".to_string(),
                        error: Some(e.to_string()),
                        elapsed_s: None, diar_source: None, speakers: None,
                    };
                    println!("{}", serde_json::to_string(&resp)?);
                }
            }
            std::io::stdout().flush()?;
        }

        eprintln!("[worker] Shutdown.");
        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
