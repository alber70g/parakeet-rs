/*
Meeting Worker — Persistent Model Server

Loads Sortformer + TDT once, keeps them in GPU memory across invocations.
Reads JSON requests from stdin, prints JSON responses to stdout.

Eliminates the ~3-4s model load+JIT cost paid on every standalone `meeting` invocation.
With diarization cache, repeat runs drop to ~2.4s (pure TDT inference).

USAGE
─────
# Build
cargo build --release --example meeting_worker --features "sortformer load-dynamic cuda"

# Start
./target/release/examples/meeting_worker.exe --cuda [--cache-dir ./.diar_cache]

# Requests (one JSON per line on stdin):
{"audio":"recording.wav","samples_dir":"./speaker_samples","cache_dir":".diar_cache"}
{"audio":"same.wav","samples_dir":"./speaker_samples","cache_dir":".diar_cache"}  <- cached!
{"quit":true}

TIMING
──────
- Model load (once at startup): ~5.5s
- First run, live diar, parallel diar+TDT: ~4s (GPU, both models warm)
- Subsequent runs, cached diar: ~2.4s (TDT only)

DIARIZATION CACHE
─────────────────
<cache_dir>/<fnv64_of_file>.diar.json — segments as JSON, keyed by FNV-64 of file bytes.
Speaker audio clips: <cache_dir>/speakers/<hash>_<spk_id>.emb (raw f32, 5s clip per speaker).
*/

#[cfg(feature = "sortformer")]
use hound::{WavSpec, WavWriter};
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, SpeakerSegment, Sortformer};
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

// ── Cached segment type ───────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct CachedSegment { start: u64, end: u64, speaker_id: usize }

#[cfg(feature = "sortformer")]
impl From<&SpeakerSegment> for CachedSegment {
    fn from(s: &SpeakerSegment) -> Self { Self { start: s.start, end: s.end, speaker_id: s.speaker_id } }
}
#[cfg(feature = "sortformer")]
impl From<&CachedSegment> for SpeakerSegment {
    fn from(s: &CachedSegment) -> Self { SpeakerSegment { start: s.start, end: s.end, speaker_id: s.speaker_id } }
}

// ── Request / Response ────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
#[derive(serde::Deserialize, Default)]
struct Request {
    audio: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")] error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] elapsed_s: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")] diar_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] speakers: Option<Vec<usize>>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
fn load_wav(path: &str) -> Result<(Vec<f32>, u32, u16), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader.samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok((audio, spec.sample_rate, spec.channels))
}

#[cfg(feature = "sortformer")]
fn to_mono(audio: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels == 1 { return audio; }
    audio.chunks(channels as usize).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
}

/// Cache key from file metadata (mtime + size). Fast: no file read needed.
/// Reliable for meeting WAV files which don't get silently mutated.
#[cfg(feature = "sortformer")]
fn file_hash(path: &str) -> std::io::Result<String> {
    let meta = std::fs::metadata(path)?;
    let size = meta.len();
    let mtime = meta.modified()
        .map(|t| t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0))
        .unwrap_or(0);
    Ok(format!("{:016x}{:032x}", size, mtime))
}

#[cfg(feature = "sortformer")]
fn format_time(secs: f32) -> String {
    let t = secs as u32;
    let (h, m, s) = (t / 3600, (t % 3600) / 60, t % 60);
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

#[cfg(feature = "sortformer")]
fn save_sample(audio: &[f32], start: usize, end: usize, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
    let spec = WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 32,
                         sample_format: hound::SampleFormat::Float };
    let mut w = WavWriter::create(path, spec)?;
    for &s in &audio[start..end.min(audio.len())] { w.write_sample(s)?; }
    w.finalize()?;
    Ok(())
}

/// Save raw f32 clip as .emb (4-byte count header + f32 values).
#[cfg(feature = "sortformer")]
fn save_emb(data: &[f32], path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(data.len() as u32).to_le_bytes())?;
    for &v in data { f.write_all(&v.to_le_bytes())?; }
    Ok(())
}

/// Save per-speaker 5s audio clips for cross-file identity transfer.
#[cfg(feature = "sortformer")]
fn save_speaker_clips(segs: &[SpeakerSegment], audio: &[f32], cache_dir: &str, hash: &str) {
    let emb_dir = PathBuf::from(cache_dir).join("speakers");
    let _ = std::fs::create_dir_all(&emb_dir);
    let mut by_spk: HashMap<usize, Vec<&SpeakerSegment>> = HashMap::new();
    for s in segs { by_spk.entry(s.speaker_id).or_default().push(s); }
    for (&spk_id, spk_segs) in &by_spk {
        if let Some(best) = spk_segs.iter().max_by_key(|s| s.end - s.start) {
            let start = best.start as usize;
            let end = ((best.start as f32 / 16_000.0 + 5.0) * 16_000.0) as usize;
            let end = end.min(best.end as usize).min(audio.len());
            if end > start + 16_000 {
                let path = emb_dir.join(format!("{}_{}.emb", hash, spk_id));
                let _ = save_emb(&audio[start..end], &path);
            }
        }
    }
}

// ── Execution configs ─────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
fn tdt_config(use_cuda: bool) -> ExecutionConfig {
    #[cfg(feature = "cuda")]
    if use_cuda { return ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda).with_intra_threads(1); }
    let _ = use_cuda;
    ExecutionConfig::new().with_intra_threads(4).with_inter_threads(0)
}

#[cfg(feature = "sortformer")]
fn sortformer_config(use_cuda: bool) -> ExecutionConfig {
    #[cfg(feature = "cuda")]
    if use_cuda { return ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda).with_intra_threads(1); }
    let _ = use_cuda;
    ExecutionConfig::new().with_intra_threads(2).with_inter_threads(0)
}

// ── Worker state ──────────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
struct Worker {
    sortformer: Option<Sortformer>,
    tdt: Option<ParakeetTDT>,
    use_cuda: bool,
    global_cache_dir: Option<String>,
}

#[cfg(feature = "sortformer")]
type ThreadResult<T> = std::result::Result<T, String>;

#[cfg(feature = "sortformer")]
impl Worker {
    fn process(&mut self, req: &Request) -> Result<(f32, String, Vec<usize>), Box<dyn std::error::Error>> {
        let req_start = Instant::now();
        let audio_path = req.audio.as_deref().unwrap();
        let samples_dir = req.samples_dir.as_deref().unwrap_or("./speaker_samples");
        let cache_dir_owned: Option<String> = req.cache_dir.clone()
            .or_else(|| self.global_cache_dir.clone());
        let cache_dir = cache_dir_owned.as_deref();
        let identify = req.identify.unwrap_or(false);

        // Load audio
        let (raw, sr, ch) = load_wav(audio_path)?;
        let audio = to_mono(raw, ch);
        if sr != 16_000 { return Err(format!("Audio must be 16kHz, got {}Hz", sr).into()); }

        // ── Diarization + TDT: cache or parallel ─────────────────────────────
        let (speaker_segments, timed_sentences, diar_source): (Vec<SpeakerSegment>, Vec<(f32,f32,String)>, String) =
        if let Some(cache) = cache_dir {
            let hash = file_hash(audio_path)?;
            let cache_path = PathBuf::from(cache).join(format!("{}.diar.json", hash));
            if cache_path.exists() {
                // Cache hit: diarization free, just run TDT
                if identify {
                    let json = std::fs::read_to_string(&cache_path)?;
                    let cached: Vec<CachedSegment> = serde_json::from_str(&json)?;
                    let segs: Vec<SpeakerSegment> = cached.iter().map(|s| s.into()).collect();
                    return Ok(self.do_identify(&audio, &segs, samples_dir, req.speakers.as_deref(), req_start));
                }
                let json = std::fs::read_to_string(&cache_path)?;
                let cached: Vec<CachedSegment> = serde_json::from_str(&json)?;
                let segs: Vec<SpeakerSegment> = cached.iter().map(|s| s.into()).collect();
                let sents = self.run_tdt_only(&audio)?;
                (segs, sents, "cache".to_string())
            } else {
                // Cache miss — run diar + TDT in parallel, save diar result
                let (segs, sents) = self.run_parallel(&audio, cache_dir, Some((&hash, cache)))?;
                (segs, sents, "live".to_string())
            }
        } else {
            if identify {
                let (segs, _) = self.run_parallel(&audio, None, None)?;
                return Ok(self.do_identify(&audio, &segs, samples_dir, req.speakers.as_deref(), req_start));
            }
            let (segs, sents) = self.run_parallel(&audio, None, None)?;
            (segs, sents, "live".to_string())
        };

        // ── Unique speakers ───────────────────────────────────────────────────
        let mut unique_speakers: Vec<usize> = {
            let mut ids: Vec<usize> = speaker_segments.iter().map(|s| s.speaker_id).collect();
            ids.sort_unstable(); ids.dedup(); ids
        };
        unique_speakers.sort_unstable();

        // ── Speaker names ─────────────────────────────────────────────────────
        let mut speaker_names: HashMap<usize, String> = HashMap::new();
        if let Some(sf) = req.speakers.as_deref() {
            if Path::new(sf).exists() {
                let json = std::fs::read_to_string(sf)?;
                let raw: HashMap<String, String> = serde_json::from_str(&json)?;
                for (k, v) in raw { if let Ok(id) = k.parse::<usize>() { speaker_names.insert(id, v); } }
            }
        }

        // ── Merge + print ─────────────────────────────────────────────────────
        let merged = merge_transcript(timed_sentences, &speaker_segments);
        print_transcript(&merged, &speaker_names, audio_path);
        if let Some(out) = req.output.as_deref() { write_transcript(&merged, &speaker_names, out)?; }

        Ok((req_start.elapsed().as_secs_f32(), diar_source, unique_speakers))
    }

    /// Run diarization + TDT in parallel threads (move models out, get them back).
    fn run_parallel(
        &mut self,
        audio: &[f32],
        cache_dir: Option<&str>,
        cache_save: Option<(&str, &str)>, // (hash, cache_dir)
    ) -> Result<(Vec<SpeakerSegment>, Vec<(f32, f32, String)>), Box<dyn std::error::Error>> {
        let mut sf = self.sortformer.take().ok_or("sortformer not available")?;
        let mut tdt = self.tdt.take().ok_or("tdt not available")?;

        let audio_for_diar = audio.to_vec();
        let audio_for_tdt = audio.to_vec();
        let use_cuda = self.use_cuda;

        // No stagger needed — CUDA already warm from model load at startup.
        let diar_handle = std::thread::spawn(move || -> ThreadResult<(Vec<SpeakerSegment>, Sortformer)> {
            let segs = sf.diarize(audio_for_diar, 16_000, 1).map_err(|e| e.to_string())?;
            Ok((segs, sf))
        });
        let _ = use_cuda;

        let tdt_handle = std::thread::spawn(move || -> ThreadResult<(Vec<(f32,f32,String)>, ParakeetTDT)> {
            let total = audio_for_tdt.len();
            let mut sentences = Vec::new();
            let mut offset = 0;
            while offset < total {
                let end = (offset + TDT_CHUNK_SAMPLES).min(total);
                let chunk = audio_for_tdt[offset..end].to_vec();
                let time_off = offset as f32 / 16_000.0;
                let result = tdt.transcribe_samples(chunk, 16_000, 1, Some(TimestampMode::Sentences))
                    .map_err(|e| e.to_string())?;
                for t in result.tokens { sentences.push((t.start + time_off, t.end + time_off, t.text)); }
                offset = end;
            }
            Ok((sentences, tdt))
        });

        let (segs, sf_back) = diar_handle.join().map_err(|_| "diar thread panicked")?.map_err(|e| e)?;
        let (sentences, tdt_back) = tdt_handle.join().map_err(|_| "tdt thread panicked")?.map_err(|e| e)?;

        self.sortformer = Some(sf_back);
        self.tdt = Some(tdt_back);

        // Cache diarization result
        if let Some((hash, cache)) = cache_save {
            let _ = std::fs::create_dir_all(cache);
            let cache_path = PathBuf::from(cache).join(format!("{}.diar.json", hash));
            let cached: Vec<CachedSegment> = segs.iter().map(|s| s.into()).collect();
            let _ = std::fs::write(&cache_path, serde_json::to_string(&cached).unwrap());
            if let Some(c) = cache_dir { save_speaker_clips(&segs, audio, c, hash); }
        }

        Ok((segs, sentences))
    }

    /// Run TDT only (cached diar path). Models must be available.
    fn run_tdt_only(&mut self, audio: &[f32]) -> Result<Vec<(f32, f32, String)>, Box<dyn std::error::Error>> {
        let tdt = self.tdt.as_mut().ok_or("tdt not available")?;
        let total = audio.len();
        let mut sentences = Vec::new();
        let mut offset = 0;
        while offset < total {
            let end = (offset + TDT_CHUNK_SAMPLES).min(total);
            let chunk = audio[offset..end].to_vec();
            let time_off = offset as f32 / 16_000.0;
            let result = tdt.transcribe_samples(chunk, 16_000, 1, Some(TimestampMode::Sentences))?;
            for t in result.tokens { sentences.push((t.start + time_off, t.end + time_off, t.text)); }
            offset = end;
        }
        Ok(sentences)
    }

    fn do_identify(
        &mut self, audio: &[f32], segs: &[SpeakerSegment],
        samples_dir: &str, speakers_file: Option<&str>, start: Instant,
    ) -> (f32, String, Vec<usize>) {
        let mut speaker_names: HashMap<usize, String> = HashMap::new();
        if let Some(sf) = speakers_file {
            if let Ok(json) = std::fs::read_to_string(sf) {
                if let Ok(raw) = serde_json::from_str::<HashMap<String, String>>(&json) {
                    for (k, v) in raw { if let Ok(id) = k.parse::<usize>() { speaker_names.insert(id, v); } }
                }
            }
        }
        let mut unique: Vec<usize> = { let mut ids: Vec<usize> = segs.iter().map(|s| s.speaker_id).collect(); ids.sort_unstable(); ids.dedup(); ids };
        unique.sort_unstable();
        for &spk in &unique {
            let name = speaker_names.get(&spk).cloned().unwrap_or_else(|| format!("Speaker {}", spk));
            if let Some(best) = segs.iter().filter(|s| s.speaker_id == spk).max_by_key(|s| s.end - s.start) {
                let s_s = best.start as f32 / 16_000.0;
                let s_e = best.end as f32 / 16_000.0;
                let len = s_e - s_s;
                if len < MIN_SPEAKER_SAMPLE_SECS { continue; }
                let end_s = (s_s + SPEAKER_SAMPLE_SECS).min(s_e);
                let p = PathBuf::from(samples_dir).join(format!("speaker_{}.wav", spk));
                let _ = save_sample(audio, best.start as usize, (end_s * 16_000.0) as usize, &p);
                println!("  Speaker {} ({}): [{} – {}] → {}", spk, name, format_time(s_s), format_time(end_s), p.display());
            }
        }
        (start.elapsed().as_secs_f32(), "live".to_string(), unique)
    }
}

// ── Transcript helpers ────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
fn merge_transcript(sentences: Vec<(f32, f32, String)>, segs: &[SpeakerSegment]) -> Vec<(f32, f32, usize, String)> {
    let mut raw: Vec<(f32, f32, usize, String)> = sentences.into_iter()
        .filter(|(_, _, t)| !t.trim().is_empty())
        .map(|(start, end, text)| {
            let spk = segs.iter()
                .filter_map(|s| {
                    let ss = s.start as f32 / 16_000.0;
                    let se = s.end as f32 / 16_000.0;
                    let ov = (end.min(se) - start.max(ss)).max(0.0);
                    if ov > 0.0 { Some((s.speaker_id, ov)) } else { None }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(id, _)| id).unwrap_or(usize::MAX);
            (start, end, spk, text)
        }).collect();
    let mut merged: Vec<(f32, f32, usize, String)> = Vec::new();
    for seg in raw.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.2 == seg.2 && seg.0 - last.1 < 2.0 {
                last.1 = seg.1; last.3.push(' '); last.3.push_str(seg.3.trim()); continue;
            }
        }
        merged.push(seg);
    }
    merged
}

#[cfg(feature = "sortformer")]
fn print_transcript(merged: &[(f32, f32, usize, String)], names: &HashMap<usize, String>, audio_path: &str) {
    println!("=== Transcript: {} ===\n", audio_path);
    for (start, end, spk, text) in merged {
        let name = if *spk == usize::MAX { "UNKNOWN".to_string() } else {
            names.get(spk).cloned().unwrap_or_else(|| format!("Speaker {}", spk))
        };
        println!("[{} – {}] {}:", format_time(*start), format_time(*end), name);
        println!("  {}\n", text.trim());
    }
}

#[cfg(feature = "sortformer")]
fn write_transcript(merged: &[(f32, f32, usize, String)], names: &HashMap<usize, String>, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for (start, end, spk, text) in merged {
        let name = if *spk == usize::MAX { "UNKNOWN".to_string() } else {
            names.get(spk).cloned().unwrap_or_else(|| format!("Speaker {}", spk))
        };
        writeln!(f, "[{} – {}] {}:", format_time(*start), format_time(*end), name)?;
        writeln!(f, "  {}\n", text.trim())?;
    }
    Ok(())
}

// ── Server args ───────────────────────────────────────────────────────────────

#[cfg(feature = "sortformer")]
struct ServerArgs { tdt_dir: String, sortformer_model: String, use_cuda: bool, cache_dir: Option<String> }

#[cfg(feature = "sortformer")]
fn parse_server_args() -> Result<ServerArgs, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut a = ServerArgs {
        tdt_dir: "./tdt".to_string(),
        sortformer_model: "diar_streaming_sortformer_4spk-v2.onnx".to_string(),
        use_cuda: false, cache_dir: None,
    };
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--tdt" => { i += 1; a.tdt_dir = raw[i].clone(); }
            "--model" => { i += 1; a.sortformer_model = raw[i].clone(); }
            "--cuda" => { a.use_cuda = true; }
            "--cache-dir" => { i += 1; a.cache_dir = Some(raw[i].clone()); }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        i += 1;
    }
    Ok(a)
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    { eprintln!("Requires --features sortformer"); return Err("sortformer feature not enabled".into()); }

    #[cfg(feature = "sortformer")]
    {
        let args = parse_server_args().map_err(|e| { eprintln!("{}", e); e })?;
        let startup = Instant::now();
        eprintln!("[worker] Loading models...");

        let mut sf = Sortformer::with_config(
            &args.sortformer_model,
            Some(sortformer_config(args.use_cuda)),
            DiarizationConfig::callhome(),
        )?;
        sf.chunk_len = 80; sf.fifo_len = 80; sf.spkcache_len = 40;

        if args.use_cuda {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        let tdt = ParakeetTDT::from_pretrained(&args.tdt_dir, Some(tdt_config(args.use_cuda)))?;

        eprintln!("[worker] Models loaded in {:.1}s. Ready.", startup.elapsed().as_secs_f32());
        println!("{{\"status\":\"ready\",\"load_s\":{:.2}}}", startup.elapsed().as_secs_f32());
        std::io::stdout().flush()?;

        let mut worker = Worker {
            sortformer: Some(sf),
            tdt: Some(tdt),
            use_cuda: args.use_cuda,
            global_cache_dir: args.cache_dir,
        };

        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let req: Request = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    let resp = Response { status: "error".into(), error: Some(format!("JSON: {}", e)),
                        elapsed_s: None, diar_source: None, speakers: None };
                    println!("{}", serde_json::to_string(&resp)?);
                    std::io::stdout().flush()?;
                    continue;
                }
            };

            if req.quit.unwrap_or(false) { eprintln!("[worker] Quit."); break; }

            if req.audio.is_none() {
                let resp = Response { status: "error".into(), error: Some("Missing 'audio'".into()),
                    elapsed_s: None, diar_source: None, speakers: None };
                println!("{}", serde_json::to_string(&resp)?);
                std::io::stdout().flush()?;
                continue;
            }

            match worker.process(&req) {
                Ok((elapsed, diar_src, speakers)) => {
                    eprintln!("[worker] ✓ Done in {:.1}s (diar: {})", elapsed, diar_src);
                    let resp = Response { status: "ok".into(), error: None,
                        elapsed_s: Some(elapsed), diar_source: Some(diar_src), speakers: Some(speakers) };
                    println!("{}", serde_json::to_string(&resp)?);
                }
                Err(e) => {
                    eprintln!("[worker] Error: {}", e);
                    let resp = Response { status: "error".into(), error: Some(e.to_string()),
                        elapsed_s: None, diar_source: None, speakers: None };
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
