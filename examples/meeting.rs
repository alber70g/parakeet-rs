/*
Meeting Transcription with Speaker Diarization

Combines Parakeet-TDT (multilingual transcription) and Sortformer v2 (speaker diarization)
into a single workflow for meeting recordings. Handles long audio by chunking TDT input.

MODELS NEEDED
─────────────
TDT:        ./tdt/encoder-model.onnx, encoder-model.onnx.data, decoder_joint-model.onnx, vocab.txt
            https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx

Sortformer: diar_streaming_sortformer_4spk-v2.onnx  (place next to the binary or pass via arg)
            https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx

AUDIO
─────
Input must be a 16kHz mono WAV. Convert with ffmpeg:
    ffmpeg -i recording.m4a -ar 16000 -ac 1 recording.wav

USAGE
─────
# Step 1 – first run: identify speakers from auto-extracted samples
cargo run --release --example meeting --features sortformer -- \
    recording.wav \
    --speakers speakers.json \
    --identify

This will:
  • Run diarization + transcription
  • Save a short audio sample per speaker to ./speaker_samples/speaker_N.wav
  • Print a [start – end] window per speaker so you can listen and identify them
  • Write a template speakers.json for you to fill in real names

# Step 2 – fill in speakers.json (maps speaker ID → real name):
{
  "0": "Alice",
  "1": "Bob",
  "2": "Charlie"
}

# Step 3 – final run with real names
cargo run --release --example meeting --features sortformer -- \
    recording.wav \
    --speakers speakers.json

OUTPUT FORMAT
─────────────
[00:01:03 – 00:01:48] Alice:
  And so the main concern was about the delivery timeline.

[00:01:48 – 00:02:05] Bob:
  Right, I think we can move that to next sprint.

...

Optionally write full transcript to a file with --output transcript.txt

LONG AUDIO NOTE
───────────────
TDT has a ~8-10 minute sequence limit. For longer recordings the audio is automatically
split into 5-minute chunks for TDT, while Sortformer runs on the full file for consistent
speaker IDs. The segments are then merged back together.
*/

#[cfg(feature = "sortformer")]
use hound::{WavSpec, WavWriter};
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
#[cfg(feature = "sortformer")]
use std::collections::HashMap;
#[cfg(feature = "sortformer")]
use std::path::{Path, PathBuf};
#[cfg(feature = "sortformer")]
use std::time::Instant;

// TDT sequence length limit: 5 minutes of audio at 16kHz
#[cfg(feature = "sortformer")]
const TDT_CHUNK_SAMPLES: usize = 16_000 * 60 * 5;
// Minimum audio samples needed to identify a speaker (1 second)
#[cfg(feature = "sortformer")]
const MIN_SPEAKER_SAMPLE_SECS: f32 = 1.0;
// Duration of speaker identification sample to extract (5 seconds)
#[cfg(feature = "sortformer")]
const SPEAKER_SAMPLE_SECS: f32 = 5.0;

#[cfg(feature = "sortformer")]
#[derive(Debug, Clone)]
struct Segment {
    start: f32,
    end: f32,
    speaker_id: usize,
    text: String,
}

#[cfg(feature = "sortformer")]
fn format_time(secs: f32) -> String {
    let total = secs as u32;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{:02}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}", m, s)
    }
}

/// Parse CLI args manually (no external dep needed).
#[cfg(feature = "sortformer")]
struct Args {
    audio_path: String,
    tdt_dir: String,
    sortformer_model: String,
    speakers_file: Option<String>,
    output_file: Option<String>,
    identify_mode: bool,
    samples_dir: String,
}

#[cfg(feature = "sortformer")]
fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 2 {
        return Err(format!(
            "Usage: {} <audio.wav> [--tdt ./tdt] [--model sortformer.onnx] \
             [--speakers speakers.json] [--output transcript.txt] [--identify] \
             [--samples-dir ./speaker_samples]",
            raw[0]
        ));
    }
    let mut args = Args {
        audio_path: raw[1].clone(),
        tdt_dir: "./tdt".to_string(),
        sortformer_model: "diar_streaming_sortformer_4spk-v2.onnx".to_string(),
        speakers_file: None,
        output_file: None,
        identify_mode: false,
        samples_dir: "./speaker_samples".to_string(),
    };
    let mut i = 2;
    while i < raw.len() {
        match raw[i].as_str() {
            "--tdt" => { i += 1; args.tdt_dir = raw[i].clone(); }
            "--model" => { i += 1; args.sortformer_model = raw[i].clone(); }
            "--speakers" => { i += 1; args.speakers_file = Some(raw[i].clone()); }
            "--output" => { i += 1; args.output_file = Some(raw[i].clone()); }
            "--identify" => { args.identify_mode = true; }
            "--samples-dir" => { i += 1; args.samples_dir = raw[i].clone(); }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        i += 1;
    }
    Ok(args)
}

/// Load and normalise a WAV to 16kHz mono f32.
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

/// Downmix to mono if needed; resample is not done here – the WAV should already be 16kHz.
#[cfg(feature = "sortformer")]
fn to_mono(audio: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels == 1 {
        return audio;
    }
    audio
        .chunks(channels as usize)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Transcribe a mono 16kHz audio slice with TDT, returning sentence-level segments.
#[cfg(feature = "sortformer")]
fn transcribe_chunk(
    model: &mut ParakeetTDT,
    chunk: &[f32],
    time_offset: f32,
) -> Result<Vec<(f32, f32, String)>, Box<dyn std::error::Error>> {
    let result = model.transcribe_samples(
        chunk.to_vec(),
        16_000,
        1,
        Some(TimestampMode::Sentences),
    )?;
    Ok(result
        .tokens
        .into_iter()
        .map(|t| (t.start + time_offset, t.end + time_offset, t.text))
        .collect())
}

/// Save a short mono f32 WAV sample for speaker identification.
#[cfg(feature = "sortformer")]
fn save_speaker_sample(
    audio: &[f32],
    start_sample: usize,
    end_sample: usize,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    let end = end_sample.min(audio.len());
    for &s in &audio[start_sample..end] {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Write the full transcript to a text file.
#[cfg(feature = "sortformer")]
fn write_transcript(
    segments: &[Segment],
    speaker_names: &HashMap<usize, String>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let mut prev_speaker: Option<usize> = None;
    for seg in segments {
        let name = speaker_names
            .get(&seg.speaker_id)
            .cloned()
            .unwrap_or_else(|| format!("Speaker {}", seg.speaker_id));
        let same_speaker = prev_speaker == Some(seg.speaker_id);
        if !same_speaker {
            if prev_speaker.is_some() {
                writeln!(file)?;
            }
            writeln!(
                file,
                "[{} – {}] {}:",
                format_time(seg.start),
                format_time(seg.end),
                name
            )?;
        } else {
            // Extend time range in header is already printed; just append text.
        }
        writeln!(file, "  {}", seg.text.trim())?;
        prev_speaker = Some(seg.speaker_id);
    }
    Ok(())
}

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("This example requires the 'sortformer' feature.");
        eprintln!("Run with: cargo run --release --example meeting --features sortformer -- <audio.wav>");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let args = match parse_args() {
            Ok(a) => a,
            Err(e) => { eprintln!("{}", e); std::process::exit(1); }
        };

        let total_start = Instant::now();

        // ── 1. Load audio ────────────────────────────────────────────────────────
        println!("=== Meeting Transcription + Diarization ===\n");
        println!("[1/4] Loading audio: {}", args.audio_path);

        let (raw_audio, sample_rate, channels) = load_wav(&args.audio_path)?;
        let audio = to_mono(raw_audio, channels);

        if sample_rate != 16_000 {
            return Err(format!(
                "Audio must be 16kHz. Got {}Hz. Convert with:\n  ffmpeg -i {} -ar 16000 -ac 1 output.wav",
                sample_rate, args.audio_path
            ).into());
        }

        let duration = audio.len() as f32 / 16_000.0;
        println!(
            "    {}Hz, {} ch → mono, {:.1}s ({:.1} min)\n",
            sample_rate,
            channels,
            duration,
            duration / 60.0
        );

        // ── 2. Speaker diarization (full audio, streaming) ───────────────────────
        println!("[2/4] Speaker diarization (Sortformer v2)…");
        let diar_start = Instant::now();

        let mut sortformer = Sortformer::with_config(
            &args.sortformer_model,
            None,
            DiarizationConfig::callhome(),
        )?;

        println!(
            "    chunk_len={}, right_context={}, latency={:.2}s",
            sortformer.chunk_len,
            sortformer.right_context,
            sortformer.latency()
        );

        // Feed entire audio at once – Sortformer handles long files natively.
        let speaker_segments = sortformer.diarize(audio.clone(), 16_000, 1)?;

        println!(
            "    {} speaker segments in {:.1}s\n",
            speaker_segments.len(),
            diar_start.elapsed().as_secs_f32()
        );

        // Collect unique speaker IDs sorted for deterministic output.
        let mut unique_speakers: Vec<usize> = {
            let mut ids: Vec<usize> = speaker_segments.iter().map(|s| s.speaker_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        unique_speakers.sort_unstable();

        // ── 3. Load speaker name mapping (optional) ──────────────────────────────
        let mut speaker_names: HashMap<usize, String> = HashMap::new();

        if let Some(ref sf) = args.speakers_file {
            if std::path::Path::new(sf).exists() {
                let json = std::fs::read_to_string(sf)?;
                let raw: HashMap<String, String> = serde_json::from_str(&json)?;
                for (k, v) in raw {
                    if let Ok(id) = k.parse::<usize>() {
                        speaker_names.insert(id, v);
                    }
                }
                println!(
                    "[3/4] Speaker names loaded from {}: {:?}\n",
                    sf, speaker_names
                );
            } else {
                println!("[3/4] speakers.json not found – will use Speaker N labels\n");
            }
        } else {
            println!("[3/4] No --speakers file given – will use Speaker N labels\n");
        }

        // ── 3b. Identify mode: extract samples + print identification windows ────
        if args.identify_mode {
            println!("=== Speaker Identification Mode ===");
            println!("Extracting a sample clip per speaker for manual identification.\n");

            // For each speaker find the first long-enough continuous segment.
            for &spk in &unique_speakers {
                let name = speaker_names
                    .get(&spk)
                    .cloned()
                    .unwrap_or_else(|| format!("Speaker {}", spk));

                // Find best segment: longest segment ≥ MIN_SPEAKER_SAMPLE_SECS
                let best = speaker_segments
                    .iter()
                    .filter(|s| s.speaker_id == spk)
                    .max_by_key(|s| s.end - s.start);

                if let Some(seg) = best {
                    let seg_start_s = seg.start as f32 / 16_000.0;
                    let seg_end_s = seg.end as f32 / 16_000.0;
                    let seg_len = seg_end_s - seg_start_s;

                    if seg_len < MIN_SPEAKER_SAMPLE_SECS {
                        println!(
                            "  Speaker {} ({}): longest segment only {:.1}s – skipping sample",
                            spk, name, seg_len
                        );
                        continue;
                    }

                    // Extract up to SPEAKER_SAMPLE_SECS
                    let sample_end_s = (seg_start_s + SPEAKER_SAMPLE_SECS).min(seg_end_s);
                    let start_idx = seg.start as usize;
                    let end_idx = (sample_end_s * 16_000.0) as usize;

                    let sample_path = PathBuf::from(&args.samples_dir)
                        .join(format!("speaker_{}.wav", spk));
                    save_speaker_sample(&audio, start_idx, end_idx, &sample_path)?;

                    println!(
                        "  Speaker {} ({}):  [{} – {}]  → {}",
                        spk,
                        name,
                        format_time(seg_start_s),
                        format_time(sample_end_s),
                        sample_path.display()
                    );
                }
            }

            // Write template speakers.json if it doesn't exist yet.
            if let Some(ref sf) = args.speakers_file {
                if !std::path::Path::new(sf).exists() {
                    let template: HashMap<String, String> = unique_speakers
                        .iter()
                        .map(|id| (id.to_string(), format!("Speaker {}", id)))
                        .collect();
                    let json = serde_json::to_string_pretty(&template)?;
                    std::fs::write(sf, &json)?;
                    println!(
                        "\nTemplate speakers.json written to {}",
                        sf
                    );
                    println!("Edit it to assign real names, then re-run without --identify.");
                }
            } else {
                println!("\nTip: re-run with --speakers speakers.json to save/load name mappings.");
            }

            println!(
                "\nTotal time: {:.1}s",
                total_start.elapsed().as_secs_f32()
            );
            return Ok(());
        }

        // ── 4. Transcription with TDT (chunked for long audio) ──────────────────
        println!("[4/4] Transcribing with Parakeet-TDT (chunk size: 5 min)…");
        let tdt_start = Instant::now();

        let mut tdt = ParakeetTDT::from_pretrained(&args.tdt_dir, None)?;

        let mut timed_sentences: Vec<(f32, f32, String)> = Vec::new();
        let total_samples = audio.len();
        let mut offset = 0usize;

        while offset < total_samples {
            let end = (offset + TDT_CHUNK_SAMPLES).min(total_samples);
            let chunk = &audio[offset..end];
            let time_offset = offset as f32 / 16_000.0;

            let chunk_duration = chunk.len() as f32 / 16_000.0;
            println!(
                "    chunk [{} – {}] ({:.1}s)…",
                format_time(time_offset),
                format_time(time_offset + chunk_duration),
                chunk_duration
            );

            let mut chunk_sentences = transcribe_chunk(&mut tdt, chunk, time_offset)?;
            timed_sentences.append(&mut chunk_sentences);
            offset = end;
        }

        println!(
            "    {} sentences in {:.1}s\n",
            timed_sentences.len(),
            tdt_start.elapsed().as_secs_f32()
        );

        // ── 5. Merge transcript + diarization ────────────────────────────────────
        // For each TDT sentence find the Sortformer speaker with maximum overlap.
        let mut segments: Vec<Segment> = timed_sentences
            .into_iter()
            .filter(|(_, _, text)| !text.trim().is_empty())
            .map(|(start, end, text)| {
                let speaker_id = speaker_segments
                    .iter()
                    .filter_map(|s| {
                        let s_start = s.start as f32 / 16_000.0;
                        let s_end = s.end as f32 / 16_000.0;
                        let overlap = (end.min(s_end) - start.max(s_start)).max(0.0);
                        if overlap > 0.0 { Some((s.speaker_id, overlap)) } else { None }
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(id, _)| id)
                    .unwrap_or(usize::MAX); // MAX = UNKNOWN
                Segment { start, end, speaker_id, text }
            })
            .collect();

        // Merge consecutive segments from the same speaker so the output is
        // grouped into readable blocks (one header per continuous speaker turn).
        let mut merged: Vec<Segment> = Vec::new();
        for seg in segments.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.speaker_id == seg.speaker_id && seg.start - last.end < 2.0 {
                    last.end = seg.end;
                    last.text.push(' ');
                    last.text.push_str(seg.text.trim());
                    continue;
                }
            }
            merged.push(seg);
        }

        // ── 6. Print transcript ──────────────────────────────────────────────────
        println!("=== Transcript ===\n");

        for seg in &merged {
            let name = if seg.speaker_id == usize::MAX {
                "UNKNOWN".to_string()
            } else {
                speaker_names
                    .get(&seg.speaker_id)
                    .cloned()
                    .unwrap_or_else(|| format!("Speaker {}", seg.speaker_id))
            };

            println!(
                "[{} – {}] {}:",
                format_time(seg.start),
                format_time(seg.end),
                name
            );
            println!("  {}\n", seg.text.trim());
        }

        // ── 7. Optional file output ───────────────────────────────────────────────
        if let Some(ref out) = args.output_file {
            write_transcript(&merged, &speaker_names, out)?;
            println!("Transcript saved to {}", out);
        }

        // ── 8. Speaker identification hint (if names not fully assigned) ─────────
        let unnamed: Vec<usize> = unique_speakers
            .iter()
            .filter(|id| !speaker_names.contains_key(id))
            .cloned()
            .collect();

        if !unnamed.is_empty() {
            println!("─────────────────────────────────────────────────────────────────────");
            println!("Some speakers have no name assigned. To identify them:");
            println!("  1. Re-run with --identify to extract audio samples per speaker.");
            println!("  2. Listen to ./speaker_samples/speaker_N.wav for each speaker.");
            println!("  3. Fill in speakers.json with real names.");
            println!("  4. Re-run without --identify to get the named transcript.");
            println!(
                "\nUnnamed speaker IDs: {:?}",
                unnamed
            );
        }

        println!(
            "\n✓ Done in {:.1}s",
            total_start.elapsed().as_secs_f32()
        );

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
