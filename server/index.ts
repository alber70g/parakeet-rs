import { spawn, type Subprocess } from "bun";
import { join } from "path";
import { createHash } from "crypto";
import { mkdir, stat } from "fs/promises";

// ── Config ─────────────────────────────────────────────────────────────────────

const PORT = Number(process.env.PORT ?? 3000);
const WORKER_EXE = process.env.WORKER_EXE ?? join(import.meta.dir, "../target/release/examples/meeting_worker.exe");
const CACHE_DIR = process.env.CACHE_DIR ?? join(import.meta.dir, "../.diar_cache");
const SAMPLES_DIR = process.env.SAMPLES_DIR ?? join(import.meta.dir, "../speaker_samples");
const USE_CUDA = process.env.NO_CUDA !== "1";
const WORKER_TIMEOUT_MS = Number(process.env.WORKER_TIMEOUT_MS ?? 300_000);
const FFMPEG = process.env.FFMPEG ?? "ffmpeg";

// CUDA libs from PyTorch uv cache — needed for ORT CUDA EP
const ORT_DYLIB_PATH = process.env.ORT_DYLIB_PATH ??
  `C:\\Python312\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime.dll`;

const CUDA_LIBS = process.env.CUDA_LIBS ??
  `C:\\Users\\${process.env.USERNAME}\\AppData\\Local\\uv\\cache\\archive-v0\\hdnzU4gUhSdieW1M98Xzu\\torch\\lib`;

// ── ffmpeg conversion + WAV cache ─────────────────────────────────────────────

const WAV_CACHE_DIR = join(CACHE_DIR, "wav");
const TRANSCRIPT_CACHE_DIR = join(CACHE_DIR, "transcript");

async function fileHash(path: string): Promise<string> {
  const file = Bun.file(path);
  const buf = await file.arrayBuffer();
  return createHash("sha256").update(Buffer.from(buf)).digest("hex");
}

async function bytesHash(buf: ArrayBuffer): Promise<string> {
  return createHash("sha256").update(Buffer.from(buf)).digest("hex");
}

function namesPath(hash: string): string {
  return join(SAMPLES_DIR, hash, "names.json");
}

async function readNames(hash: string): Promise<Record<string, string> | null> {
  try {
    const text = await Bun.file(namesPath(hash)).text();
    return JSON.parse(text) as Record<string, string>;
  } catch {
    return null;
  }
}

async function writeNames(hash: string, names: Record<string, string>): Promise<void> {
  await mkdir(join(SAMPLES_DIR, hash), { recursive: true });
  await Bun.write(namesPath(hash), JSON.stringify(names, null, 2));
}

async function readTranscriptCache(hash: string): Promise<Record<string, unknown> | null> {
  const path = join(TRANSCRIPT_CACHE_DIR, `${hash}.json`);
  try {
    const text = await Bun.file(path).text();
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function writeTranscriptCache(hash: string, data: Record<string, unknown>): Promise<void> {
  await mkdir(TRANSCRIPT_CACHE_DIR, { recursive: true });
  await Bun.write(join(TRANSCRIPT_CACHE_DIR, `${hash}.json`), JSON.stringify(data));
}

/** Convert any audio file to 16kHz mono float32 WAV via ffmpeg.
 *  Returns the path to a cached WAV — same input always yields the same path. */
async function ensureWav(sourcePath: string, hash: string): Promise<string> {
  await mkdir(WAV_CACHE_DIR, { recursive: true });
  const dest = join(WAV_CACHE_DIR, `${hash}.wav`);

  // Already converted
  try { await stat(dest); return dest; } catch {}

  const tmp = join(WAV_CACHE_DIR, `${hash}.tmp.wav`);
  const proc = spawn({
    cmd: [FFMPEG, "-y", "-i", sourcePath, "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le", tmp],
    stdout: "pipe",
    stderr: "pipe",
  });

  const exit = await proc.exited;
  if (exit !== 0) {
    const errText = await new Response(proc.stderr as ReadableStream).text();
    throw new Error(`ffmpeg failed (exit ${exit}): ${errText.slice(-500)}`);
  }

  // Atomic rename so a partial file is never used
  await Bun.file(tmp).exists(); // flush
  const fs = await import("fs/promises");
  await fs.rename(tmp, dest);

  return dest;
}

// ── Worker process wrapper ─────────────────────────────────────────────────────

type PendingReq = {
  resolve: (data: { json: string; transcript: string[] }) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

class WorkerProcess {
  private proc: Subprocess | null = null;
  private queue: PendingReq[] = [];
  private ready = false;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((e: Error) => void) | null = null;
  private lineBuffer = "";
  private transcriptBuffer: string[] = [];

  private get workerEnv() {
    return {
      ...process.env,
      ORT_DYLIB_PATH,
      PATH: `${CUDA_LIBS};${process.env.PATH ?? ""}`,
    };
  }

  async start(): Promise<void> {
    const args = [WORKER_EXE];
    if (USE_CUDA) args.push("--cuda");
    args.push("--cache-dir", CACHE_DIR);

    console.log(`[worker] spawning: ${args.join(" ")}`);

    this.proc = spawn({
      cmd: args,
      env: this.workerEnv,
      stdin: "pipe",
      stdout: "pipe",
      stderr: "inherit",
    });

    this.pumpStdout();

    await new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
      setTimeout(() => reject(new Error("Worker startup timeout (30s)")), 30_000);
    });
  }

  private async pumpStdout() {
    const stdout = this.proc?.stdout;
    if (!stdout || typeof stdout === "number") return;
    const reader = (stdout as ReadableStream<Uint8Array>).getReader();
    const dec = new TextDecoder();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) { this.onWorkerExit(); break; }
        this.lineBuffer += dec.decode(value, { stream: true });
        let nl: number;
        while ((nl = this.lineBuffer.indexOf("\n")) !== -1) {
          const line = this.lineBuffer.slice(0, nl).trimEnd();
          this.lineBuffer = this.lineBuffer.slice(nl + 1);
          this.onLine(line);
        }
      }
    } catch {
      this.onWorkerExit();
    }
  }

  private onLine(line: string) {
    if (!line) return;

    // Try to parse as JSON
    let msg: Record<string, unknown> | null = null;
    try { msg = JSON.parse(line); } catch { /* not JSON */ }

    if (!this.ready) {
      if (msg?.status === "ready") {
        this.ready = true;
        console.log(`[worker] ready (load=${msg.load_s}s)`);
        this.readyResolve?.();
      } else if (msg?.status === "error") {
        this.readyReject?.(new Error(String(msg.error ?? "Worker startup error")));
      }
      return;
    }

    if (msg === null) {
      // Plain-text transcript line — buffer until the JSON response arrives
      this.transcriptBuffer.push(line);
      return;
    }

    // JSON response — deliver to waiting request along with buffered transcript
    const pending = this.queue.shift();
    if (pending) {
      clearTimeout(pending.timer);
      pending.resolve({ json: line, transcript: this.transcriptBuffer.splice(0) });
    } else {
      console.warn("[worker] unexpected JSON (no pending request):", line);
      this.transcriptBuffer = [];
    }
  }

  private onWorkerExit() {
    console.error("[worker] process exited — draining queue");
    this.ready = false;
    for (const p of this.queue) {
      clearTimeout(p.timer);
      p.reject(new Error("Worker process exited"));
    }
    this.queue = [];
    setTimeout(() => { console.log("[worker] restarting..."); this.start().catch(console.error); }, 1000);
  }

  send(req: object): Promise<{ json: string; transcript: string[] }> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const idx = this.queue.findIndex(q => q.resolve === resolve);
        if (idx !== -1) this.queue.splice(idx, 1);
        reject(new Error(`Worker timeout after ${WORKER_TIMEOUT_MS}ms`));
      }, WORKER_TIMEOUT_MS);

      this.queue.push({ resolve, reject, timer });
      const line = JSON.stringify(req) + "\n";
      const stdin = this.proc?.stdin;
      if (stdin && typeof stdin !== "number") (stdin as import("bun").FileSink).write(line);
    });
  }

  isReady() { return this.ready; }

  async stop() {
    if (this.proc) {
      try {
        const stdin = this.proc.stdin;
        if (stdin && typeof stdin !== "number") (stdin as import("bun").FileSink).write('{"quit":true}\n');
      } catch {}
      this.proc.kill();
      this.proc = null;
    }
  }
}

// ── Global worker instance ─────────────────────────────────────────────────────

const worker = new WorkerProcess();

// ── Request helpers ────────────────────────────────────────────────────────────

async function handleTranscribe(req: Request): Promise<Response> {
  const ct = req.headers.get("content-type") ?? "";

  if (ct.includes("multipart/form-data")) {
    return handleUpload(req);
  }

  // JSON body: { audio: "/abs/path", cache_dir?, samples_dir?, speakers?, output?, identify? }
  let body: Record<string, unknown>;
  try { body = await req.json() as Record<string, unknown>; } catch {
    return jsonError("Invalid JSON body", 400);
  }

  if (typeof body.audio !== "string") return jsonError('"audio" (string path) required', 400);

  if (!worker.isReady()) return jsonError("Worker not ready yet", 503);

  let wavPath: string;
  let hash: string;
  try {
    hash = await fileHash(body.audio);
    wavPath = await ensureWav(body.audio, hash);
    console.log(`[ffmpeg] ${body.audio} → ${wavPath}`);
  } catch (e) {
    return jsonError(`Audio conversion failed: ${(e as Error).message}`, 422);
  }

  const workerReq: Record<string, unknown> = {
    audio: wavPath,
    audio_hash: hash,
    samples_dir: body.samples_dir ?? SAMPLES_DIR,
    cache_dir: body.cache_dir ?? CACHE_DIR,
  };
  if (body.speakers) workerReq.speakers = body.speakers;
  if (body.output)   workerReq.output   = body.output;
  if (body.identify) workerReq.identify  = body.identify;

  return runWorker(workerReq, hash);
}

async function handleUpload(req: Request): Promise<Response> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let form: any;
  try { form = await req.formData(); } catch {
    return jsonError("Failed to parse multipart form", 400);
  }

  const file = form.get("audio");
  if (!file) return jsonError('"audio" file field required', 400);

  if (!worker.isReady()) return jsonError("Worker not ready yet", 503);

  // Accept File/Blob (binary upload) or a string path
  let buf: ArrayBuffer;
  let filename: string;
  if (typeof file === "string") {
    // Client sent a path string in a form field — treat as local path
    buf = await Bun.file(file).arrayBuffer();
    filename = file.split(/[\\/]/).pop() ?? "audio";
  } else {
    buf = await (file as File).arrayBuffer();
    filename = (file as File).name || "audio";
  }
  const hash = await bytesHash(buf);

  // Write original upload to a stable path (so ffmpeg can read it by format/extension)
  await mkdir(WAV_CACHE_DIR, { recursive: true });
  const ext = (filename.match(/\.[^.]+$/) ?? [".bin"])[0].toLowerCase();
  const uploadPath = join(WAV_CACHE_DIR, `${hash}${ext}`);
  const uploadExists = await Bun.file(uploadPath).exists();
  if (!uploadExists) await Bun.write(uploadPath, buf);

  let wavPath: string;
  try {
    wavPath = await ensureWav(uploadPath, hash);
    console.log(`[ffmpeg] upload(${filename}) → ${wavPath}`);
  } catch (e) {
    return jsonError(`Audio conversion failed: ${(e as Error).message}`, 422);
  }

  const workerReq: Record<string, unknown> = {
    audio: wavPath,
    audio_hash: hash,
    samples_dir: SAMPLES_DIR,
    cache_dir: CACHE_DIR,
  };

  const cacheDir = form.get("cache_dir");
  if (typeof cacheDir === "string") workerReq.cache_dir = cacheDir;

  const identify = form.get("identify");
  if (identify === "true" || identify === "1") workerReq.identify = true;

  return runWorker(workerReq, hash);
}

async function runWorker(workerReq: Record<string, unknown>, hash: string): Promise<Response> {
  const isIdentify = workerReq.identify === true;
  let hasSpeakers = typeof workerReq.speakers === "string";

  // Auto-inject saved names when caller didn't supply their own speakers file
  if (!isIdentify && !hasSpeakers) {
    const savedNames = await readNames(hash);
    if (savedNames) {
      const tmpPath = join(SAMPLES_DIR, hash, "names_tmp.json");
      await Bun.write(tmpPath, JSON.stringify(savedNames));
      workerReq.speakers = tmpPath;
      hasSpeakers = true;
      console.log(`[names] auto-injecting saved names for ${hash.slice(0, 16)}…`);
    }
  }

  // Transcript cache only applies to plain transcription (no identify, no custom speaker names)
  const useCache = !isIdentify && !hasSpeakers;
  if (useCache) {
    const cached = await readTranscriptCache(hash);
    if (cached) {
      cached.transcript_source = "cache";
      cached.audio_hash = hash;
      console.log(`[transcript] cache hit for ${hash.slice(0, 16)}…`);
      return new Response(JSON.stringify(cached), { status: 200, headers: { "content-type": "application/json" } });
    }
  }

  let result: { json: string; transcript: string[] };
  try {
    result = await worker.send(workerReq);
  } catch (e) {
    return jsonError((e as Error).message, 504);
  }

  let parsed: Record<string, unknown>;
  try { parsed = JSON.parse(result.json); } catch {
    return jsonError("Worker returned non-JSON: " + result.json, 502);
  }

  if (parsed.status === "error") {
    return new Response(JSON.stringify(parsed), { status: 500, headers: { "content-type": "application/json" } });
  }

  parsed.transcript = result.transcript.join("\n");
  parsed.transcript_source = "live";
  parsed.audio_hash = hash;

  // Only cache plain anonymous transcripts — named/identify runs are not cached
  if (useCache) {
    writeTranscriptCache(hash, parsed).catch(e => console.error("[transcript] cache write failed:", e));
  }

  return new Response(JSON.stringify(parsed), { status: 200, headers: { "content-type": "application/json" } });
}

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ status: "error", error: msg }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// ── HTTP server ────────────────────────────────────────────────────────────────

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/health" && req.method === "GET") {
      return new Response(
        JSON.stringify({ status: worker.isReady() ? "ok" : "starting", worker: WORKER_EXE }),
        { headers: { "content-type": "application/json" } }
      );
    }

    if (url.pathname === "/transcribe" && req.method === "POST") {
      return handleTranscribe(req);
    }

    // GET /speakers/:hash/:id  — download a speaker clip WAV
    const speakerMatch = url.pathname.match(/^\/speakers\/([0-9a-f]{64})\/(\d+)$/);
    if (speakerMatch && req.method === "GET") {
      const [, hash, id] = speakerMatch;
      const clipPath = join(SAMPLES_DIR, hash, `speaker_${id}.wav`);
      const file = Bun.file(clipPath);
      if (!(await file.exists())) {
        return new Response(
          JSON.stringify({ status: "error", error: `Speaker clip not found. Run POST /transcribe with identify=true first.` }),
          { status: 404, headers: { "content-type": "application/json" } }
        );
      }
      return new Response(file, {
        headers: {
          "content-type": "audio/wav",
          "content-disposition": `attachment; filename="speaker_${id}.wav"`,
        },
      });
    }

    // PUT /speakers/:hash/names  — save speaker id→name mapping
    const speakerNamesMatch = url.pathname.match(/^\/speakers\/([0-9a-f]{64})\/names$/);
    if (speakerNamesMatch && req.method === "PUT") {
      const [, hash] = speakerNamesMatch;
      let names: Record<string, string>;
      try { names = await req.json() as Record<string, string>; } catch {
        return jsonError("Invalid JSON body", 400);
      }
      if (typeof names !== "object" || Array.isArray(names)) return jsonError("Body must be an object mapping speaker id to name", 400);
      await writeNames(hash, names);
      console.log(`[names] saved ${Object.keys(names).length} name(s) for ${hash.slice(0, 16)}…`);
      return new Response(JSON.stringify({ status: "ok", hash, names }), { headers: { "content-type": "application/json" } });
    }

    // GET /speakers/:hash/names  — read saved names
    if (speakerNamesMatch && req.method === "GET") {
      const [, hash] = speakerNamesMatch;
      const names = await readNames(hash);
      if (!names) return new Response(JSON.stringify({ status: "error", error: "No names saved for this audio" }), { status: 404, headers: { "content-type": "application/json" } });
      return new Response(JSON.stringify({ hash, names }), { headers: { "content-type": "application/json" } });
    }

    // GET /speakers/:hash  — list available clips for an audio file
    const speakerListMatch = url.pathname.match(/^\/speakers\/([0-9a-f]{64})$/);
    if (speakerListMatch && req.method === "GET") {
      const [, hash] = speakerListMatch;
      const clipDir = join(SAMPLES_DIR, hash);
      try {
        const fs = await import("fs/promises");
        const files = await fs.readdir(clipDir);
        const clips = files
          .filter(f => f.match(/^speaker_\d+\.wav$/))
          .map(f => {
            const id = f.match(/(\d+)/)?.[1];
            return { id: Number(id), url: `/speakers/${hash}/${id}`, file: f };
          })
          .sort((a, b) => a.id - b.id);
        return new Response(JSON.stringify({ hash, clips }), { headers: { "content-type": "application/json" } });
      } catch {
        return new Response(
          JSON.stringify({ status: "error", error: `No speaker clips found for this audio. Run POST /transcribe with identify=true first.` }),
          { status: 404, headers: { "content-type": "application/json" } }
        );
      }
    }

    return new Response(
      JSON.stringify({
        endpoints: {
          "GET /health": "Worker health check",
          "POST /transcribe": "JSON body: { audio, cache_dir?, samples_dir?, speakers?, output?, identify? } or multipart with 'audio' file",
          "GET /speakers/:hash": "List speaker clips for an audio file (requires prior identify=true run)",
          "GET /speakers/:hash/:id": "Download speaker WAV clip",
          "PUT /speakers/:hash/names": "Save speaker names: body {\"0\":\"Alice\",\"1\":\"Bob\"} — auto-applied on future transcriptions",
          "GET /speakers/:hash/names": "Read saved speaker names for an audio file",
        },
      }),
      { status: 404, headers: { "content-type": "application/json" } }
    );
  },
});

console.log(`[server] listening on http://localhost:${PORT}`);
console.log(`[server] worker:  ${WORKER_EXE}`);
console.log(`[server] cache:   ${CACHE_DIR}`);
console.log(`[server] cuda:    ${USE_CUDA}`);
console.log(`[server] ffmpeg:  ${FFMPEG}`);

worker.start().catch(e => {
  console.error("[worker] failed to start:", e.message);
  process.exit(1);
});

process.on("SIGINT",  async () => { await worker.stop(); process.exit(0); });
process.on("SIGTERM", async () => { await worker.stop(); process.exit(0); });
