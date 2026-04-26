/**
 * End-to-end flow test: upload → identify → list → download → named transcript
 * Run from project root: bun run server/test-flow.ts
 */

const BASE = "http://localhost:3000";
const MP3  = "C:/Users/Albert/Downloads/Telegram Desktop/Archive 2/meeting01.mp3";
const SPEAKERS_JSON = { "0": "Albert", "1": "Bas", "2": "Randy", "3": "Martin" };

function section(title: string) {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`  ${title}`);
  console.log("═".repeat(60));
}

function elapsed(ms: number) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(2)}s`;
}

// ── Step 1: health check ──────────────────────────────────────────────────────
section("STEP 1: health check");
const health = await fetch(`${BASE}/health`).then(r => r.json());
console.log(health);
if (health.status !== "ok") { console.error("Worker not ready — start server first"); process.exit(1); }

// ── Step 2: upload MP3, first transcription ───────────────────────────────────
section("STEP 2: upload MP3 — first transcription (live)");
const mp3File = Bun.file(MP3);
const form2 = new FormData();
form2.append("audio", mp3File, "meeting01.mp3");

const t2 = Date.now();
const resp2 = await fetch(`${BASE}/transcribe`, { method: "POST", body: form2 }).then(r => r.json());
console.log(`Wall time: ${elapsed(Date.now() - t2)}`);
console.log({
  status:            resp2.status,
  elapsed_s:         resp2.elapsed_s,
  diar_source:       resp2.diar_source,
  transcript_source: resp2.transcript_source,
  speakers:          resp2.speakers,
});
console.log("\n--- transcript (first 15 lines) ---");
resp2.transcript?.split("\n").slice(0, 15).forEach((l: string) => console.log(l));

const HASH: string = resp2.audio_hash ?? (() => {
  // derive from known value since server doesn't echo it back yet
  return "6c461802c15d21780af727e9d1065d4436fc4852999905debb69323760ea955e";
})();

// ── Step 3: identify — extract speaker clips ──────────────────────────────────
section("STEP 3: identify speakers — extract 5s clips");
const form3 = new FormData();
form3.append("audio", mp3File, "meeting01.mp3");
form3.append("identify", "true");

const t3 = Date.now();
const resp3 = await fetch(`${BASE}/transcribe`, { method: "POST", body: form3 }).then(r => r.json());
console.log(`Wall time: ${elapsed(Date.now() - t3)}`);
console.log({ status: resp3.status, elapsed_s: resp3.elapsed_s, speakers: resp3.speakers });
console.log("\nspeaker_clips:");
for (const [id, path] of Object.entries(resp3.speaker_clips ?? {})) {
  console.log(`  Speaker ${id} → ${path}`);
}

// ── Step 4: list clips via API ────────────────────────────────────────────────
section(`STEP 4: list clips — GET /speakers/${HASH.slice(0, 16)}…`);
const list = await fetch(`${BASE}/speakers/${HASH}`).then(r => r.json());
console.log(list.clips?.map((c: { id: number; url: string }) => `  Speaker ${c.id} → ${BASE}${c.url}`).join("\n"));

// ── Step 5: download clips ────────────────────────────────────────────────────
section("STEP 5: download speaker clips");
for (const clip of list.clips ?? []) {
  const t = Date.now();
  const wav = await fetch(`${BASE}${clip.url}`);
  const buf = await wav.arrayBuffer();
  const kb = Math.round(buf.byteLength / 1024);
  const outPath = `C:/Temp/speaker_${clip.id}.wav`;
  await Bun.write(outPath, buf);
  console.log(`  speaker_${clip.id}.wav — ${kb}KB  (${elapsed(Date.now() - t)}) → ${outPath}`);
}

// ── Step 6: re-transcribe with names (instant cache hit) ──────────────────────
section("STEP 6: re-transcribe with real names — should be instant cache hit");
const t6 = Date.now();
const resp6 = await fetch(`${BASE}/transcribe`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    audio:    MP3,
    speakers: await (async () => {
      const tmp = "C:/Temp/speakers.json";
      await Bun.write(tmp, JSON.stringify(SPEAKERS_JSON));
      return tmp;
    })(),
  }),
}).then(r => r.json());
console.log(`Wall time: ${elapsed(Date.now() - t6)}`);
console.log({ status: resp6.status, transcript_source: resp6.transcript_source, elapsed_s: resp6.elapsed_s });
console.log("\n--- named transcript (first 20 lines) ---");
resp6.transcript?.split("\n").slice(0, 20).forEach((l: string) => console.log(l));
