<#
.SYNOPSIS
    Transcribe a meeting recording with speaker diarization.

.DESCRIPTION
    Runs the parakeet-rs meeting pipeline (Sortformer + TDT) on a WAV file.
    Uses CUDA for both models via ORT 1.25 (Python onnxruntime-gpu) and
    cuDNN/cuBLAS from the local PyTorch uv cache.

.PARAMETER Audio
    Path to a 16 kHz mono WAV file. Convert with:
        ffmpeg -i recording.m4a -ar 16000 -ac 1 recording.wav

.PARAMETER SamplesDir
    Directory to write per-speaker WAV snippets. Default: .\speaker_samples

.PARAMETER Speakers
    Path to a speakers.json file mapping speaker IDs to real names.
    Example: { "0": "Alice", "1": "Bob" }

.PARAMETER Output
    Write the transcript to this file in addition to printing it.

.PARAMETER Identify
    Run in identification mode: extract one audio sample per speaker,
    print timestamps, and write a template speakers.json. Does not transcribe.

.PARAMETER NoCuda
    Force CPU inference (no --cuda flag). Much slower but works without a GPU.

.EXAMPLE
    # Step 1 — identify speakers
    .\scripts\transcribe.ps1 recording.wav -Identify -Speakers speakers.json

    # Step 2 — edit speakers.json, then run with names
    .\scripts\transcribe.ps1 recording.wav -Speakers speakers.json -Output transcript.txt
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Audio,

    [string]$SamplesDir = ".\speaker_samples",

    [string]$Speakers,

    [string]$Output,

    [switch]$Identify,

    [switch]$NoCuda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Locate repo root (script lives in <repo>\scripts\) ───────────────────────
$RepoRoot = Split-Path $PSScriptRoot -Parent

# ── Paths ─────────────────────────────────────────────────────────────────────
$OrtDll    = "C:\Python312\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
$CudaLibs  = "C:\Users\$env:USERNAME\AppData\Local\uv\cache\archive-v0\hdnzU4gUhSdieW1M98Xzu\torch\lib"
$CargoBin  = "$env:USERPROFILE\.cargo\bin"

# ── Validation ────────────────────────────────────────────────────────────────
if (-not (Test-Path $Audio)) {
    Write-Error "Audio file not found: $Audio"
    exit 1
}
if (-not (Test-Path $OrtDll)) {
    Write-Error "ORT DLL not found at $OrtDll`nInstall onnxruntime-gpu: pip install onnxruntime-gpu"
    exit 1
}

$UseCuda = -not $NoCuda
if ($UseCuda -and -not (Test-Path $CudaLibs)) {
    Write-Warning "PyTorch CUDA libs not found at $CudaLibs — falling back to CPU"
    $UseCuda = $false
}

# ── Environment ───────────────────────────────────────────────────────────────
$env:ORT_DYLIB_PATH = $OrtDll
if ($UseCuda) {
    $env:PATH = "$CudaLibs;$CargoBin;$env:PATH"
} else {
    $env:PATH = "$CargoBin;$env:PATH"
}

# ── Features ──────────────────────────────────────────────────────────────────
$Features = "sortformer,load-dynamic"
if ($UseCuda) { $Features += ",cuda" }

# ── Build ─────────────────────────────────────────────────────────────────────
Write-Host "Building..." -ForegroundColor Cyan
Push-Location $RepoRoot
try {
    cargo build --release --example meeting --features $Features
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }
} finally {
    Pop-Location
}

# ── Assemble arguments ────────────────────────────────────────────────────────
$Exe = Join-Path $RepoRoot "target\release\examples\meeting.exe"
$RunArgs = @($Audio, "--samples-dir", $SamplesDir)

if ($UseCuda)    { $RunArgs += "--cuda" }
if ($Identify)   { $RunArgs += "--identify" }
if ($Speakers)   { $RunArgs += @("--speakers", $Speakers) }
if ($Output)     { $RunArgs += @("--output", $Output) }

# ── Run ───────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Running: meeting.exe $RunArgs" -ForegroundColor Cyan
Write-Host ""
& $Exe @RunArgs
exit $LASTEXITCODE
