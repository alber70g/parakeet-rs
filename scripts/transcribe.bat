@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: transcribe.bat — Transcribe a meeting recording with speaker diarization
::
:: Usage:
::   transcribe.bat <audio.wav> [options]
::
:: Options:
::   --samples-dir <dir>    Directory for per-speaker WAV snippets (default: .\speaker_samples)
::   --speakers <file>      Path to speakers.json  (e.g. {"0":"Alice","1":"Bob"})
::   --output <file>        Write transcript to file in addition to stdout
::   --identify             Extract one sample per speaker, print timestamps,
::                          write template speakers.json. Does not transcribe.
::   --no-cuda              Force CPU inference (much slower, no GPU required)
::
:: Examples:
::   Step 1 — identify speakers:
::     transcribe.bat recording.wav --identify --speakers speakers.json
::
::   Step 2 — run with names:
::     transcribe.bat recording.wav --speakers speakers.json --output transcript.txt
:: ============================================================================

:: ── Paths ────────────────────────────────────────────────────────────────────
set "ORT_DLL=C:\Python312\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
set "CUDA_LIBS=C:\Users\%USERNAME%\AppData\Local\uv\cache\archive-v0\hdnzU4gUhSdieW1M98Xzu\torch\lib"
set "CARGO_BIN=%USERPROFILE%\.cargo\bin"

:: ── Repo root (script lives in <repo>\scripts\) ───────────────────────────────
set "REPO_ROOT=%~dp0.."
set "EXE=%REPO_ROOT%\target\release\examples\meeting.exe"

:: ── Parse arguments ───────────────────────────────────────────────────────────
if "%~1"=="" (
    echo ERROR: Audio file argument required.
    echo Usage: %~nx0 ^<audio.wav^> [options]
    exit /b 1
)

set "AUDIO=%~1"
set "SAMPLES_DIR=.\speaker_samples"
set "SPEAKERS_ARG="
set "OUTPUT_ARG="
set "IDENTIFY_ARG="
set "NO_CUDA=0"

shift
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--samples-dir"  ( set "SAMPLES_DIR=%~2"  & shift & shift & goto parse_args )
if /i "%~1"=="--speakers"     ( set "SPEAKERS_ARG=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--output"       ( set "OUTPUT_ARG=%~2"   & shift & shift & goto parse_args )
if /i "%~1"=="--identify"     ( set "IDENTIFY_ARG=1"   & shift           & goto parse_args )
if /i "%~1"=="--no-cuda"      ( set "NO_CUDA=1"        & shift           & goto parse_args )
echo WARNING: Unknown argument: %~1
shift
goto parse_args
:args_done

:: ── Validate audio file ───────────────────────────────────────────────────────
if not exist "%AUDIO%" (
    echo ERROR: Audio file not found: %AUDIO%
    exit /b 1
)

:: ── Validate ORT DLL ─────────────────────────────────────────────────────────
if not exist "%ORT_DLL%" (
    echo ERROR: ORT DLL not found at %ORT_DLL%
    echo Install onnxruntime-gpu: pip install onnxruntime-gpu
    exit /b 1
)

:: ── CUDA availability check ───────────────────────────────────────────────────
set "USE_CUDA=1"
if "%NO_CUDA%"=="1" set "USE_CUDA=0"
if "%USE_CUDA%"=="1" (
    if not exist "%CUDA_LIBS%\cublas64_12.dll" (
        echo WARNING: PyTorch CUDA libs not found at %CUDA_LIBS% -- falling back to CPU
        set "USE_CUDA=0"
    )
)

:: ── Environment ───────────────────────────────────────────────────────────────
set "ORT_DYLIB_PATH=%ORT_DLL%"
if "%USE_CUDA%"=="1" (
    set "PATH=%CUDA_LIBS%;%CARGO_BIN%;%PATH%"
) else (
    set "PATH=%CARGO_BIN%;%PATH%"
)

:: ── Features ──────────────────────────────────────────────────────────────────
set "FEATURES=sortformer,load-dynamic"
if "%USE_CUDA%"=="1" set "FEATURES=%FEATURES%,cuda"

:: ── Build ─────────────────────────────────────────────────────────────────────
echo Building...
pushd "%REPO_ROOT%"
cargo build --release --example meeting --features "%FEATURES%"
if errorlevel 1 (
    echo ERROR: Build failed
    popd
    exit /b 1
)
popd

:: ── Assemble run arguments ────────────────────────────────────────────────────
set "RUN_ARGS=%AUDIO% --samples-dir %SAMPLES_DIR%"
if "%USE_CUDA%"=="1"     set "RUN_ARGS=%RUN_ARGS% --cuda"
if defined IDENTIFY_ARG  set "RUN_ARGS=%RUN_ARGS% --identify"
if defined SPEAKERS_ARG  set "RUN_ARGS=%RUN_ARGS% --speakers %SPEAKERS_ARG%"
if defined OUTPUT_ARG    set "RUN_ARGS=%RUN_ARGS% --output %OUTPUT_ARG%"

:: ── Run ───────────────────────────────────────────────────────────────────────
echo.
echo Running: meeting.exe %RUN_ARGS%
echo.
"%EXE%" %RUN_ARGS%
exit /b %ERRORLEVEL%
