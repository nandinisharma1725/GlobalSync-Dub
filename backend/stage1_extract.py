"""
stage1_extract.py

Stage 1 — Video upload, audio extraction, and speaker diarization.

Completely standalone — no relative imports, no FastAPI, no Celery.
Drop this file anywhere and import it directly.

What it does:
  1. Extracts audio from the video using FFmpeg (via imageio-ffmpeg)
  2. Runs pyannote speaker diarization to detect WHO speaks WHEN
  3. Cleans up the segments (merges tiny gaps, drops micro-segments)
  4. Returns a dict ready for Stage 2

Requirements:
  pip install imageio-ffmpeg soundfile numpy pyannote.audio torch structlog

Environment variable needed for pyannote:
  HF_TOKEN=hf_...   (get from huggingface.co — accept pyannote model license)
"""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE  = 16_000   # Hz — Whisper always expects 16kHz mono
MIN_SEGMENT_DURATION = 0.5      # seconds — drop segments shorter than this
MAX_MERGE_GAP        = 0.3      # seconds — merge same-speaker gaps smaller than this
MAX_REF_DURATION     = 90.0     # seconds — cap reference audio for voice cloning later
SUPPORTED_FORMATS    = {".mp4", ".mov", ".webm", ".mkv"}
MAX_FILE_SIZE_MB     = 500


# ── File validation ───────────────────────────────────────────────────────────

def validate_video_file(video_path: str) -> list[str]:
    """
    Returns a list of error strings. Empty list means the file is valid.
    Checks: file exists, supported format, under size limit.
    """
    p = Path(video_path)
    if not p.exists():
        return [f"File not found: {video_path}"]

    errors = []
    ext = p.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        errors.append(
            f"Unsupported format '{ext}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File too large ({size_mb:.0f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    return errors


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extracts audio from a video file using FFmpeg.
    Outputs a 16kHz mono WAV — exactly what Whisper expects.

    Uses imageio-ffmpeg so FFmpeg works on Windows without
    manually adding it to PATH.

    Args:
        video_path: Path to input video (MP4, MOV, etc.)
        output_dir: Directory where audio.wav will be written

    Returns:
        Absolute path to the extracted WAV file
    """
    video_path = str(Path(video_path).resolve())
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(output_dir / "audio.wav")
    ffmpeg   = imageio_ffmpeg.get_ffmpeg_exe()

    log.info("stage1.extract_audio.start",
             video=video_path, output=wav_path)

    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-i",      video_path,
            "-vn",                      # strip video track
            "-acodec", "pcm_s16le",     # 16-bit PCM
            "-ar",     "16000",         # 16 kHz sample rate
            "-ac",     "1",             # mono
            wav_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed.\n"
            f"Command stderr:\n{result.stderr}"
        )

    if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
        raise RuntimeError(f"FFmpeg ran but produced no audio file at: {wav_path}")

    size_kb = Path(wav_path).stat().st_size // 1024
    log.info("stage1.extract_audio.done", wav=wav_path, size_kb=size_kb)
    return wav_path


def get_audio_duration(wav_path: str) -> float:
    """Returns duration of a WAV file in seconds using soundfile."""
    info = sf.info(wav_path)
    return info.frames / info.samplerate


# ── Speaker diarization ───────────────────────────────────────────────────────

def run_diarization(wav_path: str, output_dir: str) -> list[dict]:
    """
    Runs pyannote.audio speaker diarization on the extracted WAV.

    Detects speaker changes and returns a sorted list of segments:
      [{"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2}, ...]

    Caches results to diarization.json — reruns won't re-process.

    Requires:
      HF_TOKEN environment variable set to your HuggingFace token.
      Accept the model license at:
      https://huggingface.co/pyannote/speaker-diarization-3.1

    Args:
        wav_path:   Path to 16kHz mono WAV
        output_dir: Working directory (cache written here)

    Returns:
        List of {speaker, start, end} dicts sorted by start time
    """
    cache_path = Path(output_dir) / "diarization.json"
    if cache_path.exists():
        log.info("stage1.diarization.cache_hit", path=str(cache_path))
        return json.loads(cache_path.read_text())

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "  1. Create a free account at https://huggingface.co\n"
            "  2. Accept the model license at:\n"
            "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. Generate a token at https://huggingface.co/settings/tokens\n"
            "  4. Add it to your .env file:  HF_TOKEN=hf_..."
        )

    log.info("stage1.diarization.start", wav=wav_path)

    try:
        import torch
        from pyannote.audio import Pipeline as DiarizationPipeline
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Run:  pip install pyannote.audio torch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("stage1.diarization.device", device=device)

    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(torch.device(device))

    diarization = pipeline(str(Path(wav_path).resolve()))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start":   round(turn.start, 3),
            "end":     round(turn.end,   3),
        })

    segments.sort(key=lambda x: x["start"])

    cache_path.write_text(json.dumps(segments, indent=2))
    log.info("stage1.diarization.done", segments=len(segments))

    gc.collect()
    return segments


# ── Segment cleanup ───────────────────────────────────────────────────────────

def merge_short_segments(
    segments:     list[dict],
    min_duration: float = MIN_SEGMENT_DURATION,
    max_gap:      float = MAX_MERGE_GAP,
) -> list[dict]:
    """
    Cleans up raw diarization output by:
      - Merging consecutive same-speaker segments with a gap < max_gap
      - Dropping any segment shorter than min_duration

    Why: raw diarization is noisy. A speaker might have a 0.2s silence
    mid-sentence that pyannote treats as a new turn. Merging avoids
    creating hundreds of micro-chunks that break Whisper.

    Args:
        segments:     Raw diarization output
        min_duration: Drop segments shorter than this (seconds)
        max_gap:      Merge same-speaker turns with a gap smaller than this

    Returns:
        Cleaned list of segments
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        gap  = seg["start"] - last["end"]

        if seg["speaker"] == last["speaker"] and gap < max_gap:
            # Extend the previous segment rather than starting a new one
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Remove segments that are too short to produce useful transcription
    return [s for s in merged if (s["end"] - s["start"]) >= min_duration]


# ── Main entry point ──────────────────────────────────────────────────────────

def run(video_path: str, output_dir: str) -> dict:
    """
    Runs the full Stage 1 pipeline:
      validate → extract audio → diarize → clean segments

    Args:
        video_path: Path to input video file
        output_dir: Working directory for all artifacts

    Returns:
        {
          "wav_path":      "path/to/audio.wav",
          "segments":      [{"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2}, ...],
          "speaker_count": 2,
          "speaker_ids":   ["SPEAKER_00", "SPEAKER_01"],
          "duration_sec":  145.3,
        }

    Raises:
        ValueError:       If the video file fails validation
        EnvironmentError: If HF_TOKEN is not set
        RuntimeError:     If FFmpeg fails
    """
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1 — Validate
    errors = validate_video_file(video_path)
    if errors:
        raise ValueError("Video validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    # Step 2 — Extract audio
    wav_path = extract_audio(video_path, output_dir)

    duration = get_audio_duration(wav_path)
    log.info("stage1.audio_duration", seconds=round(duration, 1))

    # Step 3 — Diarize
    raw_segments = run_diarization(wav_path, output_dir)

    # Step 4 — Clean segments
    segments    = merge_short_segments(raw_segments)
    speaker_ids = sorted({s["speaker"] for s in segments})

    log.info(
        "stage1.complete",
        segments=len(segments),
        speakers=len(speaker_ids),
        duration_sec=round(duration, 1),
    )

    return {
        "wav_path":      wav_path,
        "segments":      segments,
        "speaker_count": len(speaker_ids),
        "speaker_ids":   speaker_ids,
        "duration_sec":  round(duration, 1),
    }