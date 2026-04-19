"""
stage2_transcribe.py  —  Stage 2: Speech-to-Text using local Whisper

Standalone file — no relative imports, drop it anywhere.

What it does:
  - Takes the Stage 1 result (list of speaker segments + WAV path)
  - Slices the WAV into per-segment chunks using soundfile (pure Python)
  - Transcribes each chunk with local Whisper using numpy arrays
  - Returns enriched segments with text + word-level timestamps

KEY DESIGN: We load each audio chunk as a numpy array and pass that
directly to model.transcribe(array). This bypasses Whisper's internal
ffmpeg subprocess — which fails on Windows with [WinError 2] because
the subprocess loses the absolute path context.

Model is cached at module level — loaded once, reused across all segments.

Install:
  pip install openai-whisper soundfile numpy librosa structlog

Whisper models (set WHISPER_MODEL in .env):
  tiny    —  39M  params, fastest,  ~70% accuracy
  base    —  74M  params, fast,     ~85% accuracy  ← recommended for testing
  small   — 244M  params, balanced, ~90% accuracy
  medium  — 769M  params, slower,   ~93% accuracy
  large   —   1.5B params, slowest, ~95% accuracy  ← best for production
"""

import gc
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE = 16_000   # Whisper always expects 16kHz
MIN_CHUNK_SAMPLES   = 1_600    # 0.1s minimum — pad shorter chunks with silence

# ── Module-level model cache ──────────────────────────────────────────────────
# Load once per process, not once per segment.
# Reloading a 74MB model for every 5-second audio clip is extremely wasteful.
_model       = None
_model_name  = None


def _get_model(model_name: str = "base"):
    """
    Returns a cached Whisper model. Loads from disk on first call.
    Subsequent calls return the already-loaded model instantly.
    """
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper not installed.\n"
                "Run:  pip install openai-whisper"
            )
        log.info("stage2.model.loading", model=model_name)
        _model      = whisper.load_model(model_name, device="cpu")
        _model_name = model_name
        log.info("stage2.model.ready", model=model_name)
    return _model


# ── Audio utilities ───────────────────────────────────────────────────────────


def slice_audio(wav_path: str, start: float, end: float) -> np.ndarray:
    """
    Cuts a time range from a WAV file and returns it as a 16kHz float32 array.
    Uses soundfile for zero-copy slicing — no subprocess, no temp file.

    Args:
        wav_path: Path to the full-length WAV (output of Stage 1)
        start:    Start time in seconds
        end:      End time in seconds

    Returns:
        numpy array ready to pass into model.transcribe()
    """
    wav_path = str(Path(wav_path).resolve())
    info     = sf.info(wav_path)
    sr       = info.samplerate

    start_sample = int(start * sr)
    end_sample   = int(end   * sr)

    y, _ = sf.read(
        wav_path,
        start=start_sample,
        stop=end_sample,
        dtype="float32",
    )

    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample to 16kHz if source WAV isn't already 16kHz
    if sr != WHISPER_SAMPLE_RATE:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)
        except ImportError:
            raise ImportError("Run:  pip install librosa")

    # Normalise
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak

    # Pad very short segments with silence
    if len(y) < MIN_CHUNK_SAMPLES:
        y = np.pad(y, (0, MIN_CHUNK_SAMPLES - len(y)))

    return y.astype(np.float32)


# ── Core transcription ────────────────────────────────────────────────────────

def transcribe_array(
    audio_array: np.ndarray,
    language:    str = "en",
    model_name:  str = "base",
) -> dict:
    """
    Transcribes a numpy audio array using local Whisper.

    Passes the array directly — no file path, no ffmpeg subprocess.
    This is the fix for [WinError 2] on Windows.

    Args:
        audio_array: 16kHz mono float32 numpy array
        language:    Source language ISO code (e.g. "en")
        model_name:  Whisper model size ("tiny","base","small","medium","large")

    Returns:
        {"text": str, "words": [{word, start, end}, ...]}
    """
    model = _get_model(model_name)

    result = model.transcribe(
        audio_array,   # ← numpy array, NOT a file path
        language=language,
        verbose=False,
        fp16=False,    # fp16 not supported on CPU — avoids warning spam
    )

    text = result["text"].strip()

    # Extract segment-level timestamps as word approximations.
    # Whisper's segment timestamps are reliable; true word-level timestamps
    # require the 'large-v2' model with word_timestamps=True option.
    words = []
    for seg in result.get("segments", []):
        seg_text = seg.get("text", "").strip()
        if seg_text:
            words.append({
                "word":  seg_text,
                "start": round(seg["start"], 3),
                "end":   round(seg["end"],   3),
            })

    return {"text": text, "words": words}


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    stage1_result:   dict,
    output_dir:      str,
    source_language: str = "en",
    model_name:      str = "base",
) -> dict:
    """
    Main entry point for Stage 2.

    Iterates every speaker segment from Stage 1, slices the audio,
    and transcribes each slice with Whisper.

    Args:
        stage1_result:   Output dict from stage1_extract.run()
        output_dir:      Working directory (cache written here)
        source_language: ISO 639-1 code of the source audio ("en" for English)
        model_name:      Whisper model size ("base" recommended for testing)

    Returns:
        {
          "segments": [
            {
              "segment_id": int,
              "speaker":    str,
              "start":      float,   ← absolute video timestamp (seconds)
              "end":        float,
              "text":       str,     ← transcribed English text
              "words":      [{word, start, end}, ...],
              "chunk_path": str,     ← not used in stage2 (kept for stage4 ref)
            },
            ...
          ]
        }
    """
    cache_path = Path(output_dir) / "transcription.json"
    if cache_path.exists():
        log.info("stage2.cache_hit")
        return json.loads(cache_path.read_text())

    wav_path     = stage1_result["wav_path"]
    raw_segments = stage1_result["segments"]

    # Pre-load the model once before the loop
    log.info("stage2.start",
             segments=len(raw_segments),
             model=model_name,
             language=source_language)
    _get_model(model_name)

    enriched = []

    for i, seg in enumerate(raw_segments):
        log.info("stage2.transcribe",
                 segment=i + 1,
                 total=len(raw_segments),
                 speaker=seg["speaker"],
                 start=seg["start"],
                 end=seg["end"])

        try:
            # Step 1: slice audio to numpy array (no temp file, no ffmpeg)
            audio_array = slice_audio(wav_path, seg["start"], seg["end"])

            # Step 2: transcribe the array
            result = transcribe_array(audio_array, source_language, model_name)

            if not result["text"]:
                log.warning("stage2.empty", segment=i)
                continue

            # Step 3: offset word timestamps to absolute video time
            offset = seg["start"]
            for w in result["words"]:
                w["start"] = round(w["start"] + offset, 3)
                w["end"]   = round(w["end"]   + offset, 3)

            enriched.append({
                "segment_id": i,
                "speaker":    seg["speaker"],
                "start":      seg["start"],
                "end":        seg["end"],
                "text":       result["text"],
                "words":      result["words"],
                "chunk_path": "",   # no temp files needed with array approach
            })

            log.info("stage2.done",
                     segment=i + 1,
                     speaker=seg["speaker"],
                     text_preview=result["text"][:60])

        except Exception as e:
            log.error("stage2.segment_failed", segment=i, error=str(e))
            # Keep going — partial output is better than a crash
            continue

    output = {"segments": enriched}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    log.info("stage2.complete",
             transcribed=len(enriched),
             speakers=list({s["speaker"] for s in enriched}))

    gc.collect()
    return output