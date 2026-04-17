"""
backend/pipeline/transcribe.py  (fixed for Windows — WinError 2)

Root cause of the error:
  Whisper's model.transcribe(path) internally spawns an ffmpeg subprocess
  to decode audio. On Windows, this subprocess inherits a different working
  directory, so relative (and sometimes absolute) paths silently resolve to
  nothing → [WinError 2] The system cannot find the file specified.

Fix:
  Load the audio as a float32 numpy array ourselves using soundfile,
  then pass the array directly to model.transcribe(). Whisper fully supports
  numpy array input — this bypasses its internal ffmpeg call entirely.
"""

import os
import json
import time
import gc
from pathlib import Path
import structlog
import numpy as np
import soundfile as sf
import whisper

from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()

WHISPER_SAMPLE_RATE = 16_000   # Whisper always expects 16 kHz mono float32


def slice_audio(wav_path: str, start: float, end: float, output_path: str) -> str:
    """
    Slices a WAV segment using soundfile (pure Python, no external deps).
    Writes a 16 kHz mono WAV that Whisper can consume.
    """
    wav_path    = str(Path(wav_path).resolve())
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("slice_audio.start", input=wav_path, output=output_path,
             start=start, end=end)

    info = sf.info(wav_path)
    sr   = info.samplerate

    start_sample = int(start * sr)
    end_sample   = int(end   * sr)

    y, _ = sf.read(wav_path, start=start_sample, stop=end_sample, dtype="float32")

    if y.ndim > 1:                        # stereo → mono
        y = y.mean(axis=1)

    if len(y) == 0:
        log.warning("slice_audio.empty_segment", start=start, end=end)
        y = np.zeros(int(sr * 0.1), dtype="float32")

    # Resample to 16 kHz if the source WAV isn't already 16 kHz
    if sr != WHISPER_SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)

    sf.write(output_path, y, WHISPER_SAMPLE_RATE)

    # Small delay so Windows flushes the file handle before Whisper touches it
    time.sleep(0.1)

    file_size = Path(output_path).stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Written audio file is empty: {output_path}")

    log.info("slice_audio.done", output=output_path, size=file_size)
    return output_path


def _load_audio_array(audio_path: str) -> np.ndarray:
    """
    Reads a WAV file into a 16 kHz mono float32 numpy array.
    This is what we pass directly to Whisper — no ffmpeg subprocess involved.
    """
    audio_path = str(Path(audio_path).resolve())

    y, sr = sf.read(audio_path, dtype="float32")

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != WHISPER_SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)

    # Whisper expects float32 in [-1, 1]
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak

    return y.astype(np.float32)


# Module-level model cache — load once per worker process, not per segment
_whisper_model = None

def _get_model():
    global _whisper_model
    if _whisper_model is None:
        log.info("transcribe.loading_model", model="base")
        _whisper_model = whisper.load_model("base", device="cpu")
    return _whisper_model


def transcribe_segment(audio_path: str, language: str = "en") -> dict:
    """
    Transcribes one audio segment.

    KEY CHANGE: loads the WAV into a numpy array first, then calls
    model.transcribe(array) instead of model.transcribe(path).
    This completely avoids Whisper's internal ffmpeg subprocess,
    which is the source of [WinError 2] on Windows.

    Returns:
        {"text": "...", "words": [...]}
    """
    audio_path_abs = str(Path(audio_path).resolve())

    # Verify the file is accessible before we do anything
    if not Path(audio_path_abs).exists():
        raise FileNotFoundError(f"Audio chunk not found: {audio_path_abs}")

    try:
        # ── Load audio as numpy array (bypasses Whisper's ffmpeg) ──────────
        log.info("transcribe.load_audio", path=audio_path_abs)
        audio_array = _load_audio_array(audio_path_abs)

        log.info("transcribe.start",
                 samples=len(audio_array),
                 duration_s=round(len(audio_array) / WHISPER_SAMPLE_RATE, 2))

        model = _get_model()

        # Pass numpy array directly — no file path, no ffmpeg subprocess
        result = model.transcribe(
            audio_array,           # <-- numpy array, not a file path
            language=language,
            verbose=False,
            fp16=False,            # CPU only on most dev machines
        )

        text = result["text"].strip()

        # Extract segment-level timestamps as word approximations
        words = []
        for seg in result.get("segments", []):
            seg_text = seg.get("text", "").strip()
            if seg_text:
                words.append({
                    "word":  seg_text,
                    "start": round(seg["start"], 3),
                    "end":   round(seg["end"],   3),
                })

        log.info("transcribe.done", text_preview=text[:60], segments=len(words))
        return {"text": text, "words": words}

    except Exception as e:
        log.error("transcribe.failed", path=audio_path_abs, error=str(e))
        raise


def run(stage1_result: dict, output_dir: str, source_language: str = "en") -> dict:
    """
    Main entry point for Stage 2.
    Iterates every diarized segment, slices audio, transcribes with Whisper.
    Results are cached to disk so the pipeline can resume after a crash.
    """
    cache_path = Path(output_dir) / "transcription.json"
    if cache_path.exists():
        log.info("stage2.cache_hit")
        return json.loads(cache_path.read_text())

    wav_path     = stage1_result["wav_path"]
    raw_segments = stage1_result["segments"]
    chunks_dir   = Path(output_dir) / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    enriched_segments = []

    for i, seg in enumerate(raw_segments):
        chunk_path = str(chunks_dir / f"chunk_{i:04d}.wav")

        log.info("stage2.transcribe",
                 segment=i + 1, total=len(raw_segments),
                 speaker=seg["speaker"],
                 start=seg["start"], end=seg["end"])

        try:
            slice_audio(wav_path, seg["start"], seg["end"], chunk_path)
            result = transcribe_segment(chunk_path, language=source_language)

            if not result["text"]:
                log.warning("stage2.empty_transcription", segment=i)
                continue

            # Offset word timestamps back to absolute video time
            offset = seg["start"]
            for word in result["words"]:
                word["start"] = round(word["start"] + offset, 3)
                word["end"]   = round(word["end"]   + offset, 3)

            enriched_segments.append({
                "segment_id": i,
                "speaker":    seg["speaker"],
                "start":      seg["start"],
                "end":        seg["end"],
                "text":       result["text"],
                "words":      result["words"],
                "chunk_path": chunk_path,
            })

        except Exception as e:
            log.error("stage2.segment_failed", segment=i, error=str(e))
            continue   # partial output is better than a full crash

    output = {"segments": enriched_segments}
    cache_path.write_text(json.dumps(output, indent=2))

    log.info("stage2.complete",
             total_segments=len(enriched_segments),
             speakers=list({s["speaker"] for s in enriched_segments}))
    return output