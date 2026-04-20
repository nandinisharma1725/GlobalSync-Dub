"""
stage2_transcribe.py  —  Stage 2 (improved): Speech-to-Text using local Whisper

Changes from previous version:
  - Post-transcription turn merger: consecutive same-speaker segments with
    gap <= 0.8s have their TEXT joined into one clean turn.
    Fixes fragmented output (e.g. SPEAKER_04 in 8 fragments → 2 clean turns).
  - Windows fix preserved: audio passed as numpy array, not file path.

Install:
  pip install openai-whisper soundfile numpy librosa structlog
"""

import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

WHISPER_SAMPLE_RATE = 16_000
MIN_CHUNK_SAMPLES   = 1_600

_model      = None
_model_name = None


def _get_model(model_name: str = "base"):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            import whisper
        except ImportError:
            raise ImportError("Run:  pip install openai-whisper")
        log.info("stage2.model.loading", model=model_name)
        _model      = whisper.load_model(model_name, device="cpu")
        _model_name = model_name
        log.info("stage2.model.ready", model=model_name)
    return _model


def slice_audio(wav_path: str, start: float, end: float) -> np.ndarray:
    """
    Slices WAV to a 16kHz mono float32 array.
    Passes the array to Whisper directly — bypasses its internal ffmpeg
    subprocess (Windows [WinError 2] fix).
    """
    wav_path = str(Path(wav_path).resolve())
    info     = sf.info(wav_path)
    sr       = info.samplerate

    y, _ = sf.read(
        wav_path,
        start=int(start * sr),
        stop=int(end   * sr),
        dtype="float32",
    )

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != WHISPER_SAMPLE_RATE:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)
        except ImportError:
            raise ImportError("Run:  pip install librosa")

    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak

    if len(y) < MIN_CHUNK_SAMPLES:
        y = np.pad(y, (0, MIN_CHUNK_SAMPLES - len(y)))

    return y.astype(np.float32)


def transcribe_array(
    audio_array: np.ndarray,
    language:    str = "en",
    model_name:  str = "base",
) -> dict:
    """Transcribes a numpy array with Whisper. Returns {text, words}."""
    model  = _get_model(model_name)
    result = model.transcribe(
        audio_array,
        language=language,
        verbose=False,
        fp16=False,
    )

    text  = result["text"].strip()
    words = []
    for seg in result.get("segments", []):
        t = seg.get("text", "").strip()
        if t:
            words.append({
                "word":  t,
                "start": round(seg["start"], 3),
                "end":   round(seg["end"],   3),
            })

    return {"text": text, "words": words}


def merge_consecutive_turns(segments: list, max_gap: float = 0.8) -> list:
    """
    Merges consecutive same-speaker segments where gap <= max_gap.

    Joins their text and extends the time range. Re-indexes segment_ids.

    This is the fix for fragmented transcript output like:
      SPEAKER_04  "Thanks Marcus. Well..."
      SPEAKER_04  "I prepared some handouts."          ← gap 0.5s → merge
      SPEAKER_04  "to show you how the figures..."     ← gap 0.6s → merge
    Becomes one clean turn:
      SPEAKER_04  "Thanks Marcus. Well... I prepared some handouts.
                   to show you how the figures are looking."
    """
    if not segments:
        return []

    merged  = []
    current = dict(segments[0])
    current["words"] = list(current.get("words", []))

    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        if seg["speaker"] == current["speaker"] and gap <= max_gap:
            current["text"]  = current["text"].rstrip() + " " + seg["text"].lstrip()
            current["end"]   = seg["end"]
            current["words"].extend(seg.get("words", []))
        else:
            merged.append(current)
            current = dict(seg)
            current["words"] = list(current.get("words", []))

    merged.append(current)

    for i, s in enumerate(merged):
        s["segment_id"] = i

    return merged


def run(
    stage1_result:   dict,
    output_dir:      str,
    source_language: str = "en",
    model_name:      str = "base",
) -> dict:
    """
    Main entry point for Stage 2.

    Transcribes every speaker segment from Stage 1, then merges
    fragmented consecutive turns from the same speaker.

    Returns:
        {
          "segments": [
            {
              "segment_id": int,
              "speaker":    str,
              "start":      float,
              "end":        float,
              "text":       str,
              "words":      [{word, start, end}, ...],
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

    log.info("stage2.start",
             segments=len(raw_segments),
             model=model_name,
             language=source_language)

    _get_model(model_name)

    enriched = []

    for i, seg in enumerate(raw_segments):
        log.info("stage2.transcribe",
                 segment=i + 1, total=len(raw_segments),
                 speaker=seg["speaker"],
                 start=seg["start"], end=seg["end"])

        try:
            audio_array = slice_audio(wav_path, seg["start"], seg["end"])
            result      = transcribe_array(audio_array, source_language, model_name)

            if not result["text"]:
                log.warning("stage2.empty", segment=i)
                continue

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
                "chunk_path": "",
            })

            log.info("stage2.segment_done",
                     segment=i + 1,
                     text_preview=result["text"][:60])

        except Exception as e:
            log.error("stage2.segment_failed", segment=i, error=str(e))
            continue

    # Merge fragmented same-speaker turns
    before = len(enriched)
    enriched = merge_consecutive_turns(enriched, max_gap=0.8)
    log.info("stage2.turn_merge",
             before=before, after=len(enriched),
             collapsed=before - len(enriched))

    output = {"segments": enriched}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    log.info("stage2.complete",
             segments=len(enriched),
             speakers=list({s["speaker"] for s in enriched}))

    gc.collect()
    return output