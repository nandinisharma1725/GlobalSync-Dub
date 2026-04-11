"""
backend/pipeline/transcribe.py

Stage 2 — Speech-to-Text using OpenAI Whisper

What this does:
  - Sends each speaker segment's audio to Whisper
  - Gets back text + word-level timestamps
  - Returns enriched segments ready for translation

Why word-level timestamps matter:
  The timestamps are the backbone of sync. Every word's start/end time
  is used later (Stage 5) to time-stretch the dubbed audio correctly.

Output per segment:
  {
    "speaker": "SPEAKER_00",
    "start": 2.1,
    "end": 6.8,
    "text": "Good morning everyone, let's begin the Q4 review.",
    "words": [
      {"word": "Good", "start": 2.1, "end": 2.4},
      ...
    ]
  }
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
import structlog

from openai import OpenAI
from ..utils.config import get_settings
from ..utils.storage import save_artifact

log = structlog.get_logger()
settings = get_settings()


def slice_audio(
    wav_path: str,
    start: float,
    end: float,
    output_path: str,
) -> str:
    """
    Cuts a WAV segment using FFmpeg (fast, no re-encode).
    Adds 0.1s silence padding at start/end to help Whisper.
    """
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid segment: start={start} end={end}")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(max(0, start - 0.05)),   # tiny pre-roll
            "-t",  str(duration + 0.1),           # tiny post-roll
            "-i",  wav_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg slice failed:\n{result.stderr}")
    return output_path


def transcribe_segment(
    client: OpenAI,
    audio_path: str,
    language: str = "en",
) -> dict:
    """
    Sends one audio slice to Whisper.
    Returns {"text": "...", "words": [...]}

    NOTE: verbose_json response format gives us word-level timestamps.
    """
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=settings.default_whisper_model,
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    text = response.text.strip()

    # Extract word timestamps — offset by segment start
    words = []
    if hasattr(response, "words") and response.words:
        for w in response.words:
            words.append({
                "word": w.word,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
            })

    return {"text": text, "words": words}


def run(stage1_result: dict, output_dir: str, source_language: str = "en") -> dict:
    """
    Main entry point for Stage 2.

    Processes every segment from Stage 1, transcribing each audio slice.
    Saves results to disk so the pipeline can be resumed on failure.

    Args:
        stage1_result: Output from stage1_extract.run()
        output_dir: Working directory for this job
        source_language: ISO 639-1 code (default "en" for English board meetings)

    Returns:
        {
          "segments": [
            {
              "speaker": "SPEAKER_00",
              "start": 2.1,
              "end": 6.8,
              "text": "...",
              "words": [...],
            },
            ...
          ]
        }
    """
    cache_path = Path(output_dir) / "transcription.json"
    if cache_path.exists():
        log.info("stage2.cache_hit")
        return json.loads(cache_path.read_text())

    client = OpenAI(api_key=settings.openai_api_key)

    wav_path = stage1_result["wav_path"]
    raw_segments = stage1_result["segments"]
    chunks_dir = Path(output_dir) / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    enriched_segments = []

    for i, seg in enumerate(raw_segments):
        chunk_path = str(chunks_dir / f"chunk_{i:04d}.wav")

        log.info(
            "stage2.transcribe",
            segment=i + 1,
            total=len(raw_segments),
            speaker=seg["speaker"],
            start=seg["start"],
            end=seg["end"],
        )

        try:
            # Slice audio for this segment
            slice_audio(wav_path, seg["start"], seg["end"], chunk_path)

            # Transcribe
            result = transcribe_segment(client, chunk_path, language=source_language)

            if not result["text"]:
                log.warning("stage2.empty_transcription", segment=i)
                continue

            # Offset word timestamps back to absolute video time
            offset = seg["start"]
            for word in result["words"]:
                word["start"] = round(word["start"] + offset, 3)
                word["end"] = round(word["end"] + offset, 3)

            enriched_segments.append({
                "segment_id": i,
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": result["text"],
                "words": result["words"],
                "chunk_path": chunk_path,
            })

        except Exception as e:
            log.error("stage2.segment_failed", segment=i, error=str(e))
            # Skip failed segments — partial output is better than a full crash
            continue

    output = {"segments": enriched_segments}
    cache_path.write_text(json.dumps(output, indent=2))

    log.info(
        "stage2.complete",
        total_segments=len(enriched_segments),
        speakers=list({s["speaker"] for s in enriched_segments}),
    )
    return output