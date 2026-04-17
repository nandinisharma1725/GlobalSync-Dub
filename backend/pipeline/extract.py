"""
backend/pipeline/extract.py

Stage 1 — Video → Audio extraction + Speaker Diarization

What this does:
  1. Strips the audio track from the uploaded MP4 using FFmpeg
  2. Runs pyannote speaker diarization to detect WHO speaks WHEN
  3. Returns a list of utterance segments with speaker IDs and timestamps

Output structure:
  [
    {"speaker": "SPEAKER_00", "start": 0.5,  "end": 4.2},
    {"speaker": "SPEAKER_01", "start": 4.5,  "end": 9.1},
    ...
  ]
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
import structlog
import imageio_ffmpeg

from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extracts audio from video as a 16kHz mono WAV.
    16kHz mono is the format Whisper expects — converting here saves processing later.

    Args:
        video_path: Path to input MP4
        output_dir: Directory to write output WAV

    Returns:
        Path to extracted WAV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(output_dir / "audio.wav")

    log.info("stage1.extract_audio", video=video_path, output=wav_path)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", "16000",           # 16 kHz sample rate (Whisper standard)
            "-ac", "1",               # mono
            wav_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    log.info("stage1.extract_audio.done", wav=wav_path)
    return wav_path


def run_diarization(wav_path: str, output_dir: str) -> list[dict]:
    """
    Detects speaker turns using pyannote.audio.

    Requires HF_TOKEN in .env — accept the pyannote model license at:
    https://huggingface.co/pyannote/speaker-diarization-3.1

    Args:
        wav_path: Path to 16kHz mono WAV
        output_dir: Directory to cache diarization output

    Returns:
        List of {speaker, start, end} dicts sorted by start time
    """
    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline

    cache_path = Path(output_dir) / "diarization.json"

    # Cache diarization result so we don't re-run on retries
    if cache_path.exists():
        log.info("stage1.diarization.cache_hit", path=str(cache_path))
        return json.loads(cache_path.read_text())

    log.info("stage1.diarization.start", wav=wav_path)

    hf_token = settings.hf_token
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not set. Get a token at https://huggingface.co and "
            "accept the pyannote/speaker-diarization-3.1 model license."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("stage1.diarization.device", device=device)

    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(torch.device(device))

    diarization = pipeline(wav_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    # Sort by start time
    segments.sort(key=lambda x: x["start"])

    cache_path.write_text(json.dumps(segments, indent=2))
    log.info("stage1.diarization.done", segments=len(segments))
    return segments


def merge_short_segments(segments: list[dict], min_duration: float = 0.5) -> list[dict]:
    """
    Merges consecutive segments from the same speaker if they are very short.
    Avoids creating tiny audio chunks that produce poor TTS output.
    """
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        same_speaker = seg["speaker"] == last["speaker"]

        # Merge if same speaker and short gap (< 0.3s)
        if same_speaker and gap < 0.3:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Remove segments shorter than min_duration
    merged = [s for s in merged if (s["end"] - s["start"]) >= min_duration]
    return merged


def run(video_path: str, output_dir: str) -> dict:
    """
    Main entry point for Stage 1.

    Returns:
        {
          "wav_path": "...",
          "segments": [...],
          "speaker_count": N,
        }
    """
    wav_path = extract_audio(video_path, output_dir)
    segments = run_diarization(wav_path, output_dir)
    segments = merge_short_segments(segments)

    speaker_ids = list({s["speaker"] for s in segments})
    log.info(
        "stage1.complete",
        segments=len(segments),
        speakers=len(speaker_ids),
        speaker_ids=speaker_ids,
    )

    return {
        "wav_path": wav_path,
        "segments": segments,
        "speaker_count": len(speaker_ids),
        "speaker_ids": speaker_ids,
    }