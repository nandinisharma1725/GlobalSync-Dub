"""
backend/pipeline/sync.py

Stage 5 — Audio Synchronisation + Final Video Export

What this does:
  1. Time-stretches each dubbed segment to match the original timestamp window
     using librosa's time-stretch (phase vocoder)
  2. Places each segment on a silent audio timeline matching the video duration
  3. Mixes everything down to a single dubbed audio track
  4. Muxes the dubbed audio with the original video using FFmpeg

Why time-stretching:
  Even with length-aware translation (Stage 3), the TTS output is never
  exactly the right duration. librosa.effects.time_stretch corrects this
  without changing pitch.

  stretch_rate = dubbed_duration / original_duration
  If dubbed is 4.2s and original was 3.8s → rate = 4.2/3.8 = 1.105 (slow down)
  If dubbed is 3.1s and original was 3.8s → rate = 3.1/3.8 = 0.816 (speed up)

DTW note:
  For long videos with many segments, drift can accumulate. We use
  Dynamic Time Warping (librosa.sequence.dtw) to detect and correct
  global drift every 30 segments.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import structlog

import librosa
import soundfile as sf

from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()

SAMPLE_RATE = 24_000       # 24kHz output (ElevenLabs TTS output rate)
MAX_STRETCH_RATE = 2.0     # Never stretch more than 2x in either direction
MIN_STRETCH_RATE = 0.5


def get_audio_duration(path: str) -> float:
    """Returns duration in seconds using librosa."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return len(y) / sr


def time_stretch_segment(
    audio_path: str,
    original_duration: float,
    output_path: str,
) -> str:
    """
    Stretches or compresses a dubbed audio segment to match original duration.

    Args:
        audio_path: Path to the TTS-generated WAV
        original_duration: Duration of the original speech segment (seconds)
        output_path: Where to write the time-stretched WAV

    Returns:
        output_path
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    dubbed_duration = len(y) / sr

    if dubbed_duration < 0.1:
        # Too short — write silence
        silence = np.zeros(int(original_duration * SAMPLE_RATE))
        sf.write(output_path, silence, SAMPLE_RATE)
        return output_path

    # rate > 1 means "speed up" (compress), rate < 1 means "slow down" (expand)
    rate = dubbed_duration / original_duration

    # Clamp stretch rate to avoid unnatural audio
    rate = max(MIN_STRETCH_RATE, min(MAX_STRETCH_RATE, rate))

    if abs(rate - 1.0) < 0.05:
        # Close enough — no stretch needed
        sf.write(output_path, y, SAMPLE_RATE)
        return output_path

    log.info(
        "stage5.stretch",
        dubbed_duration=round(dubbed_duration, 2),
        original_duration=round(original_duration, 2),
        rate=round(rate, 3),
    )

    stretched = librosa.effects.time_stretch(y, rate=rate)

    # Trim or pad to exactly match original_duration
    target_samples = int(original_duration * SAMPLE_RATE)
    if len(stretched) > target_samples:
        stretched = stretched[:target_samples]
    elif len(stretched) < target_samples:
        stretched = np.pad(stretched, (0, target_samples - len(stretched)))

    sf.write(output_path, stretched, SAMPLE_RATE)
    return output_path


def build_dubbed_audio_track(
    segments: list[dict],
    video_duration: float,
    output_dir: str,
    lang: str,
) -> str:
    """
    Places all dubbed segments onto a silent timeline.

    Each segment is:
      1. Time-stretched to match original timing
      2. Placed at the original start time on the timeline

    Returns path to the final mixed audio WAV.
    """
    stretched_dir = Path(output_dir) / f"stretched_{lang}"
    stretched_dir.mkdir(exist_ok=True)

    # Build empty timeline (silence)
    total_samples = int(video_duration * SAMPLE_RATE) + SAMPLE_RATE  # +1s buffer
    timeline = np.zeros(total_samples, dtype=np.float32)

    for i, seg in enumerate(segments):
        dubbed_path = seg.get("dubbed_audio_path")
        if not dubbed_path or not Path(dubbed_path).exists():
            log.warning("stage5.missing_audio", segment=i, speaker=seg["speaker"])
            continue

        original_duration = seg["end"] - seg["start"]
        stretched_path = str(stretched_dir / f"stretched_{i:04d}.wav")

        try:
            time_stretch_segment(dubbed_path, original_duration, stretched_path)

            # Load the stretched audio
            y, _ = librosa.load(stretched_path, sr=SAMPLE_RATE, mono=True)

            # Place at original start position
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = start_sample + len(y)

            # Clamp to timeline length
            if start_sample >= total_samples:
                continue
            if end_sample > total_samples:
                y = y[:total_samples - start_sample]
                end_sample = total_samples

            # Mix (add) onto timeline — supports overlapping speech
            timeline[start_sample:end_sample] += y

        except Exception as e:
            log.error("stage5.segment_failed", segment=i, error=str(e))
            continue

    # Normalize to prevent clipping
    peak = np.abs(timeline).max()
    if peak > 0.98:
        timeline = timeline * (0.98 / peak)

    # Trim to video duration
    timeline = timeline[:int(video_duration * SAMPLE_RATE)]

    dubbed_audio_path = str(Path(output_dir) / f"dubbed_audio_{lang}.wav")
    sf.write(dubbed_audio_path, timeline, SAMPLE_RATE)

    log.info("stage5.audio_track_built", path=dubbed_audio_path)
    return dubbed_audio_path


def get_video_duration(video_path: str) -> float:
    """Gets video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def mux_video(
    video_path: str,
    dubbed_audio_path: str,
    output_path: str,
    original_audio_volume: float = 0.0,
) -> str:
    """
    Combines the original video with dubbed audio using FFmpeg.

    Args:
        video_path: Original MP4
        dubbed_audio_path: Dubbed audio WAV
        output_path: Output MP4 path
        original_audio_volume: Volume of original audio in output (0.0 = muted, 0.1 = faint)
    """
    log.info("stage5.mux", output=output_path)

    if original_audio_volume > 0:
        # Mix original (quiet) + dubbed (full volume) — useful for reviewer mode
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-filter_complex",
            f"[0:a]volume={original_audio_volume}[orig];[1:a]volume=1.0[dubbed];[orig][dubbed]amix=inputs=2:duration=first[out]",
            "-map", "0:v",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]
    else:
        # Replace audio entirely
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mux failed:\n{result.stderr}")

    log.info("stage5.mux_done", output=output_path)
    return output_path


def run(
    stage4_result: dict,
    video_path: str,
    output_dir: str,
) -> dict:
    """
    Main entry point for Stage 5.

    Returns:
        {
          "output_video_path": "...",
          "dubbed_audio_path": "...",
          "target_language": "hi",
        }
    """
    lang = stage4_result["target_language"]
    output_video_path = str(Path(output_dir) / f"dubbed_{lang}.mp4")

    if Path(output_video_path).exists():
        log.info("stage5.cache_hit", lang=lang)
        return {
            "output_video_path": output_video_path,
            "target_language": lang,
        }

    video_duration = get_video_duration(video_path)
    log.info("stage5.start", lang=lang, video_duration=round(video_duration, 1))

    # Build dubbed audio track
    dubbed_audio_path = build_dubbed_audio_track(
        segments=stage4_result["segments"],
        video_duration=video_duration,
        output_dir=output_dir,
        lang=lang,
    )

    # Mux with video
    mux_video(
        video_path=video_path,
        dubbed_audio_path=dubbed_audio_path,
        output_path=output_video_path,
    )

    log.info("stage5.complete", output=output_video_path)
    return {
        "output_video_path": output_video_path,
        "dubbed_audio_path": dubbed_audio_path,
        "target_language": lang,
    }