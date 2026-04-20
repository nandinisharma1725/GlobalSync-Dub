"""
stage1_extract.py  —  Stage 1 (improved): Video upload, audio extraction, speaker diarization

Changes from previous version:
  - MIN_SEGMENT_DURATION raised 0.5s → 1.0s  (filters noise/cross-talk fragments)
  - MAX_MERGE_GAP raised 0.3s → 0.6s         (meetings have natural pauses mid-thought)
  - Orphan reassignment pass: a tiny segment sandwiched between turns of the
    same speaker (e.g. the 0.44s "are" fragment) is reassigned to that speaker
    instead of creating a spurious new speaker entry
  - Two-pass merge: standard merge → orphan fix → merge again

Install:
  pip install imageio-ffmpeg soundfile numpy structlog pyannote.audio torch torchaudio

Environment variable (for diarization):
  HF_TOKEN=hf_...   (huggingface.co → Settings → Tokens)
  Accept model license: huggingface.co/pyannote/speaker-diarization-3.1
"""

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE  = 16_000
MIN_SEGMENT_DURATION = 1.0    # raised from 0.5 — filters micro-fragments
MAX_MERGE_GAP        = 0.6    # raised from 0.3 — handles natural pauses mid-sentence
ORPHAN_GAP_THRESHOLD = 1.5    # max gap to a neighbor for orphan reassignment
SUPPORTED_FORMATS    = {".mp4", ".mov", ".webm", ".mkv"}
MAX_FILE_SIZE_MB     = 500


# ── Validation ────────────────────────────────────────────────────────────────

def validate_video_file(video_path: str) -> list:
    p = Path(video_path)
    if not p.exists():
        return [f"File not found: {video_path}"]
    errors = []
    ext = p.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        errors.append(
            f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File too large ({size_mb:.0f} MB). Maximum: {MAX_FILE_SIZE_MB} MB."
        )
    return errors


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Strips audio from video using FFmpeg and writes a 16kHz mono WAV.
    Uses imageio-ffmpeg so FFmpeg works on Windows without PATH setup.
    Returns: absolute path to audio.wav
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        raise ImportError("Run:  pip install imageio-ffmpeg")

    video_path = str(Path(video_path).resolve())
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(output_dir / "audio.wav")

    log.info("stage1.extract_audio.start", video=video_path, output=wav_path)

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-i",      video_path,
            "-vn",                   # no video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar",     "16000",      # 16 kHz
            "-ac",     "1",          # mono
            wav_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")
    if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
        raise RuntimeError(f"FFmpeg ran but produced no file at: {wav_path}")

    log.info("stage1.extract_audio.done",
             wav=wav_path, size_kb=Path(wav_path).stat().st_size // 1024)
    return wav_path


def get_audio_duration(wav_path: str) -> float:
    """Returns duration of a WAV file in seconds."""
    import soundfile as sf
    info = sf.info(wav_path)
    return info.frames / info.samplerate


# ── Speaker diarization ───────────────────────────────────────────────────────

def run_diarization(
    wav_path:     str,
    output_dir:   str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list:
    """
    Runs pyannote.audio speaker diarization on the extracted WAV.
    Returns [{speaker, start, end}, ...] sorted by start time.

    Tip: pass num_speakers if you know the board size in advance.
    Results cached to diarization.json so reruns skip this step.
    """
    cache_path = Path(output_dir) / "diarization.json"
    if cache_path.exists():
        log.info("stage1.diarization.cache_hit")
        return json.loads(cache_path.read_text())

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set.\n"
            "  1. Go to https://huggingface.co (free account)\n"
            "  2. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. Create token:   https://huggingface.co/settings/tokens\n"
            "  4. Add to .env:    HF_TOKEN=hf_..."
        )

    try:
        import torch
        from pyannote.audio import Pipeline as DiarizationPipeline
    except ImportError as e:
        raise ImportError(f"{e}\nRun:  pip install pyannote.audio torch torchaudio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("stage1.diarization.start", wav=wav_path, device=device)

    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(torch.device(device))

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    elif min_speakers is not None or max_speakers is not None:
        if min_speakers: kwargs["min_speakers"] = min_speakers
        if max_speakers: kwargs["max_speakers"] = max_speakers

    diarization = pipeline(str(Path(wav_path).resolve()), **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start":   round(turn.start, 3),
            "end":     round(turn.end,   3),
        })
    segments.sort(key=lambda x: x["start"])

    cache_path.write_text(json.dumps(segments, indent=2))
    log.info("stage1.diarization.done", raw_segments=len(segments))
    gc.collect()
    return segments


# ── Segment cleanup (3-pass improved) ────────────────────────────────────────

def merge_short_segments(
    segments:     list,
    min_duration: float = MIN_SEGMENT_DURATION,
    max_gap:      float = MAX_MERGE_GAP,
    orphan_gap:   float = ORPHAN_GAP_THRESHOLD,
) -> list:
    """
    Cleans up raw diarization output in three passes.

    Pass 1 — Standard merge
      Consecutive same-speaker segments with gap < max_gap get joined.
      max_gap = 0.6s handles natural pauses mid-sentence in meetings.

    Pass 2 — Orphan reassignment
      A short segment (< min_duration) sandwiched between turns of the SAME
      other speaker gets reassigned to that speaker.

      Example from real data:
        SPEAKER_02 (20s) → SPEAKER_02 "are" (0.44s) → SPEAKER_02 (5s)
        The 0.44s fragment is an orphan and gets reassigned to SPEAKER_02.

      This fixes misattributed fragments like "are", "What do you...", etc.

    Pass 3 — Re-merge
      After reassignment, orphans may now be adjacent to their new speaker's
      turns — merge again to produce clean output.

    Final — drop any remaining segments shorter than min_duration.
    """
    if not segments:
        return []

    # ── Pass 1: standard gap-based merge ─────────────────────────────────────
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap  = seg["start"] - last["end"]
        if seg["speaker"] == last["speaker"] and gap < max_gap:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # ── Pass 2: orphan reassignment ───────────────────────────────────────────
    result = []
    for i, seg in enumerate(merged):
        dur = seg["end"] - seg["start"]
        seg = seg.copy()

        if dur < min_duration:
            prev     = merged[i - 1] if i > 0             else None
            nxt      = merged[i + 1] if i < len(merged)-1 else None
            prev_gap = seg["start"] - prev["end"] if prev else 999
            next_gap = nxt["start"] - seg["end"]  if nxt  else 999

            # Sandwiched between the same speaker → reassign
            if (prev and nxt
                    and prev["speaker"] == nxt["speaker"]
                    and prev_gap < orphan_gap
                    and next_gap < orphan_gap):
                seg["speaker"]    = prev["speaker"]
                seg["reassigned"] = True

            # Trailing orphan right after a speaker turn
            elif prev and prev_gap < orphan_gap:
                seg["speaker"]    = prev["speaker"]
                seg["reassigned"] = True

        result.append(seg)

    # ── Pass 3: re-merge after reassignment ───────────────────────────────────
    final = [result[0].copy()]
    for seg in result[1:]:
        last = final[-1]
        gap  = seg["start"] - last["end"]
        if seg["speaker"] == last["speaker"] and gap < max_gap:
            last["end"] = seg["end"]
        else:
            final.append(seg.copy())

    # ── Final filter ──────────────────────────────────────────────────────────
    clean = [s for s in final if (s["end"] - s["start"]) >= min_duration]

    reassigned = sum(1 for s in result if s.get("reassigned"))
    log.info("stage1.merge_complete",
             raw=len(segments), after_merge=len(clean),
             dropped=len(segments) - len(clean),
             reassigned=reassigned)
    return clean


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    video_path:   str,
    output_dir:   str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> dict:
    """
    Full Stage 1 pipeline: validate → extract → diarize → clean.

    Returns:
        {
          "wav_path":      str,
          "segments":      [{speaker, start, end}, ...],
          "speaker_count": int,
          "speaker_ids":   [str, ...],
          "duration_sec":  float,
        }
    """
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    errors = validate_video_file(video_path)
    if errors:
        raise ValueError("Video validation failed:\n" +
                         "\n".join(f"  • {e}" for e in errors))

    wav_path  = extract_audio(video_path, output_dir)
    duration  = get_audio_duration(wav_path)

    raw_segs  = run_diarization(
        wav_path, output_dir,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    segments    = merge_short_segments(raw_segs)
    speaker_ids = sorted({s["speaker"] for s in segments})

    log.info("stage1.complete",
             segments=len(segments), speakers=len(speaker_ids),
             duration_sec=round(duration, 1))

    return {
        "wav_path":      wav_path,
        "segments":      segments,
        "speaker_count": len(speaker_ids),
        "speaker_ids":   speaker_ids,
        "duration_sec":  round(duration, 1),
    }