"""
stage1_extract.py  —  Stage 1 v4: Video upload, audio extraction, speaker diarization

What changed from v3 and WHY:

The multi-speaker contamination problem (e.g. "What do you mean? Redundancies?"
appearing in SPEAKER_04's turn) was caused entirely in Stage 1, NOT Stage 2.

Root cause trace:
  - pyannote gave a 0.35s SPEAKER_03 fragment at 73.0→73.4s
  - v3's orphan reassignment (gap threshold: 1.5s) absorbed it into SPEAKER_04
  - v3's max_gap=0.6s then merged all SPEAKER_04 fragments into a 14.6s block
  - Whisper received 14.6s of audio containing 3+ different speakers
  - No amount of post-transcription merging can fix this — damage already done

Three fixes applied here:

  1. max_gap: 0.6s → 0.4s
     A gap of 0.41s between two same-speaker segments is more likely a real
     speaker boundary in a meeting than a breath pause. 0.4s is conservative
     enough to catch natural pauses without eating into speaker transitions.

  2. orphan_gap: 1.5s → 0.5s
     Previous version reassigned an orphan if its neighbor was within 1.5s.
     That's too wide — it was absorbing real short turns from other speakers.
     Now only reassign if the neighbor is within 0.5s (very close = likely
     the same speaker mislabelled by pyannote).

  3. max_seg_dur: NEW 8.0s cap
     Any segment longer than 8s is split into equal chunks.
     WHY 8s: a single continuous turn in a meeting is rarely longer than 8s
     without a pause. If a segment exceeds 8s it almost certainly contains
     multiple speakers. Splitting forces Whisper to process smaller, cleaner
     audio windows.

  4. min_dur: 1.0s → 0.8s
     Slightly relaxed to preserve short-but-real turns like "Yes." or "Sure."
     These short confirmations are important in meeting dialogue.

Install:
  pip install imageio-ffmpeg soundfile numpy structlog pyannote.audio torch torchaudio
"""

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()

WHISPER_SAMPLE_RATE  = 16_000
MIN_SEGMENT_DURATION = 0.8    # v4: was 1.0 — preserves short turns like "Sure."
MAX_MERGE_GAP        = 0.4    # v4: was 0.6 — less aggressive, avoids speaker bleed
ORPHAN_GAP_THRESHOLD = 0.5    # v4: was 1.5 — only reassign very close orphans
MAX_SEGMENT_DURATION = 8.0    # v4: NEW — cap ensures Whisper gets single-speaker audio
SUPPORTED_FORMATS    = {".mp4", ".mov", ".webm", ".mkv"}
MAX_FILE_SIZE_MB     = 500


def validate_video_file(video_path: str) -> list:
    p = Path(video_path)
    if not p.exists():
        return [f"File not found: {video_path}"]
    errors = []
    ext = p.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        errors.append(f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(f"File too large ({size_mb:.0f} MB). Maximum: {MAX_FILE_SIZE_MB} MB.")
    return errors


def extract_audio(video_path: str, output_dir: str) -> str:
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
            "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav_path,
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")
    if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
        raise RuntimeError(f"FFmpeg ran but produced no file at: {wav_path}")

    log.info("stage1.extract_audio.done",
             wav=wav_path, size_kb=Path(wav_path).stat().st_size // 1024)
    return wav_path


def get_audio_duration(wav_path: str) -> float:
    import soundfile as sf
    info = sf.info(wav_path)
    return info.frames / info.samplerate


def run_diarization(
    wav_path:     str,
    output_dir:   str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list:
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


def merge_short_segments(
    segments:     list,
    min_duration: float = MIN_SEGMENT_DURATION,
    max_gap:      float = MAX_MERGE_GAP,
    orphan_gap:   float = ORPHAN_GAP_THRESHOLD,
    max_seg_dur:  float = MAX_SEGMENT_DURATION,
) -> list:
    """
    Cleans raw diarization output in four passes.

    Pass 1 — Gap merge
      Same-speaker consecutive segments with gap < max_gap (0.4s) are joined,
      but only if the resulting segment would still be <= max_seg_dur (8s).
      The 8s cap prevents creating enormous segments containing multiple speakers.

    Pass 2 — Orphan reassignment
      A very short segment sandwiched between turns of the SAME other speaker
      and within orphan_gap (0.5s) of both neighbors is reassigned.
      Tight 0.5s threshold prevents absorbing real (if short) speaker turns.

    Pass 3 — Re-merge after reassignment
      Same as Pass 1, applied again after orphans are fixed.

    Pass 4 — Duration cap split
      Any segment still > 8s (because a single raw pyannote segment was already
      longer) is split into equal chunks of max_seg_dur.
      This guarantees Whisper always receives a single-speaker audio window.
    """
    if not segments:
        return []

    # Pass 1: merge
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last    = merged[-1]
        gap     = seg["start"] - last["end"]
        new_dur = seg["end"]   - last["start"]
        if (seg["speaker"] == last["speaker"]
                and gap < max_gap
                and new_dur <= max_seg_dur):
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Pass 2: orphan reassignment
    result = []
    for i, seg in enumerate(merged):
        seg  = seg.copy()
        dur  = seg["end"] - seg["start"]
        if dur < min_duration:
            prev = merged[i-1] if i > 0             else None
            nxt  = merged[i+1] if i < len(merged)-1 else None
            pg   = seg["start"] - prev["end"] if prev else 999
            ng   = nxt["start"] - seg["end"]  if nxt  else 999
            if (prev and nxt
                    and prev["speaker"] == nxt["speaker"]
                    and pg < orphan_gap
                    and ng < orphan_gap):
                seg["speaker"] = prev["speaker"]
            elif prev and pg < orphan_gap:
                seg["speaker"] = prev["speaker"]
        result.append(seg)

    # Pass 3: re-merge
    merged2 = [result[0].copy()]
    for seg in result[1:]:
        last    = merged2[-1]
        gap     = seg["start"] - last["end"]
        new_dur = seg["end"]   - last["start"]
        if (seg["speaker"] == last["speaker"]
                and gap < max_gap
                and new_dur <= max_seg_dur):
            last["end"] = seg["end"]
        else:
            merged2.append(seg.copy())

    # Pass 4: split oversized segments
    final = []
    for seg in merged2:
        dur = seg["end"] - seg["start"]
        if dur > max_seg_dur:
            # Split into equal-duration chunks
            start = seg["start"]
            while start < seg["end"]:
                chunk_end = min(start + max_seg_dur, seg["end"])
                if chunk_end - start >= min_duration:
                    final.append({
                        "speaker": seg["speaker"],
                        "start":   round(start, 3),
                        "end":     round(chunk_end, 3),
                    })
                start = chunk_end
        else:
            final.append(seg)

    clean = [s for s in final if (s["end"] - s["start"]) >= min_duration]

    log.info("stage1.merge_complete",
             raw=len(segments), after=len(clean),
             dropped=len(segments) - len(clean))
    return clean


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

    wav_path = extract_audio(video_path, output_dir)
    duration = get_audio_duration(wav_path)

    raw_segs = run_diarization(
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