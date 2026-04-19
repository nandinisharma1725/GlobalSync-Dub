"""
test_stage1_stage2.py

Demo runner — Stage 1 + Stage 2 together.
Extracts audio, diarizes speakers, transcribes each speaker turn.

Usage:
    # Full run (needs HF_TOKEN for diarization)
    python test_stage1_stage2.py --input meeting.mp4

    # Skip diarization (no HF_TOKEN needed, whole video = one speaker)
    python test_stage1_stage2.py --input meeting.mp4 --skip-diarization

    # Use a faster/slower Whisper model
    python test_stage1_stage2.py --input meeting.mp4 --model tiny
    python test_stage1_stage2.py --input meeting.mp4 --model small

    # Already ran Stage 1? Pass its output dir to skip re-diarizing
    python test_stage1_stage2.py --input meeting.mp4 --output ./stage1_output/abc123

Available Whisper models (--model):
    tiny    fastest (~5x real-time on CPU), lower accuracy
    base    good balance, recommended for testing  [default]
    small   better accuracy, ~2x slower than base
    medium  near-production accuracy, slow on CPU
    large   best accuracy, very slow on CPU (use GPU)
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


# ── Colour helpers ────────────────────────────────────────────────────────────
def _color_ok():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return getattr(sys.stdout, "isatty", lambda: False)()

_C = _color_ok()
def _c(t, code): return f"\033[{code}m{t}\033[0m" if _C else t
def green(t):  return _c(t, "32")
def blue(t):   return _c(t, "34")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")
def cyan(t):   return _c(t, "36")


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_dur(s):
    s = int(s)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:   return f"{h}h {m}m {s}s"
    if m:   return f"{m}m {s}s"
    return  f"{s}s"

def step(n, t, label):
    print(f"\n  {blue(f'[{n}/{t}]')} {label}")

def ok(msg):   print(f"         {green('✓')} {msg}")
def warn(msg): print(f"         {yellow('⚠')} {msg}")
def err(msg):  print(f"         {red('✗')} {msg}")

def header(title):
    bar = "━" * 58
    print(f"\n{bold(bar)}\n  {bold(title)}\n{bold(bar)}\n")


# ── Skip-diarization mode ─────────────────────────────────────────────────────
def fake_diarization(wav_path, output_dir):
    """No pyannote — whole audio = one speaker segment."""
    import soundfile as sf
    info = sf.info(wav_path)
    dur  = info.frames / info.samplerate
    segs = [{"speaker": "SPEAKER_00", "start": 0.0, "end": round(dur, 3)}]
    (Path(output_dir) / "diarization.json").write_text(
        json.dumps(segs, indent=2))
    return segs


# ── Transcript printer ────────────────────────────────────────────────────────
SPEAKER_COLORS = [blue, cyan, yellow, green]

def print_transcript(segments, speaker_ids):
    """Prints a readable transcript with speaker labels."""
    color_map = {sid: SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
                 for i, sid in enumerate(sorted(speaker_ids))}

    print(f"\n  {bold('── Transcript ──────────────────────────────────────')}\n")
    for seg in segments:
        spk   = seg["speaker"]
        cfn   = color_map.get(spk, dim)
        ts    = f"{seg['start']:.1f}s"
        label = cfn(f"  {spk:<14}")
        print(f"{label} {dim(f'[{ts}]')}  {seg['text']}")
    print()


def print_stage2_summary(result, stage1, elapsed):
    segs     = result["segments"]
    spk_ids  = sorted({s["speaker"] for s in segs})

    print(f"\n  {bold('── Stage 2 result ───────────────────────────────────')}\n")
    print(f"  {'Segments transcribed:':<24} {bold(str(len(segs)))}")
    print(f"  {'Speakers:':<24} {bold(str(len(spk_ids)))}")

    # Per-speaker word count
    for sid in spk_ids:
        my_segs   = [s for s in segs if s["speaker"] == sid]
        word_count = sum(len(s["text"].split()) for s in my_segs)
        talk_time  = sum(s["end"] - s["start"] for s in my_segs)
        print(f"\n  {blue(sid)}")
        print(f"    Turns: {len(my_segs)}  "
              f"Words: ~{word_count}  "
              f"Speaking time: {fmt_dur(talk_time)}")
        # Preview first turn
        if my_segs:
            preview = my_segs[0]["text"][:80]
            if len(my_segs[0]["text"]) > 80:
                preview += "…"
            print(f"    First turn: {dim(preview)}")

    out_dir = Path(stage1["wav_path"]).parent
    print(f"\n  {'Output folder:':<24} {dim(str(out_dir))}")
    print(f"  {'transcription.json:':<24} {green('saved')}")
    print()
    print(f"  Completed in {fmt_dur(elapsed)}")
    print()
    print(bold("━" * 58))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 + Stage 2 demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to video file (MP4/MOV/WebM/MKV)")
    parser.add_argument("--output", "-o", default="./pipeline_output",
                        help="Output directory (default: ./pipeline_output)")
    parser.add_argument("--model",  "-m", default="base",
                        choices=["tiny","base","small","medium","large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--skip-diarization", action="store_true",
                        help="Skip pyannote — treat whole video as one speaker.\n"
                             "Use when HF_TOKEN isn't set yet.")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Tell pyannote exactly how many speakers to find.\n"
                             "Improves accuracy for large meetings.")
    parser.add_argument("--language", default="en",
                        help="Source language ISO code (default: en)")
    parser.add_argument("--job-id",
                        help="Job ID for output folder name (auto if not set)")
    args = parser.parse_args()

    header("MNC Dubbing — Stage 1 + Stage 2")

    video_path = str(Path(args.input).resolve())
    job_id     = args.job_id or str(uuid.uuid4())[:8]
    output_dir = str(Path(args.output).resolve() / job_id)
    t_total    = time.time()

    print(f"  Input:    {dim(video_path)}")
    print(f"  Output:   {dim(output_dir)}")
    print(f"  Whisper:  {dim(args.model)}")
    print(f"  Job ID:   {dim(job_id)}")
    if args.skip_diarization:
        print(f"  Mode:     {yellow('skip-diarization')} — single speaker, no HF token needed")
    if args.num_speakers:
        print(f"  Speakers: {dim(str(args.num_speakers))} (hint passed to pyannote)")

    # ── Import modules ────────────────────────────────────────────────────────
    here = Path(__file__).parent
    sys.path.insert(0, str(here))

    try:
        import stage1_extract
    except ImportError as e:
        err(f"Cannot import stage1_extract.py: {e}")
        sys.exit(1)

    try:
        import stage2_transcribe
    except ImportError as e:
        err(f"Cannot import stage2_transcribe.py: {e}")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    #  STAGE 1
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {bold('── STAGE 1: Audio Extraction & Diarization ─────────')}")
    t1 = time.time()

    # ── Step 1: Validate ──────────────────────────────────────────────────
    step(1, 3, "Validating video file…")
    errors = stage1_extract.validate_video_file(video_path)
    if errors:
        for e in errors:
            err(e)
        sys.exit(1)
    size_mb = Path(video_path).stat().st_size / 1e6
    ok(f"{Path(video_path).name}  ({size_mb:.1f} MB)  {green('valid')}")

    # ── Step 2: Extract audio ─────────────────────────────────────────────
    step(2, 3, "Extracting audio with FFmpeg (16kHz mono WAV)…")
    try:
        wav_path = stage1_extract.extract_audio(video_path, output_dir)
        duration = stage1_extract.get_audio_duration(wav_path)
        wav_mb   = Path(wav_path).stat().st_size / 1e6
        ok(f"audio.wav  ({wav_mb:.1f} MB  ·  {fmt_dur(duration)})")
    except Exception as e:
        err(str(e))
        sys.exit(1)

    # ── Step 3: Diarize ───────────────────────────────────────────────────
    if args.skip_diarization:
        step(3, 3, "Speaker diarization… (skipped — single-speaker mode)")
        raw_segs = fake_diarization(wav_path, output_dir)
        warn("pyannote skipped. All audio attributed to SPEAKER_00.")
    else:
        hf = os.environ.get("HF_TOKEN", "").strip()
        if not hf:
            step(3, 3, "Speaker diarization…")
            err("HF_TOKEN not set in .env")
            print(f"\n  {yellow('To enable diarization:')}")
            print("  1. Go to https://huggingface.co → create free account")
            print("  2. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  3. Create token:   https://huggingface.co/settings/tokens")
            print("  4. Add to .env:    HF_TOKEN=hf_...")
            print(f"\n  {yellow('Tip:')} use --skip-diarization to test without a token.")
            sys.exit(1)

        step(3, 3, "Detecting speakers with pyannote.audio…")
        print(f"         {dim('First run downloads ~1 GB model — takes a few minutes.')}")
        try:
            raw_segs = stage1_extract.run_diarization(
                wav_path, output_dir,
                # num_speakers=args.num_speakers,
            )
        except Exception as e:
            err(str(e))
            sys.exit(1)

    segs        = stage1_extract.merge_short_segments(raw_segs)
    speaker_ids = sorted({s["speaker"] for s in segs})
    ok(f"{len(segs)} segments  ·  {len(speaker_ids)} speaker(s): "
       f"{', '.join(speaker_ids)}")

    stage1_result = {
        "wav_path":      wav_path,
        "segments":      segs,
        "speaker_count": len(speaker_ids),
        "speaker_ids":   speaker_ids,
        "duration_sec":  round(duration, 1),
    }
    (Path(output_dir) / "stage1_result.json").write_text(
        json.dumps(stage1_result, indent=2))

    s1_elapsed = time.time() - t1
    print(f"\n  {green('Stage 1 done')} in {fmt_dur(s1_elapsed)}")

    # ════════════════════════════════════════════════════════════════
    #  STAGE 2
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {bold('── STAGE 2: Whisper Transcription ───────────────────')}")

    model_sizes = {"tiny": "39 MB", "base": "74 MB", "small": "244 MB",
                   "medium": "769 MB", "large": "1.5 GB"}
    print(f"  Model: {bold(args.model)} ({model_sizes.get(args.model, '?')})  "
          f"  Language: {bold(args.language)}")
    print(f"  {dim('Audio is passed as numpy arrays — no ffmpeg subprocess (Windows fix).')}")

    t2 = time.time()
    try:
        stage2_result = stage2_transcribe.run(
            stage1_result   = stage1_result,
            output_dir      = output_dir,
            source_language = args.language,
            model_name      = args.model,
        )
    except Exception as e:
        err(f"Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    s2_elapsed = time.time() - t2
    ok(f"{len(stage2_result['segments'])} segments transcribed "
       f"in {fmt_dur(s2_elapsed)}")

    # ── Print transcript ──────────────────────────────────────────────────
    if stage2_result["segments"]:
        print_transcript(stage2_result["segments"], speaker_ids)
    else:
        warn("No segments were transcribed. "
             "Check that the video has clear speech.")

    # ── Summary ───────────────────────────────────────────────────────────
    print_stage2_summary(stage2_result, stage1_result, time.time() - t_total)


if __name__ == "__main__":
    main()