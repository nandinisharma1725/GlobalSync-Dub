"""
test_stage1.py

Interactive test runner for Stage 1 — runs the full pipeline and
prints a clean summary. Use this to demo Stage 1 to someone.

Usage:
    # Full run with diarization (needs HF_TOKEN in .env)
    python test_stage1.py --input meeting.mp4

    # Quick test — skip diarization, treat whole video as one speaker
    python test_stage1.py --input meeting.mp4 --skip-diarization

    # Specify output folder
    python test_stage1.py --input meeting.mp4 --output ./my_output
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# ── Load .env if present ──────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# ── Colour helpers (Windows-safe) ─────────────────────────────────────────────
def _supports_color():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = _supports_color()

def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):  return c(t, "32")
def blue(t):   return c(t, "34")
def yellow(t): return c(t, "33")
def red(t):    return c(t, "31")
def bold(t):   return c(t, "1")
def dim(t):    return c(t, "2")


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_header():
    print()
    print(bold("━" * 55))
    print(bold("  MNC Dubbing — Stage 1: Audio Extraction & Diarization"))
    print(bold("━" * 55))
    print()

def print_step(n, total, label):
    print(f"  {blue(f'[{n}/{total}]')} {label}")

def print_ok(msg):
    print(f"         {green('✓')} {msg}")

def print_warn(msg):
    print(f"         {yellow('⚠')} {msg}")

def print_err(msg):
    print(f"         {red('✗')} {msg}")

def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def print_result(result: dict):
    segs = result["segments"]
    spk  = result["speaker_ids"]

    print()
    print(bold("  ── Stage 1 result ──────────────────────────────────"))
    print()
    print(f"  {'Audio file:':<18} {dim(result['wav_path'])}")
    print(f"  {'Video duration:':<18} {format_duration(result['duration_sec'])}")
    print(f"  {'Speakers found:':<18} {bold(str(result['speaker_count']))}")
    print(f"  {'Total segments:':<18} {bold(str(len(segs)))}")
    print()

    # Per-speaker summary
    for sid in spk:
        speaker_segs = [s for s in segs if s["speaker"] == sid]
        total_talk   = sum(s["end"] - s["start"] for s in speaker_segs)
        print(f"  {blue(sid)}")
        print(f"    Turns: {len(speaker_segs)}   "
              f"Total speaking time: {format_duration(total_talk)}")
        # Show first 3 turns
        for seg in speaker_segs[:3]:
            dur = seg["end"] - seg["start"]
            time_str = f'{seg["start"]:7.2f}s → {seg["end"]:7.2f}s'
            print(f"    {dim(time_str)}  "
                  f"({dur:.1f}s)")
        if len(speaker_segs) > 3:
            print(f"    {dim(f'... and {len(speaker_segs) - 3} more turns')}")
        print()

    # Show output files
    out_dir = Path(result["wav_path"]).parent
    json_cache = out_dir / "diarization.json"
    print(f"  {'Output folder:':<18} {dim(str(out_dir))}")
    print(f"  {'WAV extracted:':<18} {green('yes')} — {dim(result['wav_path'])}")
    print(f"  {'JSON cache:':<18} {green('yes') if json_cache.exists() else yellow('not yet')} "
          f"— {dim(str(json_cache))}")
    print()
    print(bold("  ── Ready for Stage 2 (Whisper transcription) ────────"))
    print()


# ── Skip-diarization mode (fast demo / no HF token) ──────────────────────────

def fake_diarization(wav_path: str, output_dir: str) -> list[dict]:
    """
    Skips pyannote entirely.
    Treats the whole video as one speaker (SPEAKER_00).
    Used when --skip-diarization is passed, or HF_TOKEN isn't set yet.
    """
    import soundfile as sf
    info = sf.info(wav_path)
    duration = info.frames / info.samplerate
    segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": round(duration, 3)}]
    cache = Path(output_dir) / "diarization.json"
    cache.write_text(json.dumps(segments, indent=2))
    return segments


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test Stage 1: audio extraction + speaker diarization",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to video file (MP4, MOV, WebM, MKV)")
    parser.add_argument("--output", "-o", default="./stage1_output",
                        help="Output directory (default: ./stage1_output)")
    parser.add_argument("--skip-diarization", action="store_true",
                        help="Skip pyannote — treat whole video as one speaker.\n"
                             "Use this if you haven't set HF_TOKEN yet.")
    parser.add_argument("--job-id",
                        help="Job ID (auto-generated if not set)")
    args = parser.parse_args()

    print_header()

    video_path = str(Path(args.input).resolve())
    job_id     = args.job_id or str(uuid.uuid4())[:8]
    output_dir = str(Path(args.output).resolve() / job_id)

    print(f"  Input:    {dim(video_path)}")
    print(f"  Output:   {dim(output_dir)}")
    print(f"  Job ID:   {dim(job_id)}")
    if args.skip_diarization:
        print(f"  Mode:     {yellow('skip-diarization')} (single speaker, no HF token needed)")
    print()

    # ── Import stage1_extract from same directory ──────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import stage1_extract
    except ImportError as e:
        print_err(f"Could not import stage1_extract.py: {e}")
        print_err("Make sure stage1_extract.py is in the same folder as this script.")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Step 1: Validate ──────────────────────────────────────────────────
    print_step(1, 3, "Validating video file…")
    errors = stage1_extract.validate_video_file(video_path)
    if errors:
        for e in errors:
            print_err(e)
        sys.exit(1)
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    print_ok(f"{Path(video_path).name}  ({size_mb:.1f} MB)")
    print()

    # ── Step 2: Extract audio ─────────────────────────────────────────────
    print_step(2, 3, "Extracting audio (FFmpeg → 16kHz mono WAV)…")
    try:
        wav_path = stage1_extract.extract_audio(video_path, output_dir)
        duration = stage1_extract.get_audio_duration(wav_path)
        wav_mb   = Path(wav_path).stat().st_size / (1024 * 1024)
        print_ok(f"audio.wav  ({wav_mb:.1f} MB · {format_duration(duration)})")
    except Exception as e:
        print_err(str(e))
        sys.exit(1)
    print()

    # ── Step 3: Diarize ───────────────────────────────────────────────────
    if args.skip_diarization:
        print_step(3, 3, "Speaker diarization… (skipped — single-speaker mode)")
        raw_segments = fake_diarization(wav_path, output_dir)
        print_warn("pyannote skipped. All audio attributed to SPEAKER_00.")
    else:
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            print_step(3, 3, "Speaker diarization…")
            print_err("HF_TOKEN not set in .env")
            print()
            print(f"  {yellow('To enable diarization:')}")
            print(f"  1. Create a free account at https://huggingface.co")
            print(f"  2. Accept the model license at:")
            print(f"     https://huggingface.co/pyannote/speaker-diarization-3.1")
            print(f"  3. Generate a token at https://huggingface.co/settings/tokens")
            print(f"  4. Add to your .env:  HF_TOKEN=hf_...")
            print()
            print(f"  {yellow('Tip:')} use --skip-diarization to test without a token.")
            sys.exit(1)

        print_step(3, 3, "Speaker diarization (pyannote — this takes a while)…")
        print(f"         {dim('First run downloads ~1GB model weights. Subsequent runs use cache.')}")
        try:
            raw_segments = stage1_extract.run_diarization(wav_path, output_dir)
        except Exception as e:
            print_err(str(e))
            sys.exit(1)

    # ── Merge and clean segments ──────────────────────────────────────────
    segments    = stage1_extract.merge_short_segments(raw_segments)
    speaker_ids = sorted({s["speaker"] for s in segments})
    print_ok(f"{len(segments)} segments · {len(speaker_ids)} speaker(s)")
    print()

    # ── Build result dict ─────────────────────────────────────────────────
    result = {
        "wav_path":      wav_path,
        "segments":      segments,
        "speaker_count": len(speaker_ids),
        "speaker_ids":   speaker_ids,
        "duration_sec":  round(duration, 1),
    }

    # Save full result JSON
    result_path = Path(output_dir) / "stage1_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    elapsed = time.time() - t_start

    # ── Print summary ─────────────────────────────────────────────────────
    print_result(result)
    print(f"  Completed in {format_duration(elapsed)}")
    print()
    print(f"  {green('Stage 1 finished successfully.')} Output is in:")
    print(f"  {dim(output_dir)}")
    print()
    print(bold("━" * 55))
    print()


if __name__ == "__main__":
    main()