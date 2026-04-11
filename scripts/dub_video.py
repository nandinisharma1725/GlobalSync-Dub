#!/usr/bin/env python3
"""
scripts/dub_video.py

CLI tool for testing the dubbing pipeline directly.
No API, no Celery — just runs the pipeline synchronously.

Usage:
  python dub_video.py --input meeting.mp4 --language hi --output ./output
  python dub_video.py --input meeting.mp4 --language es --output ./output --skip-diarization
  python dub_video.py --list-languages
"""

import argparse
import sys
import os
import uuid
from pathlib import Path

# Add parent directory to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import structlog
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ]
)

from backend.pipeline import stage1_extract, stage2_transcribe, stage3_translate, stage4_tts, stage5_sync
from backend.pipeline.translate import LANGUAGE_NAMES

log = structlog.get_logger()


def list_languages():
    print("\nSupported target languages:")
    print(f"{'Code':<8} {'Language'}")
    print("-" * 28)
    for code, name in sorted(LANGUAGE_NAMES.items(), key=lambda x: x[1]):
        print(f"{code:<8} {name}")


def main():
    parser = argparse.ArgumentParser(
        description="MNC Meeting Dubbing — CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Path to input MP4 video")
    parser.add_argument("--language", "-l", help="Target language code (e.g. hi, es, fr)")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--list-languages", action="store_true", help="List supported languages")
    parser.add_argument("--job-id", help="Job ID (auto-generated if not set)")
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip speaker diarization (treat entire audio as one speaker). Faster for testing.",
    )
    args = parser.parse_args()

    if args.list_languages:
        list_languages()
        return

    if not args.input:
        parser.error("--input is required")
    if not args.language:
        parser.error("--language is required")

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    if args.language not in LANGUAGE_NAMES:
        print(f"Error: Unsupported language '{args.language}'")
        list_languages()
        sys.exit(1)

    job_id = args.job_id or str(uuid.uuid4())[:8]
    output_dir = str(Path(args.output) / job_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  MNC Meeting Dubbing")
    print(f"{'='*50}")
    print(f"  Input:    {args.input}")
    print(f"  Language: {LANGUAGE_NAMES[args.language]} ({args.language})")
    print(f"  Output:   {output_dir}")
    print(f"  Job ID:   {job_id}")
    print(f"{'='*50}\n")

    # ── Stage 1 ──────────────────────────────────────────────────────
    print("[1/5] Extracting audio & diarizing speakers...")
    if args.skip_diarization:
        # Create a single "SPEAKER_00" segment for the whole video
        from backend.pipeline.sync import get_video_duration
        duration = get_video_duration(args.input)
        wav_path = stage1_extract.extract_audio(args.input, output_dir)
        stage1 = {
            "wav_path": wav_path,
            "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}],
            "speaker_count": 1,
            "speaker_ids": ["SPEAKER_00"],
        }
    else:
        stage1 = stage1_extract.run(args.input, output_dir)

    print(f"     → {stage1['speaker_count']} speaker(s), {len(stage1['segments'])} segments\n")

    # ── Stage 2 ──────────────────────────────────────────────────────
    print("[2/5] Transcribing with Whisper...")
    stage2 = stage2_transcribe.run(stage1, output_dir)
    print(f"     → {len(stage2['segments'])} segments transcribed\n")

    # ── Stage 3 ──────────────────────────────────────────────────────
    print(f"[3/5] Translating to {LANGUAGE_NAMES[args.language]}...")
    stage3 = stage3_translate.run(stage2, output_dir, args.language)
    print(f"     → {len(stage3['segments'])} segments translated\n")

    # ── Stage 4 ──────────────────────────────────────────────────────
    print("[4/5] Cloning voices & synthesizing speech...")
    stage4 = stage4_tts.run(stage3, stage1, output_dir, job_id)
    print(f"     → {len(stage4['voice_map'])} voice clone(s) created\n")

    # ── Stage 5 ──────────────────────────────────────────────────────
    print("[5/5] Syncing & exporting final video...")
    stage5 = stage5_sync.run(stage4, args.input, output_dir)

    output_path = stage5["output_video_path"]
    print(f"\n{'='*50}")
    print(f"  Done! Output saved to:")
    print(f"  {output_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()