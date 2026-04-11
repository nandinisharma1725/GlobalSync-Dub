"""
backend/pipeline/orchestrator.py

Pipeline Orchestrator — runs all 5 stages in order.

This is what the Celery worker calls. Each stage:
  - Checks a cache file before running (so it can resume on failure)
  - Reports progress back via a callback

Progress states:
  extracting → transcribing → translating → synthesizing → syncing → done
"""

import json
from pathlib import Path
from typing import Callable, Optional
import structlog

from . import extract, transcribe, translate, tts, sync
from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()


def run_pipeline(
    job_id: str,
    video_path: str,
    target_language: str,
    output_dir: str,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    Runs the complete dubbing pipeline.

    Args:
        job_id: Unique job identifier
        video_path: Path to the original MP4
        target_language: ISO 639-1 code e.g. "hi"
        output_dir: Working directory for all artifacts
        on_progress: Callback(status: str, percent: int) called at each stage

    Returns:
        {
          "output_video_path": "...",
          "target_language": "hi",
          "job_id": "...",
        }
    """
    def progress(status: str, pct: int):
        log.info("pipeline.progress", job_id=job_id, status=status, pct=pct)
        if on_progress:
            on_progress(status, pct)

    progress("extracting", 5)
    stage1 = extract.run(
        video_path=video_path,
        output_dir=output_dir,
    )

    progress("transcribing", 20)
    stage2 = transcribe.run(
        stage1_result=stage1,
        output_dir=output_dir,
        source_language="en",
    )

    progress("translating", 40)
    stage3 = translate.run(
        stage2_result=stage2,
        output_dir=output_dir,
        target_language=target_language,
    )

    progress("synthesizing", 60)
    stage4 = tts.run(
        stage3_result=stage3,
        stage1_result=stage1,
        output_dir=output_dir,
        job_id=job_id,
    )

    progress("syncing", 85)
    stage5 = sync.run(
        stage4_result=stage4,
        video_path=video_path,
        output_dir=output_dir,
    )

    progress("done", 100)

    # Write a manifest so the API can serve the result
    manifest = {
        "job_id": job_id,
        "target_language": target_language,
        "output_video_path": stage5["output_video_path"],
        "speaker_count": stage1["speaker_count"],
        "segment_count": len(stage2["segments"]),
        "voice_map": stage4.get("voice_map", {}),
    }
    manifest_path = Path(output_dir) / f"manifest_{target_language}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Clean up cloned voices to avoid ElevenLabs storage charges
    try:
        tts.cleanup_cloned_voices(job_id, stage4.get("voice_map", {}))
    except Exception as e:
        log.warning("pipeline.cleanup_failed", error=str(e))

    return manifest