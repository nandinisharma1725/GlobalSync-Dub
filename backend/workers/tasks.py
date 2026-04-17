"""
backend/workers/tasks.py

Background task runner — uses threading to process dubbing jobs without Redis/Celery.

Jobs are processed in background threads while the API responds immediately.
"""

import os
import json
from pathlib import Path
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import structlog

from ..utils.config import get_settings
from ..utils.storage import get_local_path
from ..pipeline.orchestrator import run_pipeline

log = structlog.get_logger()
settings = get_settings()

# ── Thread Pool Executor ──────────────────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=2)


# ── In-memory progress store ──────────────────────────────────────────────────
# In production, replace this with Redis pub/sub or a DB column.
_progress_store: dict[str, dict] = {}


def set_job_progress(job_id: str, status: str, percent: int, error: str = ""):
    _progress_store[job_id] = {
        "status": status,
        "percent": percent,
        "error": error,
    }
    # Also write to disk for persistence across worker restarts
    progress_path = Path(settings.local_storage_path) / job_id / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(_progress_store[job_id]))


def get_job_progress(job_id: str) -> dict:
    # Try memory first
    if job_id in _progress_store:
        return _progress_store[job_id]
    # Fall back to disk
    progress_path = Path(settings.local_storage_path) / job_id / "progress.json"
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    return {"status": "unknown", "percent": 0, "error": ""}


# ── Tasks ─────────────────────────────────────────────────────────────────────

def dub_video(
    job_id: str,
    video_path: str,
    target_language: str,
):
    """
    Main dubbing task. Called by the API after upload.

    Args:
        job_id: Unique identifier for this job
        video_path: Local path to the uploaded video
        target_language: Target language ISO code e.g. "hi"
    """
    output_dir = str(Path(settings.local_storage_path) / job_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log.info("task.dub_video.start", job_id=job_id, lang=target_language)
    set_job_progress(job_id, "starting", 2)

    def on_progress(status: str, pct: int):
        set_job_progress(job_id, status, pct)

    try:
        result = run_pipeline(
            job_id=job_id,
            video_path=video_path,
            target_language=target_language,
            output_dir=output_dir,
            on_progress=on_progress,
        )
        set_job_progress(job_id, "done", 100)
        log.info("task.dub_video.complete", job_id=job_id)
        return result

    except Exception as exc:
        error_msg = str(exc)
        log.error("task.dub_video.failed", job_id=job_id, error=error_msg)
        set_job_progress(job_id, "failed", 0, error=error_msg)
        raise


def start_dub_job(job_id: str, video_path: str, target_language: str):
    """Queue a dubbing job to run in the background thread pool."""
    return executor.submit(dub_video, job_id, video_path, target_language)