"""
backend/workers/tasks.py

Celery async tasks — runs the pipeline in the background.

Workers are separate processes from the API, so heavy GPU/CPU work
doesn't block the web server.

Start workers with:
  celery -A backend.workers.tasks worker --loglevel=info --concurrency=2
"""

import os
import json
from pathlib import Path
from celery import Celery
import structlog

from ..utils.config import get_settings
from ..utils.storage import get_local_path
from ..pipeline.orchestrator import run_pipeline

log = structlog.get_logger()
settings = get_settings()

# ── Celery App ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "mnc_dubbing",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,       # results expire after 24h
    task_track_started=True,
    worker_prefetch_multiplier=1,   # process one task at a time (GPU bottleneck)
)


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

@celery_app.task(
    bind=True,
    name="dub_video",
    max_retries=2,
    default_retry_delay=30,
)
def dub_video_task(
    self,
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
        # Update Celery task state for monitoring tools (Flower)
        self.update_state(state="PROGRESS", meta={"status": status, "percent": pct})

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
        raise self.retry(exc=exc)