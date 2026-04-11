"""
backend/api/main.py

FastAPI application — exposes the dubbing pipeline as a REST API.

Endpoints:
  POST /api/jobs              — Upload video, start dubbing job
  GET  /api/jobs/{id}         — Get job status & progress
  GET  /api/jobs/{id}/download/{filename} — Download dubbed video
  GET  /api/languages         — List supported languages
  GET  /health                — Health check
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import structlog

from ..utils.config import get_settings
from ..utils.storage import save_upload, get_download_url, get_local_path
from ..workers.tasks import dub_video_task, get_job_progress
from ..pipeline.translate import LANGUAGE_NAMES

log = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="MNC Dubbing API",
    description="AI-powered corporate meeting video dubbing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}


# ── Languages ─────────────────────────────────────────────────────────────────

@app.get("/api/languages")
async def list_languages():
    """Returns all supported target languages."""
    return [
        {"code": code, "name": name}
        for code, name in sorted(LANGUAGE_NAMES.items(), key=lambda x: x[1])
    ]


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.post("/api/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(..., description="MP4 video file"),
    target_language: str = Form(..., description="Target language ISO code e.g. 'hi'"),
):
    """
    Accepts a video upload and starts the dubbing pipeline asynchronously.

    Returns immediately with a job_id — poll GET /api/jobs/{id} for status.
    """
    # Validate file type
    if not video.filename.lower().endswith((".mp4", ".mov", ".webm", ".mkv")):
        raise HTTPException(400, "Only MP4, MOV, WebM, MKV files are supported.")

    # Validate language
    if target_language not in LANGUAGE_NAMES:
        raise HTTPException(
            400,
            f"Unsupported language '{target_language}'. "
            f"Supported: {list(LANGUAGE_NAMES.keys())}",
        )

    # Validate file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_bytes = await video.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            413,
            f"File too large. Maximum is {settings.max_upload_size_mb}MB.",
        )

    # Save upload
    job_id, video_path = await save_upload(file_bytes, video.filename)
    log.info("api.job_created", job_id=job_id, lang=target_language, size_mb=round(len(file_bytes) / 1e6, 1))

    # Dispatch to Celery worker
    dub_video_task.apply_async(
        args=[job_id, video_path, target_language],
        task_id=job_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "target_language": target_language,
        "language_name": LANGUAGE_NAMES[target_language],
        "poll_url": f"/api/jobs/{job_id}",
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Returns the current status of a dubbing job.

    Status values:
      queued → extracting → transcribing → translating → synthesizing → syncing → done | failed
    """
    progress = get_job_progress(job_id)

    if not progress or progress.get("status") == "unknown":
        raise HTTPException(404, f"Job '{job_id}' not found.")

    response = {
        "job_id": job_id,
        "status": progress["status"],
        "percent": progress["percent"],
    }

    if progress.get("error"):
        response["error"] = progress["error"]

    if progress["status"] == "done":
        # Look for output file
        output_dir = Path(settings.local_storage_path) / job_id
        # Find dubbed MP4 (language code is in the filename)
        output_files = list(output_dir.glob("dubbed_*.mp4"))
        if output_files:
            filename = output_files[0].name
            response["download_url"] = get_download_url(job_id, filename)
            response["filename"] = filename

    return response


@app.get("/api/jobs/{job_id}/download/{filename}")
async def download_dubbed_video(job_id: str, filename: str):
    """Serves the dubbed MP4 for download."""
    # Validate filename (prevent path traversal)
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")

    try:
        local_path = get_local_path(job_id, filename)
    except FileNotFoundError:
        raise HTTPException(404, "File not found.")

    return FileResponse(
        path=local_path,
        filename=filename,
        media_type="video/mp4",
    )