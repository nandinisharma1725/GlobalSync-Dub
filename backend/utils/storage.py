"""
backend/utils/storage.py
Abstraction over local disk and AWS S3.
Use STORAGE_BACKEND=local during dev, s3 in production.
"""
import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .config import get_settings

settings = get_settings()


def _local_path(job_id: str, filename: str) -> Path:
    p = Path(settings.local_storage_path) / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


async def save_upload(file_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """
    Saves an uploaded file.
    Returns (job_id, file_path_or_s3_key).
    """
    job_id = str(uuid.uuid4())
    ext = Path(original_filename).suffix
    filename = f"original{ext}"

    if settings.storage_backend == "s3":
        key = f"{job_id}/{filename}"
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        s3.put_object(Bucket=settings.s3_bucket_name, Key=key, Body=file_bytes)
        return job_id, key
    else:
        path = _local_path(job_id, filename)
        async with aiofiles.open(path, "wb") as f:
            await f.write(file_bytes)
        return job_id, str(path)


def get_local_path(job_id: str, filename: str) -> str:
    """Returns a local path, downloading from S3 first if needed."""
    local = _local_path(job_id, filename)
    if local.exists():
        return str(local)

    if settings.storage_backend == "s3":
        key = f"{job_id}/{filename}"
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(settings.s3_bucket_name, key, str(local))
        return str(local)

    raise FileNotFoundError(f"File not found: {job_id}/{filename}")


def save_artifact(job_id: str, filename: str, data: bytes) -> str:
    """Saves a pipeline artifact (audio chunk, dubbed file, etc.)."""
    path = _local_path(job_id, filename)
    path.write_bytes(data)

    if settings.storage_backend == "s3":
        key = f"{job_id}/{filename}"
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        s3.put_object(Bucket=settings.s3_bucket_name, Key=key, Body=data)

    return str(path)


def get_download_url(job_id: str, filename: str) -> str:
    """Returns a presigned S3 URL or a local API path."""
    if settings.storage_backend == "s3":
        key = f"{job_id}/{filename}"
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": key},
            ExpiresIn=3600,
        )
    return f"/api/jobs/{job_id}/download/{filename}"