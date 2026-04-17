"""
backend/pipeline/tts.py

Stage 4 — Text-to-Speech

This stage is responsible for synthesizing translated segments into WAV
files that Stage 5 will time-stretch and mux into the final video. The
real implementation typically uses ElevenLabs (configured via
`elevenlabs_api_key`). To avoid crashes when the provider/key is missing
this module returns the expected structure and records diagnostic
messages so the pipeline can continue.
"""

import json
from pathlib import Path
import structlog
from typing import Dict
import time

import httpx

from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()


def _synthesize_local(text: str, out_path: str) -> tuple[bool, str]:
    """Try offline synthesis with pyttsx3. Returns (success, error_or_empty)."""
    try:
        import pyttsx3
    except Exception as e:
        return False, f"pyttsx3 not available: {e}"

    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return True, ""
    except Exception as e:
        return False, str(e)


def run(
    stage3_result: dict,
    stage1_result: dict,
    output_dir: str,
    job_id: str,
) -> dict:
    """
    Main entry point for Stage 4 (TTS).

    Args:
        stage3_result: Output from translate.run()
        stage1_result: Output from extract.run()
        output_dir: Working directory for this job
        job_id: Unique job identifier (used for provider cleanup)

    Returns:
        {
          "target_language": "hi",
          "voice_map": { "SPEAKER_00": "..." },
          "segments": [ { ...segment fields..., "dubbed_audio_path": "..." }, ... ]
        }
    """
    lang = stage3_result.get("target_language")
    cache_path = Path(output_dir) / f"tts_{lang}.json"
    if cache_path.exists():
        log.info("stage4.cache_hit", lang=lang)
        return json.loads(cache_path.read_text(encoding='utf-8'))

    segments = stage3_result.get("segments", [])
    tts_segments = []

    dubbed_dir = Path(output_dir) / f"dubbed_{lang}"
    dubbed_dir.mkdir(parents=True, exist_ok=True)

    API_BASE = "https://api.elevenlabs.io/v1"

    def _choose_voice_id(api_key: str) -> str | None:
        """Return a usable voice_id from the ElevenLabs account, or None."""
        try:
            headers = {"xi-api-key": api_key}
            r = httpx.get(f"{API_BASE}/voices", headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()

            # API may return {"voices": [...]}
            voices = []
            if isinstance(data, dict) and "voices" in data and isinstance(data["voices"], list):
                voices = data["voices"]
            elif isinstance(data, list):
                voices = data
            elif isinstance(data, dict):
                # Sometimes the endpoint returns a mapping of id->obj
                # pick the first key as voice id
                first_key = next(iter(data.keys()), None)
                if first_key and isinstance(data[first_key], dict):
                    return first_key

            if not voices:
                return None

            # Prefer a premade voice if available, else first voice
            for v in voices:
                # Voice may include 'voice_id' or 'id'
                vid = v.get("voice_id") or v.get("id") or v.get("voice_id")
                if vid:
                    return vid

            return None
        except Exception:
            return None

    def _synthesize(api_key: str, voice_id: str, text: str) -> tuple[bool, bytes | str]:
        """Call ElevenLabs TTS, return (success, content_or_error)."""
        try:
            headers = {
                "xi-api-key": api_key,
                "Accept": "audio/wav",
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "voice_settings": {"stability": 0.6, "similarity_boost": 0.75},
            }
            r = httpx.post(f"{API_BASE}/text-to-speech/{voice_id}", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return True, r.content
            return False, f"status_code: {r.status_code}, body: {r.text}"
        except Exception as e:
            return False, str(e)

    for seg in segments:
        seg_out: Dict = {
            "segment_id": seg.get("segment_id"),
            "speaker": seg.get("speaker"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
            "words": seg.get("words"),
            "chunk_path": seg.get("chunk_path"),
            "translated_text": seg.get("translated_text", seg.get("text")),
            "target_language": lang,
            # Filled in if synthesis succeeds; left None otherwise
            "dubbed_audio_path": None,
            # Diagnostic string when synthesis failed or skipped
            "tts_error": "",
        }

        # Basic sanity: if there's no ElevenLabs key, try an offline
        # fallback (pyttsx3). If that fails, record a helpful error.
        if not settings.elevenlabs_api_key:
            # Try local TTS first
            text = seg_out["translated_text"] or seg_out["text"] or ""
            if not text.strip():
                seg_out["tts_error"] = "Empty text — skipping synthesis."
            else:
                out_file = str(dubbed_dir / f"dubbed_{int(seg.get('segment_id', len(tts_segments))):04d}.wav")
                ok, err = _synthesize_local(text, out_file)
                if ok:
                    seg_out["dubbed_audio_path"] = out_file
                    seg_out["tts_error"] = ""
                else:
                    seg_out["tts_error"] = f"Local TTS failed: {err}. Provide elevenlabs_api_key or install pyttsx3."
        else:
            voice_id = _choose_voice_id(settings.elevenlabs_api_key)
            if not voice_id:
                seg_out["tts_error"] = "No usable ElevenLabs voice found for this API key."
            else:
                text = seg_out["translated_text"] or seg_out["text"] or ""
                if not text.strip():
                    seg_out["tts_error"] = "Empty text — skipping synthesis."
                else:
                    ok, result = _synthesize(settings.elevenlabs_api_key, voice_id, text)
                    if not ok:
                        seg_out["tts_error"] = str(result)
                    else:
                        # Write WAV
                        idx = seg.get("segment_id") if seg.get("segment_id") is not None else len(tts_segments)
                        out_path = str(dubbed_dir / f"dubbed_{int(idx):04d}.wav")
                        try:
                            with open(out_path, "wb") as fh:
                                fh.write(result)
                            seg_out["dubbed_audio_path"] = out_path
                            seg_out["tts_error"] = ""
                        except Exception as e:
                            seg_out["tts_error"] = f"write_failed: {e}"

                    # Be a little polite to the API
                    time.sleep(0.2)

        tts_segments.append(seg_out)

    output = {
        "target_language": lang,
        "voice_map": {},
        "segments": tts_segments,
    }

    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    log.info("stage4.complete", lang=lang, segments=len(tts_segments))
    return output


def cleanup_cloned_voices(job_id: str, voice_map: dict) -> None:
    """
    Cleanup provider-side cloned voices (no-op if none).
    """
    if not voice_map:
        return
    try:
        log.info("stage4.cleanup", job_id=job_id, voices=list(voice_map.values()))
        # Real implementation would call ElevenLabs to delete cloned voices.
    except Exception as e:
        log.warning("stage4.cleanup_failed", error=str(e))