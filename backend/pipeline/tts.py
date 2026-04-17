"""
backend/pipeline/tts.py

Stage 4 — Text-to-Speech

This stage is responsible for synthesizing translated segments into WAV
files that Stage 5 will time-stretch and mux into the final video. The
real implementation typically uses ElevenLabs (configured via
`elevenlabs_api_key`). When ElevenLabs fails (invalid key, quota
exhausted, network error) the module falls back to gTTS (Google
Text-to-Speech) and then to pyttsx3 for offline synthesis.
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


# ── Fallback TTS engines ─────────────────────────────────────────────────────

def _synthesize_gtts(text: str, lang: str, out_path: str) -> tuple[bool, str]:
    """Try synthesis with gTTS (Google Text-to-Speech).

    gTTS outputs MP3, so we convert to WAV via librosa + soundfile.
    Returns (success, error_or_empty).
    """
    try:
        from gtts import gTTS
    except ImportError:
        return False, "gTTS not installed (pip install gTTS)"

    try:
        import librosa
        import soundfile as sf
        import tempfile
        import os

        # gTTS only outputs MP3; write to temp, convert to WAV
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name

        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(tmp_mp3)

        # Convert MP3 → WAV (24 kHz mono to match pipeline sample rate)
        y, _ = librosa.load(tmp_mp3, sr=24_000, mono=True)
        sf.write(out_path, y, 24_000)

        # Clean up temp MP3
        try:
            os.unlink(tmp_mp3)
        except OSError:
            pass

        return True, ""
    except Exception as e:
        return False, str(e)


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


def _fallback_synthesize(text: str, lang: str, out_path: str) -> tuple[bool, str]:
    """Try all fallback TTS engines in order: gTTS → pyttsx3.

    Returns (success, combined_error_message).
    """
    errors = []

    # 1. Try gTTS (supports many languages including Hindi)
    ok, err = _synthesize_gtts(text, lang, out_path)
    if ok:
        return True, ""
    errors.append(f"gTTS: {err}")

    # 2. Try pyttsx3 (offline, but limited language support)
    ok, err = _synthesize_local(text, out_path)
    if ok:
        return True, ""
    errors.append(f"pyttsx3: {err}")

    return False, " | ".join(errors)


# ── ElevenLabs helpers ────────────────────────────────────────────────────────

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
            first_key = next(iter(data.keys()), None)
            if first_key and isinstance(data[first_key], dict):
                return first_key

        if not voices:
            return None

        # Prefer a premade voice if available, else first voice
        for v in voices:
            vid = v.get("voice_id") or v.get("id") or v.get("voice_id")
            if vid:
                return vid

        return None
    except Exception:
        return None


def _synthesize_elevenlabs(api_key: str, voice_id: str, text: str) -> tuple[bool, bytes | str]:
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


# ── Main entry point ──────────────────────────────────────────────────────────

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

    # Check cache — but only use it if the cached segments actually have audio
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        cached_segs = cached.get("segments", [])
        has_audio = any(
            s.get("dubbed_audio_path") and Path(s["dubbed_audio_path"]).exists()
            for s in cached_segs
        )
        if has_audio:
            log.info("stage4.cache_hit", lang=lang)
            return cached
        else:
            log.warning(
                "stage4.cache_stale",
                lang=lang,
                reason="cached segments have no audio files — re-synthesizing",
            )
            cache_path.unlink(missing_ok=True)

    segments = stage3_result.get("segments", [])
    tts_segments = []

    dubbed_dir = Path(output_dir) / f"dubbed_{lang}"
    dubbed_dir.mkdir(parents=True, exist_ok=True)

    # ── Attempt ElevenLabs first ──────────────────────────────────────────────
    use_elevenlabs = False
    voice_id = None
    elevenlabs_error = ""

    if settings.elevenlabs_api_key:
        voice_id = _choose_voice_id(settings.elevenlabs_api_key)
        if voice_id:
            use_elevenlabs = True
        else:
            elevenlabs_error = "No usable ElevenLabs voice found for this API key"
            log.warning("stage4.elevenlabs_unavailable", reason=elevenlabs_error)
    else:
        elevenlabs_error = "No ElevenLabs API key configured"
        log.info("stage4.no_elevenlabs_key", fallback="gTTS/pyttsx3")

    succeeded = 0
    failed = 0
    fallback_used = False

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
            "dubbed_audio_path": None,
            "tts_error": "",
        }

        text = seg_out["translated_text"] or seg_out["text"] or ""
        if not text.strip():
            seg_out["tts_error"] = "Empty text — skipping synthesis."
            log.warning("stage4.empty_text", segment=seg_out["segment_id"])
            failed += 1
            tts_segments.append(seg_out)
            continue

        idx = seg.get("segment_id") if seg.get("segment_id") is not None else len(tts_segments)
        out_path = str(dubbed_dir / f"dubbed_{int(idx):04d}.wav")
        synthesized = False

        # ── Try ElevenLabs ────────────────────────────────────────────────
        if use_elevenlabs:
            ok, result = _synthesize_elevenlabs(settings.elevenlabs_api_key, voice_id, text)
            if ok:
                try:
                    with open(out_path, "wb") as fh:
                        fh.write(result)
                    seg_out["dubbed_audio_path"] = out_path
                    seg_out["tts_error"] = ""
                    synthesized = True
                except Exception as e:
                    log.warning("stage4.write_failed", segment=idx, error=str(e))
            else:
                log.warning(
                    "stage4.elevenlabs_failed",
                    segment=idx,
                    error=str(result)[:200],
                )
                # Disable ElevenLabs for remaining segments to avoid
                # hammering a broken API and wasting time
                use_elevenlabs = False
                elevenlabs_error = str(result)

            # Rate limit politeness
            if synthesized:
                time.sleep(0.2)

        # ── Fallback: gTTS / pyttsx3 ──────────────────────────────────────
        if not synthesized:
            if not fallback_used:
                log.info(
                    "stage4.using_fallback",
                    reason=elevenlabs_error[:200] if elevenlabs_error else "ElevenLabs not configured",
                )
                fallback_used = True

            ok, err = _fallback_synthesize(text, lang, out_path)
            if ok:
                seg_out["dubbed_audio_path"] = out_path
                seg_out["tts_error"] = ""
                synthesized = True
            else:
                seg_out["tts_error"] = f"All TTS engines failed: {err}"
                log.error(
                    "stage4.all_tts_failed",
                    segment=idx,
                    error=err[:200],
                )

        if synthesized:
            succeeded += 1
        else:
            failed += 1

        tts_segments.append(seg_out)

    # ── Summary log ───────────────────────────────────────────────────────────
    if failed > 0:
        log.warning(
            "stage4.partial_failure",
            lang=lang,
            succeeded=succeeded,
            failed=failed,
            total=len(tts_segments),
        )
    else:
        log.info("stage4.complete", lang=lang, segments=len(tts_segments))

    output = {
        "target_language": lang,
        "voice_map": {},
        "segments": tts_segments,
    }

    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
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