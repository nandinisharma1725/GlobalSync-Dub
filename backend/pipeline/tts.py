"""
backend/pipeline/tts.py

Stage 4 — Text-to-Speech with Voice Cloning (ElevenLabs)

What this does:
  1. For each unique speaker, builds a voice clone from their audio samples
  2. Generates dubbed audio for each segment in the target language
  3. The cloned voice preserves vocal identity (tone, accent cadence) across languages

Voice cloning strategy:
  - Collect ALL audio chunks for a speaker from Stage 1
  - Concatenate them into a 30–90 second reference sample
  - Create an "instant voice clone" via ElevenLabs API
  - Reuse the same clone for all that speaker's segments

ElevenLabs models used:
  - eleven_multilingual_v2: Best quality, 29 languages
  - eleven_flash_v2_5: Faster, lower latency (use for long videos)
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
import structlog

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()

# ElevenLabs model choice:
# - eleven_multilingual_v2  → highest quality (recommended)
# - eleven_flash_v2_5       → faster, slightly lower quality
TTS_MODEL = "eleven_multilingual_v2"


def build_speaker_reference(
    speaker_id: str,
    segments: list[dict],
    output_dir: str,
    max_duration: float = 90.0,
) -> Optional[str]:
    """
    Concatenates audio chunks from a speaker into a single reference file.
    ElevenLabs instant cloning works best with 30–90 seconds of clean audio.

    Returns path to concatenated WAV, or None if not enough audio.
    """
    speaker_dir = Path(output_dir) / "speaker_refs"
    speaker_dir.mkdir(exist_ok=True)
    ref_path = str(speaker_dir / f"{speaker_id}_reference.wav")

    if Path(ref_path).exists():
        return ref_path

    # Gather chunks for this speaker
    speaker_chunks = [
        seg["chunk_path"]
        for seg in segments
        if seg["speaker"] == speaker_id and Path(seg.get("chunk_path", "")).exists()
    ]

    if not speaker_chunks:
        log.warning("stage4.no_chunks", speaker=speaker_id)
        return None

    # Build an FFmpeg concat input file
    concat_list = speaker_dir / f"{speaker_id}_concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{os.path.abspath(p)}'" for p in speaker_chunks)
    )

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-t", str(max_duration),    # cap at max_duration
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            ref_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error("stage4.concat_failed", speaker=speaker_id, stderr=result.stderr)
        return None

    log.info("stage4.reference_built", speaker=speaker_id, path=ref_path)
    return ref_path


def clone_speaker_voice(
    client: ElevenLabs,
    speaker_id: str,
    reference_audio_path: str,
    job_id: str,
) -> Optional[str]:
    """
    Creates an instant voice clone via ElevenLabs API.
    Returns the ElevenLabs voice_id for this speaker.
    """
    voice_name = f"job_{job_id}_{speaker_id}"

    log.info("stage4.clone_voice", speaker=speaker_id, voice_name=voice_name)

    with open(reference_audio_path, "rb") as f:
        voice = client.clone(
            name=voice_name,
            description=f"Auto-cloned voice for speaker {speaker_id}, job {job_id}",
            files=[f],
        )

    log.info("stage4.clone_done", speaker=speaker_id, voice_id=voice.voice_id)
    return voice.voice_id


def synthesize_segment(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    output_path: str,
) -> str:
    """
    Generates audio for one translated segment using ElevenLabs TTS.
    Saves to output_path as MP3, then converts to WAV.
    """
    mp3_path = output_path.replace(".wav", ".mp3")

    # Generate audio
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=TTS_MODEL,
        voice_settings=VoiceSettings(
            stability=0.5,         # 0-1: higher = more consistent, less expressive
            similarity_boost=0.8,  # 0-1: higher = closer to original voice
            style=0.2,             # 0-1: style exaggeration (keep low for corporate)
            use_speaker_boost=True,
        ),
    )

    # Write MP3
    with open(mp3_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    # Convert to WAV for librosa processing in Stage 5
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path,
         "-acodec", "pcm_s16le", "-ar", "24000", output_path],
        capture_output=True,
        check=True,
    )
    os.remove(mp3_path)
    return output_path


def run(
    stage3_result: dict,
    stage1_result: dict,
    output_dir: str,
    job_id: str,
) -> dict:
    """
    Main entry point for Stage 4.

    Clones voices for all speakers, then synthesizes dubbed audio for every segment.

    Returns:
        {
          "target_language": "hi",
          "voice_map": {"SPEAKER_00": "voice_id_xxx", ...},
          "segments": [
            {
              ...all previous fields...,
              "dubbed_audio_path": "...",
            },
            ...
          ]
        }
    """
    lang = stage3_result["target_language"]
    cache_path = Path(output_dir) / f"tts_{lang}.json"

    if cache_path.exists():
        log.info("stage4.cache_hit", lang=lang)
        return json.loads(cache_path.read_text())

    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    segments = stage3_result["segments"]

    dubbed_dir = Path(output_dir) / f"dubbed_{lang}"
    dubbed_dir.mkdir(exist_ok=True)

    # ── Step 1: Build voice clones for each unique speaker ──────────────
    speaker_ids = list({seg["speaker"] for seg in segments})
    voice_map: dict[str, str] = {}
    voice_cache_path = Path(output_dir) / "voice_map.json"

    # Load existing voice map (avoid re-cloning for multiple target languages)
    if voice_cache_path.exists():
        voice_map = json.loads(voice_cache_path.read_text())

    for speaker_id in speaker_ids:
        if speaker_id in voice_map:
            log.info("stage4.voice_reuse", speaker=speaker_id)
            continue

        ref_path = build_speaker_reference(speaker_id, segments, output_dir)
        if ref_path is None:
            log.warning("stage4.no_reference", speaker=speaker_id)
            continue

        voice_id = clone_speaker_voice(client, speaker_id, ref_path, job_id)
        if voice_id:
            voice_map[speaker_id] = voice_id

    voice_cache_path.write_text(json.dumps(voice_map, indent=2))

    # ── Step 2: Synthesize each segment ─────────────────────────────────
    dubbed_segments = []

    for i, seg in enumerate(segments):
        speaker = seg["speaker"]
        voice_id = voice_map.get(speaker)
        text = seg["translated_text"]

        if not voice_id:
            log.warning("stage4.no_voice_for_speaker", speaker=speaker, segment=i)
            dubbed_segments.append({**seg, "dubbed_audio_path": None})
            continue

        output_path = str(dubbed_dir / f"segment_{i:04d}.wav")

        log.info(
            "stage4.synthesize",
            segment=i + 1,
            total=len(segments),
            speaker=speaker,
            text_preview=text[:50],
        )

        try:
            synthesize_segment(client, text, voice_id, output_path)
            dubbed_segments.append({**seg, "dubbed_audio_path": output_path})
        except Exception as e:
            log.error("stage4.synthesis_failed", segment=i, error=str(e))
            dubbed_segments.append({**seg, "dubbed_audio_path": None, "tts_error": str(e)})

    output = {
        "target_language": lang,
        "voice_map": voice_map,
        "segments": dubbed_segments,
    }
    cache_path.write_text(json.dumps(output, indent=2))

    log.info("stage4.complete", segments=len(dubbed_segments), voices=len(voice_map))
    return output


def cleanup_cloned_voices(job_id: str, voice_map: dict) -> None:
    """
    Deletes cloned voices from ElevenLabs after the job is done.
    Important: ElevenLabs charges for stored voices — clean up after use.

    Call this from your job cleanup task.
    """
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    for speaker_id, voice_id in voice_map.items():
        try:
            client.voices.delete(voice_id)
            log.info("stage4.cleanup.voice_deleted", speaker=speaker_id, voice_id=voice_id)
        except Exception as e:
            log.warning("stage4.cleanup.failed", voice_id=voice_id, error=str(e))