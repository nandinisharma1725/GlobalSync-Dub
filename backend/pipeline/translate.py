"""
backend/pipeline/translate.py

Stage 3 — Length-Aware Translation using GPT-4o

The core challenge: translated text is often longer or shorter than the original.
If we just translate naively, the dubbed audio will be out of sync.

Solution: prompt GPT-4o to produce a translation that matches the original
spoken duration. We pass:
  - The original text
  - The original spoken duration (in seconds)
  - A corporate-formal tone instruction
  - The target language

GPT-4o returns a translation tuned to fit within that time window.
"""

import json
from pathlib import Path
from typing import Optional
import structlog

from openai import OpenAI
from ..utils.config import get_settings

log = structlog.get_logger()
settings = get_settings()

# Map of language codes → full language name for the prompt
LANGUAGE_NAMES = {
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Mandarin Chinese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "nl": "Dutch",
    "tr": "Turkish",
    "sv": "Swedish",
    "pl": "Polish",
    "ru": "Russian",
    "id": "Indonesian",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "da": "Danish",
    "bg": "Bulgarian",
    "ms": "Malay",
    "sk": "Slovak",
    "hr": "Croatian",
    "ta": "Tamil",
    "fil": "Filipino",
}

SYSTEM_PROMPT = """You are a professional corporate translator for MNC board meeting recordings.

Rules you MUST follow:
1. Translate the provided English text to {target_language}.
2. The translation must be speakable in approximately {duration:.1f} seconds.
   - Average speaking pace is 2.5–3.5 words/second depending on language.
   - If the direct translation is too long, use concise equivalents. Never omit meaning.
   - If the direct translation is too short, you may use slightly more formal phrasing.
3. Preserve formal corporate register (no slang, no contractions).
4. Preserve names of people, companies, products, and financial figures exactly as-is.
5. Respond with ONLY the translated text — no preamble, no explanation, no quotes."""


def translate_segment(
    client: OpenAI,
    text: str,
    duration: float,
    target_language: str,
    target_language_name: str,
) -> str:
    """
    Translates a single segment with GPT-4o using a duration-aware prompt.

    Args:
        text: English source text
        duration: Original spoken duration in seconds
        target_language: ISO 639-1 code
        target_language_name: Full language name for the prompt

    Returns:
        Translated text string
    """
    system = SYSTEM_PROMPT.format(
        target_language=target_language_name,
        duration=duration,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.3,    # low temp = more consistent, less creative
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


def run(
    stage2_result: dict,
    output_dir: str,
    target_language: str,
) -> dict:
    """
    Main entry point for Stage 3.

    Translates every transcribed segment into the target language.

    Args:
        stage2_result: Output from stage2_transcribe.run()
        output_dir: Working directory for this job
        target_language: ISO 639-1 code e.g. "hi", "es", "fr"

    Returns:
        {
          "target_language": "hi",
          "segments": [
            {
              ...all stage2 fields...,
              "translated_text": "...",
            },
            ...
          ]
        }
    """
    cache_path = Path(output_dir) / f"translation_{target_language}.json"
    if cache_path.exists():
        log.info("stage3.cache_hit", lang=target_language)
        return json.loads(cache_path.read_text())

    if target_language not in LANGUAGE_NAMES:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Supported codes: {list(LANGUAGE_NAMES.keys())}"
        )

    target_language_name = LANGUAGE_NAMES[target_language]
    client = OpenAI(api_key=settings.openai_api_key)
    segments = stage2_result["segments"]

    translated_segments = []

    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        text = seg["text"]

        log.info(
            "stage3.translate",
            segment=i + 1,
            total=len(segments),
            speaker=seg["speaker"],
            lang=target_language,
            duration=duration,
            text_preview=text[:60],
        )

        try:
            translated = translate_segment(
                client=client,
                text=text,
                duration=duration,
                target_language=target_language,
                target_language_name=target_language_name,
            )

            translated_segments.append({
                **seg,
                "translated_text": translated,
                "target_language": target_language,
            })

        except Exception as e:
            log.error("stage3.segment_failed", segment=i, error=str(e))
            # Fall back to original English text if translation fails
            translated_segments.append({
                **seg,
                "translated_text": text,
                "target_language": "en",
                "translation_error": str(e),
            })

    output = {
        "target_language": target_language,
        "target_language_name": target_language_name,
        "segments": translated_segments,
    }
    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    log.info("stage3.complete", segments=len(translated_segments), lang=target_language)
    return output