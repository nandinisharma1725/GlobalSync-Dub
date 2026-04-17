"""
backend/pipeline/translate.py

Stage 3 — Translation using Deep-Translator (free, no API key needed)

Supports multiple free translation backends without authentication.
"""

import json
from pathlib import Path
from typing import Optional
import structlog

from deep_translator import GoogleTranslator
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

def translate_segment(
    text: str,
    target_language: str,
) -> str:
    """
    Translates text using Google Translate (free, no API key needed).

    Args:
        text: Source text in English
        target_language: ISO 639-1 code

    Returns:
        Translated text string
    """
    try:
        # Map our language codes to deep-translator codes if needed
        lang_map = {
            "hi": "hi",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "ja": "ja",
            "zh": "zh",
            "ar": "ar",
            "pt": "pt",
            "it": "it",
            "ko": "ko",
            "nl": "nl",
            "tr": "tr",
            "sv": "sv",
            "pl": "pl",
            "ru": "ru",
            "id": "id",
            "uk": "uk",
            "el": "el",
            "cs": "cs",
            "fi": "fi",
            "ro": "ro",
            "da": "da",
            "bg": "bg",
            "ms": "ms",
            "sk": "sk",
            "hr": "hr",
            "ta": "ta",
            "fil": "fil",
        }
        
        target_lang = lang_map.get(target_language, target_language)
        translator = GoogleTranslator(source_language='en', target_language=target_lang)
        result = translator.translate(text)
        return result
    except Exception as e:
        log.error("translate.failed", text_length=len(text), target=target_language, error=str(e))
        # Fallback: return original text
        return text


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

    segments = stage2_result["segments"]
    translated_segments = []

    for i, seg in enumerate(segments):
        text = seg["text"]

        log.info(
            "stage3.translate",
            segment=i + 1,
            total=len(segments),
            speaker=seg["speaker"],
            lang=target_language,
            text_preview=text[:60],
        )

        try:
            translated = translate_segment(
                text=text,
                target_language=target_language,
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
        "segments": translated_segments,
    }
    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    log.info("stage3.complete", segments=len(translated_segments), lang=target_language)
    return output