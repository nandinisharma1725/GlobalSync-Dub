"""
stage3_translate.py  —  Stage 3: Translation using deep-translator (FREE)

Standalone file — drop alongside stage1_extract.py and stage2_transcribe.py.

Install:  pip install deep-translator

Supported language codes for --lang:
  hi  Hindi        es  Spanish     fr  French      de  German
  ja  Japanese     zh  Mandarin    ar  Arabic       pt  Portuguese
  it  Italian      ko  Korean      nl  Dutch        tr  Turkish
  sv  Swedish      pl  Polish      ru  Russian      id  Indonesian
  uk  Ukrainian    el  Greek       cs  Czech        fi  Finnish
  ro  Romanian     da  Danish      bg  Bulgarian    ms  Malay
  sk  Slovak       hr  Croatian    ta  Tamil        fil Filipino
"""

import json
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

LANGUAGE_CODES: dict = {
    "hi":  "hi",     "es":  "es",     "fr":  "fr",     "de":  "de",
    "ja":  "ja",     "zh":  "zh-CN",  "ar":  "ar",     "pt":  "pt",
    "it":  "it",     "ko":  "ko",     "nl":  "nl",     "tr":  "tr",
    "sv":  "sv",     "pl":  "pl",     "ru":  "ru",     "id":  "id",
    "uk":  "uk",     "el":  "el",     "cs":  "cs",     "fi":  "fi",
    "ro":  "ro",     "da":  "da",     "bg":  "bg",     "ms":  "ms",
    "sk":  "sk",     "hr":  "hr",     "ta":  "ta",     "fil": "tl",
}

LANGUAGE_NAMES: dict = {
    "hi": "Hindi",         "es": "Spanish",      "fr": "French",
    "de": "German",        "ja": "Japanese",     "zh": "Mandarin Chinese",
    "ar": "Arabic",        "pt": "Portuguese",   "it": "Italian",
    "ko": "Korean",        "nl": "Dutch",        "tr": "Turkish",
    "sv": "Swedish",       "pl": "Polish",       "ru": "Russian",
    "id": "Indonesian",    "uk": "Ukrainian",    "el": "Greek",
    "cs": "Czech",         "fi": "Finnish",      "ro": "Romanian",
    "da": "Danish",        "bg": "Bulgarian",    "ms": "Malay",
    "sk": "Slovak",        "hr": "Croatian",     "ta": "Tamil",
    "fil": "Filipino",
}

_REQUEST_DELAY = 0.3


def translate_text(text: str, target_language: str) -> str:
    """Translates one English string. Falls back to original on error."""
    if not text or not text.strip():
        return text

    google_code = LANGUAGE_CODES.get(target_language)
    if not google_code:
        raise ValueError(f"Unsupported language: '{target_language}'")

    if len(text) > 4900:
        log.warning("stage3.text_too_long", chars=len(text))
        text = text[:4900]

    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        raise ImportError("Run:  pip install deep-translator")

    result = GoogleTranslator(source="en", target=google_code).translate(text)
    return result.strip() if result else text


def run(stage2_result: dict, output_dir: str, target_language: str) -> dict:
    """
    Translates every segment from Stage 2 into target_language.

    Args:
        stage2_result:   Output from stage2_transcribe.run()
        output_dir:      Working directory (cache: translation_{lang}.json)
        target_language: e.g. "hi" for Hindi

    Returns:
        {
          "target_language":      "hi",
          "target_language_name": "Hindi",
          "segments": [
            {
              ...all stage2 fields...,
              "translated_text": str,
              "original_text":   str,
              "target_language": str,
            }, ...
          ]
        }
    """
    if target_language not in LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Supported: {sorted(LANGUAGE_CODES.keys())}"
        )

    cache_path = Path(output_dir) / f"translation_{target_language}.json"
    if cache_path.exists():
        log.info("stage3.cache_hit", lang=target_language)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    lang_name = LANGUAGE_NAMES[target_language]
    segments  = stage2_result.get("segments", [])

    log.info("stage3.start", segments=len(segments),
             target=target_language, target_name=lang_name)

    translated_segments = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()

        log.info("stage3.translate", segment=i+1, total=len(segments),
                 speaker=seg.get("speaker"), preview=text[:60])

        if not text:
            translated_segments.append({
                **seg, "translated_text": "",
                "original_text": text, "target_language": target_language,
            })
            continue

        try:
            translated = translate_text(text, target_language)
            log.info("stage3.translated", segment=i+1,
                     original=text[:50], translated=translated[:50])
        except Exception as e:
            log.error("stage3.segment_failed", segment=i, error=str(e))
            translated = text  # fall back to English

        translated_segments.append({
            **seg, "translated_text": translated,
            "original_text": text, "target_language": target_language,
        })

        time.sleep(_REQUEST_DELAY)

    output = {
        "target_language":      target_language,
        "target_language_name": lang_name,
        "segments":             translated_segments,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log.info("stage3.complete", lang=target_language,
             segments=len(translated_segments))
    return output


def list_supported_languages() -> list:
    return sorted([(c, LANGUAGE_NAMES[c]) for c in LANGUAGE_CODES],
                  key=lambda x: x[1])