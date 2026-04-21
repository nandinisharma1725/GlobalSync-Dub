"""
stage2_transcribe.py  —  Stage 2 (v3): Speech-to-Text using local Whisper

What changed from v2:
  1. Post-transcription merge gap reduced 0.8s → 0.35s
     WHY: 0.8s was merging fast back-and-forth exchanges (e.g. a 0.6s gap
     between Marcus and David's turns got incorrectly joined into one segment).
     0.35s still merges natural mid-sentence pauses (0.2-0.3s) but correctly
     keeps genuine speaker turns separate.

  2. initial_prompt parameter added to transcribe_array()
     WHY: Whisper has no context about who is speaking or the company name.
     Passing a short prompt ("Participants: Marcus, Maya, David, Anna.") fixes
     errors like "Meyer"→Maya, "office"→offer, "Courts"→Quartz. This is a
     general technique for any meeting — caller provides names/context.

  3. condition_on_previous_text=False + no_speech_threshold=0.6
     WHY: condition_on_previous_text=True (Whisper default) causes the model
     to "drift" and repeat earlier text in long segments. Setting it False makes
     each segment independent. Higher no_speech_threshold avoids transcribing
     background noise as garbled words.

  4. Auto model upgrade for very short segments (< 2s → use 'small')
     WHY: base model error rate spikes for clips under 2 seconds. Short
     interjections like "Sure." or "Good." get mangled with base model.

Install:
  pip install openai-whisper soundfile numpy librosa structlog
"""

import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

WHISPER_SAMPLE_RATE = 16_000
MIN_CHUNK_SAMPLES   = 1_600     # 0.1s — pad shorter chunks with silence
SHORT_SEGMENT_SECS  = 2.0       # segments shorter than this get 'small' model

_model      = None
_model_name = None


def _get_model(model_name: str = "base"):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            import whisper
        except ImportError:
            raise ImportError("Run:  pip install openai-whisper")
        log.info("stage2.model.loading", model=model_name)
        _model      = whisper.load_model(model_name, device="cpu")
        _model_name = model_name
        log.info("stage2.model.ready", model=model_name)
    return _model


# ── Audio utilities ───────────────────────────────────────────────────────────

def slice_audio(wav_path: str, start: float, end: float) -> np.ndarray:
    """
    Slices WAV to 16kHz mono float32 numpy array.
    Passes the array to Whisper directly — no ffmpeg subprocess (Windows fix).
    """
    wav_path = str(Path(wav_path).resolve())
    info     = sf.info(wav_path)
    sr       = info.samplerate

    y, _ = sf.read(
        wav_path,
        start=int(start * sr),
        stop=int(end   * sr),
        dtype="float32",
    )

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != WHISPER_SAMPLE_RATE:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)
        except ImportError:
            raise ImportError("Run:  pip install librosa")

    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak

    if len(y) < MIN_CHUNK_SAMPLES:
        y = np.pad(y, (0, MIN_CHUNK_SAMPLES - len(y)))

    return y.astype(np.float32)


# ── Initial prompt builder ────────────────────────────────────────────────────

def build_initial_prompt(
    speaker_names: list = None,
    company_name:  str  = None,
    domain:        str  = None,
) -> str:
    """
    Builds a short Whisper initial_prompt for a corporate meeting.

    Why this helps accuracy:
      Whisper generates text by predicting likely next tokens. If it has
      never seen a name like "Maya" it defaults to the phonetically similar
      common word it knows ("Meyer"). Providing the name in the prompt tells
      Whisper to prefer it when the audio sounds like that name.

      This is a general technique — NOT tuned to any specific video.
      The caller passes the meeting-specific names; this function formats them.

    Args:
        speaker_names: List of participant first names ["Marcus", "Maya", ...]
        company_name:  Organisation name e.g. "Quartz Power Group"
        domain:        Topic hint e.g. "sales figures, Q4 review, market share"

    Returns:
        A string of <= 244 characters (Whisper's prompt token budget)
    """
    parts = ["Corporate meeting."]
    if company_name:
        parts.append(f"Company: {company_name}.")
    if speaker_names:
        parts.append(f"Speakers: {', '.join(speaker_names)}.")
    if domain:
        parts.append(f"Topic: {domain}.")
    prompt = " ".join(parts)
    # Whisper silently ignores prompts over ~244 chars — trim to be safe
    return prompt[:244]


# ── Core transcription ────────────────────────────────────────────────────────

def transcribe_array(
    audio_array:    np.ndarray,
    language:       str  = "en",
    model_name:     str  = "base",
    initial_prompt: str  = "",
) -> dict:
    """
    Transcribes a numpy array with Whisper.

    Key options compared to v2:
      condition_on_previous_text=False  — prevents text drift in long segments
      no_speech_threshold=0.6           — avoids transcribing background noise
      initial_prompt                    — primes vocabulary with names/company

    Returns {"text": str, "words": [{word, start, end}, ...]}
    """
    model = _get_model(model_name)

    transcribe_kwargs = dict(
        language                   = language,
        verbose                    = False,
        fp16                       = False,
        condition_on_previous_text = False,   # prevents drift/repetition
        no_speech_threshold        = 0.6,     # higher = less background noise transcribed
        compression_ratio_threshold= 2.4,     # default; filters repetitive outputs
    )
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    result = model.transcribe(audio_array, **transcribe_kwargs)

    text  = result["text"].strip()
    words = []
    for seg in result.get("segments", []):
        t = seg.get("text", "").strip()
        if t:
            words.append({
                "word":  t,
                "start": round(seg["start"], 3),
                "end":   round(seg["end"],   3),
            })

    return {"text": text, "words": words}


# ── Post-transcription turn merger ────────────────────────────────────────────

def merge_consecutive_turns(segments: list, max_gap: float = 0.35) -> list:
    """
    Merges consecutive same-speaker segments where the gap is <= max_gap.

    max_gap is now 0.35s (was 0.8s in v2).

    Why 0.35s:
      - Natural mid-sentence pauses in meetings: 0.1–0.3s  → merge (correct)
      - Fast back-and-forth exchange gaps: 0.4–0.8s         → separate (correct)
      - Previous 0.8s was merging Marcus (SPEAKER_02) with David (SPEAKER_02
        misattributed) who had a 0.57s gap — creating multi-speaker segments.

    Text is joined with a space. segment_ids are re-indexed from 0.
    """
    if not segments:
        return []

    merged  = []
    current = dict(segments[0])
    current["words"] = list(current.get("words", []))

    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        if seg["speaker"] == current["speaker"] and gap <= max_gap:
            current["text"]  = current["text"].rstrip() + " " + seg["text"].lstrip()
            current["end"]   = seg["end"]
            current["words"].extend(seg.get("words", []))
        else:
            merged.append(current)
            current = dict(seg)
            current["words"] = list(current.get("words", []))

    merged.append(current)

    for i, s in enumerate(merged):
        s["segment_id"] = i

    return merged


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    stage1_result:   dict,
    output_dir:      str,
    source_language: str  = "en",
    model_name:      str  = "base",
    speaker_names:   list = None,
    company_name:    str  = None,
    meeting_topic:   str  = None,
) -> dict:
    """
    Main entry point for Stage 2.

    New parameters vs v2:
      speaker_names: ["Marcus", "Maya", "David", "Anna"]
        Pass participant names to improve Whisper accuracy on proper nouns.
        These are fed into an initial_prompt that primes Whisper's vocabulary.
        Leave None if you don't know the names — it still works, just with
        potentially more name/word substitution errors.

      company_name: "Quartz Power Group"
        Company name for the initial_prompt. Fixes brand-name transcription.

      meeting_topic: "Q4 sales review, market share, cost-cutting"
        Domain hint for the initial_prompt.

    Returns:
        {
          "segments": [
            {
              "segment_id": int,
              "speaker":    str,
              "start":      float,
              "end":        float,
              "text":       str,
              "words":      [{word, start, end}, ...],
            },
            ...
          ]
        }
    """
    cache_path = Path(output_dir) / "transcription.json"
    if cache_path.exists():
        log.info("stage2.cache_hit")
        return json.loads(cache_path.read_text())

    wav_path     = stage1_result["wav_path"]
    raw_segments = stage1_result["segments"]

    # Build initial_prompt from caller-supplied context
    initial_prompt = build_initial_prompt(
        speaker_names = speaker_names,
        company_name  = company_name,
        domain        = meeting_topic,
    )
    if initial_prompt:
        log.info("stage2.initial_prompt", prompt=initial_prompt)

    log.info("stage2.start",
             segments=len(raw_segments),
             model=model_name,
             language=source_language)

    _get_model(model_name)  # load once before the loop

    enriched = []

    for i, seg in enumerate(raw_segments):
        seg_duration = seg["end"] - seg["start"]

        # Auto-upgrade to 'small' for very short segments where base struggles
        effective_model = model_name
        if model_name == "base" and seg_duration < SHORT_SEGMENT_SECS:
            effective_model = "small"
            _get_model("small")   # load small model if needed
            log.info("stage2.model_upgrade",
                     segment=i+1, duration=round(seg_duration,2),
                     reason="short segment < 2s")

        log.info("stage2.transcribe",
                 segment=i + 1, total=len(raw_segments),
                 speaker=seg["speaker"],
                 start=seg["start"], end=seg["end"],
                 duration=round(seg_duration, 2))

        try:
            audio_array = slice_audio(wav_path, seg["start"], seg["end"])

            result = transcribe_array(
                audio_array    = audio_array,
                language       = source_language,
                model_name     = effective_model,
                initial_prompt = initial_prompt,
            )

            if not result["text"]:
                log.warning("stage2.empty", segment=i)
                continue

            # Offset word timestamps to absolute video time
            offset = seg["start"]
            for w in result["words"]:
                w["start"] = round(w["start"] + offset, 3)
                w["end"]   = round(w["end"]   + offset, 3)

            enriched.append({
                "segment_id": i,
                "speaker":    seg["speaker"],
                "start":      seg["start"],
                "end":        seg["end"],
                "text":       result["text"],
                "words":      result["words"],
                "chunk_path": "",
            })

            log.info("stage2.segment_done",
                     segment=i + 1,
                     speaker=seg["speaker"],
                     text_preview=result["text"][:70])

        except Exception as e:
            log.error("stage2.segment_failed", segment=i, error=str(e))
            continue

    # Merge consecutive same-speaker turns (gap <= 0.35s only)
    before   = len(enriched)
    enriched = merge_consecutive_turns(enriched, max_gap=0.35)

    log.info("stage2.turn_merge",
             before=before, after=len(enriched),
             collapsed=before - len(enriched))

    output = {"segments": enriched}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    log.info("stage2.complete",
             segments=len(enriched),
             speakers=list({s["speaker"] for s in enriched}))

    gc.collect()
    return output