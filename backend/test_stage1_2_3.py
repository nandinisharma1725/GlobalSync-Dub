"""
test_stage1_2_3.py — Demo runner for Stage 1 + Stage 2 + Stage 3

Usage:
    python test_stage1_2_3.py --input meeting.mp4 --lang hi
    python test_stage1_2_3.py --input meeting.mp4 --lang hi --skip-diarization
    python test_stage1_2_3.py --input meeting.mp4 --lang hi --speakers "Marcus,Maya,David,Anna" --company "Quartz Power Group"
    python test_stage1_2_3.py --list-languages
"""

import argparse, json, os, sys, time, uuid
from pathlib import Path

# ── .env loader ───────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Colours ───────────────────────────────────────────────────────────────────
def _col_ok():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
            return True
        except Exception: return False
    return getattr(sys.stdout, "isatty", lambda: False)()

_C = _col_ok()
def _c(t, code): return f"\033[{code}m{t}\033[0m" if _C else t
green   = lambda t: _c(t,"32"); blue  = lambda t: _c(t,"34")
yellow  = lambda t: _c(t,"33"); red   = lambda t: _c(t,"31")
bold    = lambda t: _c(t,"1");  dim   = lambda t: _c(t,"2")
cyan    = lambda t: _c(t,"36"); magenta=lambda t: _c(t,"35")

def fmt_dur(s):
    s=int(s); m,s=divmod(s,60); h,m=divmod(m,60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"

def hdr(t):
    bar="━"*60
    print(f"\n{bold(bar)}\n  {bold(t)}\n{bold(bar)}\n")

def step(n,t,l): print(f"\n  {blue(f'[{n}/{t}]')} {l}")
def ok(m):   print(f"         {green('✓')} {m}")
def warn(m): print(f"         {yellow('⚠')} {m}")
def err(m):  print(f"         {red('✗')} {m}")

SPEAKER_COLORS = [blue, cyan, yellow, green, magenta]

def fake_diarize(wav_path, output_dir):
    import soundfile as sf
    info = sf.info(wav_path); dur = info.frames/info.samplerate
    segs = [{"speaker":"SPEAKER_00","start":0.0,"end":round(dur,3)}]
    (Path(output_dir)/"diarization.json").write_text(json.dumps(segs,indent=2))
    return segs

def print_transcript(segments, speaker_ids, lang_name):
    cmap = {sid: SPEAKER_COLORS[i%len(SPEAKER_COLORS)]
            for i,sid in enumerate(sorted(speaker_ids))}
    print(f"\n  {bold('── Bilingual Transcript (' + lang_name + ') ────────────────────')}\n")
    for seg in segments:
        spk = seg["speaker"]; cfn = cmap.get(spk, dim)
        ts  = f"{seg['start']:.1f}s"
        eng = seg.get("text","")
        tr  = seg.get("translated_text","")
        print(f"{cfn(f'  {spk:<14}')} {dim(f'[{ts}]')}")
        if eng: print(f"  {dim('EN:')} {eng[:100]}")
        if tr:  print(f"  {bold(lang_name[:2]+':')} {tr[:100]}")
        print()

def print_summary(s1,s2,s3,elapsed):
    segs=s3["segments"]; lang=s3["target_language"]
    lname=s3["target_language_name"]
    ne=[s for s in segs if s.get("translated_text","").strip()]
    out=Path(s1["wav_path"]).parent
    print(f"\n  {bold('── Summary ──────────────────────────────────────────')}\n")
    print(f"  {'Video duration:':<24} {fmt_dur(s1['duration_sec'])}")
    print(f"  {'Speakers found:':<24} {bold(str(s1['speaker_count']))}")
    print(f"  {'Segments transcribed:':<24} {bold(str(len(s2['segments'])))}")
    print(f"  {'Segments translated:':<24} {bold(str(len(segs)))}  →  {bold(lname)}")
    print(f"  {'Non-empty translations:':<24} {len(ne)}/{len(segs)}")
    print(f"\n  {'Output folder:':<24} {dim(str(out))}")
    print(f"  {'Cache files:':<24} stage1_result.json, transcription.json, translation_{lang}.json")
    print(f"\n  Completed in {fmt_dur(elapsed)}\n")
    print(bold("━"*60)); print()

def main():
    p = argparse.ArgumentParser(
        description="Stage 1+2+3 demo",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input","-i",help="Path to video file")
    p.add_argument("--lang","-l",default="hi",
                   help="Target language code (default: hi)\nRun --list-languages to see all")
    p.add_argument("--output","-o",default="./pipeline_output")
    p.add_argument("--model","-m",default="base",
                   choices=["tiny","base","small","medium","large"])
    p.add_argument("--skip-diarization",action="store_true")
    p.add_argument("--num-speakers",type=int,default=None)
    p.add_argument("--language",default="en",
                   help="Source spoken language (default: en)")
    p.add_argument("--speakers",default=None,
                   help="Comma-separated names e.g. 'Marcus,Maya,David,Anna'")
    p.add_argument("--company",default=None)
    p.add_argument("--topic",default=None)
    p.add_argument("--job-id",default=None)
    p.add_argument("--list-languages",action="store_true")
    args = p.parse_args()

    here = Path(__file__).parent
    sys.path.insert(0, str(here))

    try:
        import stage1_extract    as s1mod
        import stage2_transcribe as s2mod
        import stage3_translate  as s3mod
    except ImportError as e:
        print(f"Cannot import module: {e}"); sys.exit(1)

    if args.list_languages:
        print(f"\n  {bold('Supported target languages:')}\n")
        for code, name in s3mod.list_supported_languages():
            print(f"  {blue(f'{code:>4}')}  {name}")
        print(); sys.exit(0)

    if not args.input:
        p.print_help(); sys.exit(1)

    if args.lang not in s3mod.LANGUAGE_CODES:
        err(f"Unsupported language: '{args.lang}'")
        print("  Run --list-languages to see valid codes."); sys.exit(1)

    video   = str(Path(args.input).resolve())
    job_id  = args.job_id or str(uuid.uuid4())[:8]
    out_dir = str(Path(args.output).resolve() / job_id)
    lname   = s3mod.LANGUAGE_NAMES[args.lang]
    t_total = time.time()

    hdr("MNC Dubbing — Stage 1 + 2 + 3")
    print(f"  Input:    {dim(video)}")
    print(f"  Output:   {dim(out_dir)}")
    print(f"  Whisper:  {dim(args.model)}")
    print(f"  Target:   {bold(lname)} ({args.lang})")
    print(f"  Job ID:   {dim(job_id)}")
    if args.skip_diarization: print(f"  Mode:     {yellow('skip-diarization')}")
    if args.speakers:         print(f"  Speakers: {dim(args.speakers)}")
    if args.company:          print(f"  Company:  {dim(args.company)}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n  {bold('── STAGE 1: Audio Extraction & Diarization ─────────')}")
    t1 = time.time()

    step(1,3,"Validating video file…")
    errs = s1mod.validate_video_file(video)
    if errs:
        for e in errs: err(e)
        sys.exit(1)
    ok(f"{Path(video).name}  ({Path(video).stat().st_size/1e6:.1f} MB)  valid")

    step(2,3,"Extracting audio (FFmpeg → 16kHz mono WAV)…")
    try:
        wav  = s1mod.extract_audio(video, out_dir)
        dur  = s1mod.get_audio_duration(wav)
        ok(f"audio.wav  ({Path(wav).stat().st_size/1e6:.1f} MB  ·  {fmt_dur(dur)})")
    except Exception as e:
        err(str(e)); sys.exit(1)

    if args.skip_diarization:
        step(3,3,"Speaker diarization… (skipped)")
        raw = fake_diarize(wav, out_dir)
        warn("pyannote skipped — all audio attributed to SPEAKER_00")
    else:
        hf = os.environ.get("HF_TOKEN","").strip()
        if not hf:
            step(3,3,"Speaker diarization…")
            err("HF_TOKEN not set in .env")
            print(f"\n  {yellow('Tip:')} use --skip-diarization to test without a token.")
            sys.exit(1)
        step(3,3,"Detecting speakers with pyannote.audio…")
        print(f"         {dim('First run downloads ~1 GB model.')}")
        try:
            kwargs = {}
            if args.num_speakers: kwargs["num_speakers"] = args.num_speakers
            raw = s1mod.run_diarization(wav, out_dir, **kwargs)
        except Exception as e:
            err(str(e)); sys.exit(1)

    segs    = s1mod.merge_short_segments(raw)
    spk_ids = sorted({s["speaker"] for s in segs})
    ok(f"{len(segs)} segments  ·  {len(spk_ids)} speaker(s): {', '.join(spk_ids)}")

    s1 = {"wav_path":wav,"segments":segs,"speaker_count":len(spk_ids),
          "speaker_ids":spk_ids,"duration_sec":round(dur,1)}
    (Path(out_dir)/"stage1_result.json").write_text(json.dumps(s1,indent=2))
    print(f"\n  {green('Stage 1 done')} in {fmt_dur(time.time()-t1)}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n  {bold('── STAGE 2: Whisper Transcription ───────────────────')}")
    model_sz={"tiny":"39 MB","base":"74 MB","small":"244 MB",
               "medium":"769 MB","large":"1.5 GB"}
    print(f"  Model: {bold(args.model)} ({model_sz.get(args.model,'?')})  "
          f"Language: {bold(args.language)}")

    spk_names = [n.strip() for n in args.speakers.split(',')
                 if n.strip()] if args.speakers else None
    t2 = time.time()
    try:
        s2 = s2mod.run(
            stage1_result=s1, output_dir=out_dir,
            source_language=args.language, model_name=args.model,
            speaker_names=spk_names, company_name=args.company,
            meeting_topic=args.topic,
        )
    except Exception as e:
        err(f"Stage 2 failed: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    ok(f"{len(s2['segments'])} segments transcribed in {fmt_dur(time.time()-t2)}")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    print(f"\n  {bold('── STAGE 3: Translation → ' + lname + ' ─────────────────────')}")
    print(f"  {dim('Google Translate via deep-translator (free, no API key).')}")
    print(f"  {dim('~0.3s delay between calls to avoid rate limiting.')}")

    t3 = time.time()
    try:
        s3 = s3mod.run(
            stage2_result=s2, output_dir=out_dir,
            target_language=args.lang,
        )
    except Exception as e:
        err(f"Stage 3 failed: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    ok(f"{len(s3['segments'])} segments translated into {lname} "
       f"in {fmt_dur(time.time()-t3)}")

    if s3["segments"]:
        print_transcript(s3["segments"], spk_ids, lname)
    else:
        warn("No segments. Check the video has clear speech.")

    print_summary(s1, s2, s3, time.time()-t_total)


if __name__ == "__main__":
    main()