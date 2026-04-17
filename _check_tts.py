import json, os

job_dir = "storage/3aa324af-8bac-428b-9e36-86ebcb2449f3"
d = json.load(open(os.path.join(job_dir, "tts_hi.json"), "r", encoding="utf-8"))

for s in d["segments"]:
    audio = s.get("dubbed_audio_path")
    has = "YES" if (audio and os.path.exists(audio)) else "NO"
    err = s.get("tts_error", "")[:80]
    print(f"seg{s['segment_id']} has_audio={has} err={err}")

wav = os.path.join(job_dir, "dubbed_audio_hi.wav")
mp4 = os.path.join(job_dir, "dubbed_hi.mp4")
print(f"wav_exists={os.path.exists(wav)} wav_size={os.path.getsize(wav) if os.path.exists(wav) else 0}")
print(f"mp4_exists={os.path.exists(mp4)} mp4_size={os.path.getsize(mp4) if os.path.exists(mp4) else 0}")
