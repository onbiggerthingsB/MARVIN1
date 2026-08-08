"""Generate the canned acknowledgment clips once. ElevenLabs if keyed, else `say`."""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PHRASES = {
    "got_it": "Got it.",
    "on_it": "On it.",
    "one_moment": "One moment.",
    "pulling_up": "Pulling it up now.",
    "done_sir": "Done, sir.",
    "standing_by": "Standing by.",
    "cannot_hear": "I can't hear you — check the microphone.",
    "rate_limited": "We're rate limited. Queuing it.",
    "backup_voice": "Running on the backup voice.",
    "which_one": "Which one do you mean?",
    "welcome_back": "Welcome back, sir.",
    "canceling": "Canceling.",
}
OUT = Path(__file__).resolve().parent.parent / "static" / "clips"


def eleven(text: str, path: Path, key: str, voice: str) -> None:
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_64",
        data=json.dumps({"text": text, "model_id": "eleven_flash_v2_5"}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"})
    path.write_bytes(urllib.request.urlopen(req).read())


def say(text: str, path: Path) -> None:
    subprocess.run(["say", "-v", "Daniel", "-o", str(path),
                    "--data-format=LEI16@22050", text], check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    key, voice = os.environ.get("ELEVENLABS_API_KEY"), os.environ.get("ELEVENLABS_VOICE_ID")
    use_eleven = bool(key and voice) and "--say" not in sys.argv
    manifest = {}
    for slug, phrase in PHRASES.items():
        ext = "mp3" if use_eleven else "wav"
        f = OUT / f"{slug}.{ext}"
        (eleven(phrase, f, key, voice) if use_eleven else say(phrase, f))
        manifest[slug] = f"/static/clips/{f.name}"
        print("wrote", f.name)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
