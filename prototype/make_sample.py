#!/usr/bin/env python3
"""
Generate a small spoken `sample.wav` so the whole pipeline can be run
end-to-end without supplying your own audio.

The text is a short, legal-flavoured meeting snippet that exercises every
extractive feature: it contains decisions ("we decided", "the court ruled"),
action items ("needs to submit", "follow up", "deadline") and clear keywords.

Everything is fully offline / API-free, matching the rest of the project:
  1. `pyttsx3`  -- cross-platform offline TTS, if installed.
  2. Windows `System.Speech` -- built into Windows, no install needed.

Usage:
    python make_sample.py                 # writes sample.wav
    python make_sample.py --out demo.wav
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SAMPLE_TEXT = (
    "Good morning. This is the weekly case review for the Henderson matter. "
    "The client met with counsel on Monday to review the contract. "
    "After due diligence, we decided to proceed with the proposed settlement. "
    "The court ruled that the prior agreement remains valid and enforceable. "
    "The litigation team needs to submit the disclosure documents before the "
    "Friday deadline. Sarah will draft the response and follow up with the "
    "regulator next week. We agreed to reconvene once the filing is complete."
)


def _gen_pyttsx3(text: str, out_path: Path) -> bool:
    try:
        import pyttsx3
    except Exception:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as exc:  # pragma: no cover - depends on local TTS backend
        print(f"  pyttsx3 failed ({exc}); trying next method.")
        return False


def _gen_windows_speech(text: str, out_path: Path) -> bool:
    """Use the System.Speech synthesizer that ships with Windows (offline)."""
    if sys.platform != "win32":
        return False
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return False

    safe_text = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 1; "
        f"$s.SetOutputToWaveFile('{out_path}'); "
        f"$s.Speak('{safe_text}'); "
        "$s.Dispose();"
    )
    try:
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as exc:  # pragma: no cover - platform dependent
        print(f"  Windows System.Speech failed ({exc}).")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="sample.wav",
        help="Output WAV path (default: sample.wav).",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # System.Speech writes WAV directly; pyttsx3 may need a temp file first.
    tmp = Path(tempfile.gettempdir()) / "a2n_sample_tts.wav"

    print(f"Generating spoken sample audio -> {out_path}")
    ok = _gen_pyttsx3(SAMPLE_TEXT, tmp)
    if ok:
        shutil.move(str(tmp), str(out_path))
    else:
        ok = _gen_windows_speech(SAMPLE_TEXT, out_path)

    if not ok:
        sys.exit(
            "Could not generate sample audio.\n"
            "Install a TTS backend:  pip install pyttsx3\n"
            "(On Windows, System.Speech is built in and should have worked.)\n"
            "Alternatively, drop any .wav/.mp3 in the project and pass it with "
            "--audio."
        )

    size_kb = out_path.stat().st_size / 1024
    print(f"Done: {out_path}  ({size_kb:.0f} KB)")
    print("Now run:  python main.py --audio sample.wav")


if __name__ == "__main__":
    main()
