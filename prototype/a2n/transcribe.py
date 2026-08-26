"""
Local speech-to-text using OpenAI Whisper.

Whisper runs entirely on the local machine (CPU or GPU). No audio is sent to
any external service -- this is the on-device transcription stage from the
paper. The first run downloads the chosen Whisper checkpoint from OpenAI's
public store and caches it under ~/.cache/whisper; thereafter it is fully
offline.

Requires ffmpeg to be installed and on PATH (Whisper uses it to decode audio).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe an audio file with a local Whisper model.

    Parameters
    ----------
    audio_path : path to .wav/.mp3/.m4a/etc.
    model_name : tiny | base | small | medium | large | turbo
                 (larger = more accurate, more VRAM/time)
    language   : ISO code such as "en"; None lets Whisper auto-detect.

    Returns
    -------
    The raw Whisper result dict, including "text" and "segments"
    (each segment has "start", "end", "text").
    """
    # Imported lazily so that unit tests of the extractive logic do not require
    # whisper/torch to be installed.
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language, verbose=False)
    return result
