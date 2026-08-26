"""
Stage 2 - Speaker diarization with pyannote.

Uses speaker-diarization-community-1, which reports lower DER than 3.1 across
AMI / AliMeeting / AISHELL-4 and, more importantly for this pipeline, exposes
an *exclusive* (non-overlapping) diarization. Overlapping speaker regions are
the main source of noise when reconciling diarization spans against word
timestamps, and the exclusive variant removes that class of error entirely.

The model is gated on Hugging Face. Accept the conditions at
https://huggingface.co/pyannote/speaker-diarization-community-1 and set
HF_TOKEN before running - discovering this mid-build is a classic time sink.

Run standalone:
    python src/stages/diarize.py data/audio/hearing.wav --out work/hearing/speakers.json
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.config import load_config, DiarizeConfig            # noqa: E402
from src.schemas import DiarizationResult, SpeakerTurn       # noqa: E402
from src.util import gpu                                      # noqa: E402
from src.util.io import write_json                            # noqa: E402
from src.util.logging import get_logger                       # noqa: E402

log = get_logger("diarize")


def _hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    return None


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

# pyannote reads audio through torchaudio and then asserts that a requested
# window yielded exactly the number of samples the container's duration
# implies. For compressed formats that assertion is unsafe, and MP3 is the
# worst case: the codec's encoder delay and end padding mean the header's
# frame count and the decodable sample count disagree by a few thousand
# samples. The pipeline then dies on the very first ten-second window with
#
#     resulted in 219395 samples instead of the expected 220500 samples
#
# which reads like a corrupt file and is not one. Every Oyez argument is an
# MP3, so on this project's own evaluation corpus this fires every time.
#
# Rather than loosening the check - the check is right, and a diarization
# silently misaligned by 25 ms is worse than one that stops - the input is
# decoded to 16 kHz mono PCM first. That is the sample rate pyannote resamples
# to anyway, so nothing is lost, the sample count becomes exact by
# construction, and the decode is a few seconds on an hour of audio.

WAV_SUFFIXES = {".wav", ".wave"}


def _decode_to_wav(audio_path: str) -> tuple[str, object]:
    """
    Return (path_to_use, tempdir_or_None).

    The caller must keep the returned TemporaryDirectory alive until pyannote
    has finished with the file, and is responsible for cleaning it up.
    """
    import shutil
    import subprocess
    import tempfile

    src = pathlib.Path(audio_path)
    if src.suffix.lower() in WAV_SUFFIXES:
        return audio_path, None

    if shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg not found, so %s is being handed to pyannote as-is. "
            "Compressed formats can fail with a sample-count mismatch; if "
            "that happens, install ffmpeg (winget install Gyan.FFmpeg) or "
            "convert the file to WAV.", src.name,
        )
        return audio_path, None

    tmp = tempfile.TemporaryDirectory(prefix="lausanne_diar_")
    out = pathlib.Path(tmp.name) / (src.stem + ".wav")
    log.info("decoding %s to 16 kHz mono PCM for diarization", src.name)
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-i", str(src), "-ac", "1", "-ar", "16000",
             "-acodec", "pcm_s16le", str(out)],
            check=True, capture_output=True, timeout=1800,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        log.warning(
            "ffmpeg decode failed (%s); using the original file. %s",
            exc, detail.decode("utf-8", "replace")[:300],
        )
        tmp.cleanup()
        return audio_path, None

    if not out.exists() or out.stat().st_size == 0:
        log.warning("ffmpeg produced no output; using the original file")
        tmp.cleanup()
        return audio_path, None

    return str(out), tmp


def _to_turns(annotation) -> list[SpeakerTurn]:
    turns = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=float(segment.start),
                end=float(segment.end),
                speaker=str(label),
            )
        )
    turns.sort(key=lambda t: t.start)
    return turns


def diarize(audio_path: str, cfg: DiarizeConfig, ceiling_mb: int) -> DiarizationResult:
    import torch
    from pyannote.audio import Pipeline

    token = _hf_token()
    if token is None:
        raise RuntimeError(
            "No Hugging Face token found. The pyannote pipelines are gated.\n"
            "  1. Accept conditions at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  2. Create a READ token at https://hf.co/settings/tokens\n"
            "  3. setx HF_TOKEN <your-token>   (then open a new terminal)"
        )

    gpu.assert_headroom(ceiling_mb, "diarize")
    t0 = time.perf_counter()

    model_name = cfg.model
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except Exception as exc:
        log.warning(
            "could not load %s (%s); falling back to %s",
            model_name, exc, cfg.fallback_model,
        )
        model_name = cfg.fallback_model
        try:
            pipeline = Pipeline.from_pretrained(model_name, token=token)
        except TypeError:
            # Older pyannote.audio used use_auth_token= instead of token=.
            pipeline = Pipeline.from_pretrained(
                model_name, use_auth_token=token
            )

    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained returned None for {model_name}. "
            f"This almost always means the gate has not been accepted for "
            f"your account, or the token lacks read scope."
        )

    if cfg.device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    else:
        log.warning("running diarization on CPU - expect ~5x slower")

    hints = {}
    if cfg.min_speakers is not None:
        hints["min_speakers"] = cfg.min_speakers
    if cfg.max_speakers is not None:
        hints["max_speakers"] = cfg.max_speakers

    log.info("diarizing with %s %s", model_name, hints or "(no speaker hints)")
    decoded, tmpdir = _decode_to_wav(audio_path)
    try:
        output = pipeline(decoded, **hints)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    # community-1 returns an object exposing .speaker_diarization; 3.1 returns
    # the Annotation directly.
    primary = getattr(output, "speaker_diarization", output)
    turns = _to_turns(primary)

    exclusive_turns = []
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        exclusive_turns = _to_turns(exclusive)
        log.info("exclusive diarization available (%d turns)", len(exclusive_turns))

    speakers = sorted({t.speaker for t in turns})
    elapsed = time.perf_counter() - t0

    log.info(
        "%d speakers, %d turns, %.1fs wall", len(speakers), len(turns), elapsed
    )
    return DiarizationResult(
        audio_path=str(audio_path),
        model=model_name,
        device=cfg.device,
        num_speakers_detected=len(speakers),
        turns=turns,
        exclusive_turns=exclusive_turns,
        elapsed_seconds=elapsed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Local speaker diarization")
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.min_speakers is not None:
        cfg.diarize.min_speakers = args.min_speakers
    if args.max_speakers is not None:
        cfg.diarize.max_speakers = args.max_speakers
    if args.device:
        cfg.diarize.device = args.device

    try:
        result = diarize(args.audio, cfg.diarize, cfg.memory.diarize_ceiling_mb)
    except Exception as exc:
        log.error("diarization failed: %s", exc, exc_info=True)
        return 1

    write_json(args.out, result)
    log.info("wrote %s", args.out)
    gpu.log_state("diarize exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
