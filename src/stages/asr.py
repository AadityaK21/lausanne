"""
Stage 1 - Local speech recognition with faster-whisper.

Runs as its own process and exits. CTranslate2 (faster-whisper's backend)
does release GPU memory on interpreter exit, but PyTorch - loaded later by
pyannote - does not reliably return CUDA memory in-process. Keeping every
model in a separate, short-lived process is what makes the three-model
pipeline fit in 8 GB of VRAM at all.

Run standalone:
    python src/stages/asr.py data/audio/hearing.wav --out work/hearing/transcript.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.config import load_config, ASRConfig            # noqa: E402
from src.schemas import ASRResult, ASRSegment, Word      # noqa: E402
from src.util import gpu                                  # noqa: E402
from src.util.io import write_json                        # noqa: E402
from src.util.logging import get_logger                   # noqa: E402

log = get_logger("asr")


def _audio_duration(path: str) -> float:
    """Duration in seconds. Tries soundfile, then librosa, then ffprobe."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            text=True, timeout=60,
        )
        return float(out.strip())
    except Exception as exc:
        log.warning("could not determine audio duration: %s", exc)
        return 0.0


def transcribe(audio_path: str, cfg: ASRConfig, ceiling_mb: int) -> ASRResult:
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    gpu.assert_headroom(ceiling_mb, "asr")
    duration = _audio_duration(audio_path)
    log.info(
        "transcribing %s (%.1f min) with %s / %s",
        pathlib.Path(audio_path).name, duration / 60, cfg.model, cfg.compute_type,
    )

    t0 = time.perf_counter()
    model = WhisperModel(
        cfg.model, device=cfg.device, compute_type=cfg.compute_type
    )

    decode_kwargs = dict(
        beam_size=cfg.beam_size,
        temperature=cfg.temperature,
        word_timestamps=cfg.word_timestamps,
        language=cfg.language,
        condition_on_previous_text=cfg.condition_on_previous_text,
        initial_prompt=cfg.initial_prompt,
        vad_filter=cfg.vad_filter,
    )

    # Batched inference is markedly faster on long audio, but not every
    # faster-whisper build accepts every kwarg through the batched path.
    try:
        pipeline = BatchedInferencePipeline(model=model)
        segments_iter, info = pipeline.transcribe(
            audio_path, batch_size=cfg.batch_size, **decode_kwargs
        )
        log.info("using batched inference (batch_size=%d)", cfg.batch_size)
    except Exception as exc:
        log.warning("batched path unavailable (%s); falling back to sequential", exc)
        segments_iter, info = model.transcribe(audio_path, **decode_kwargs)

    segments = []
    for i, seg in enumerate(segments_iter):
        words = []
        for w in (getattr(seg, "words", None) or []):
            words.append(
                Word(
                    start=float(w.start),
                    end=float(w.end),
                    word=w.word,
                    probability=float(getattr(w, "probability", 0.0) or 0.0),
                )
            )
        segments.append(
            ASRSegment(
                id=i,
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text,
                words=words,
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                compression_ratio=float(
                    getattr(seg, "compression_ratio", 0.0) or 0.0
                ),
            )
        )
        if i % 25 == 0 and i:
            log.info("  %d segments (%.1f min of audio)", i, seg.end / 60)

    elapsed = time.perf_counter() - t0
    if duration <= 0 and segments:
        duration = segments[-1].end

    result = ASRResult(
        audio_path=str(audio_path),
        audio_duration=duration,
        language=getattr(info, "language", cfg.language or "en"),
        language_probability=float(getattr(info, "language_probability", 1.0)),
        model=cfg.model,
        compute_type=cfg.compute_type,
        device=cfg.device,
        segments=segments,
        elapsed_seconds=elapsed,
        rtf=(elapsed / duration) if duration > 0 else 0.0,
    )
    log.info(
        "%d segments, %d words, %.1fs wall (RTF %.3f)",
        len(segments),
        sum(len(s.words) for s in segments),
        elapsed,
        result.rtf,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Local Whisper transcription")
    ap.add_argument("audio", help="path to audio file (.wav/.mp3/.m4a)")
    ap.add_argument("--out", required=True, help="output transcript.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--language", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.asr.model = args.model
    if args.compute_type:
        cfg.asr.compute_type = args.compute_type
    if args.device:
        cfg.asr.device = args.device
    if args.language:
        cfg.asr.language = args.language

    if not pathlib.Path(args.audio).exists():
        log.error("audio file not found: %s", args.audio)
        return 2

    try:
        result = transcribe(args.audio, cfg.asr, cfg.memory.asr_ceiling_mb)
    except Exception as exc:
        log.error("ASR failed: %s", exc, exc_info=True)
        return 1

    write_json(args.out, result)
    log.info("wrote %s", args.out)
    gpu.log_state("asr exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
