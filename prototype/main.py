#!/usr/bin/env python3
"""
Audio-to-Notes -- command-line entry point.

A fully local, API-free pipeline that turns spoken audio into structured
notes, exactly as described in the accompanying paper:

  1. Transcribe with a local Whisper model (on-device, no API).
  2. Generate notes in one of two local modes:
       - extractive : deterministic, rule-based, cannot fabricate content.
       - llm        : local Meta Llama-3.3-70B-Instruct (4-bit), fluent prose.
       - both       : run both and write both sets of notes.
  3. Save transcript, time-stamped transcript, notes (.md and .docx),
     and the raw Whisper JSON.

Examples
--------
  python main.py --audio meeting.m4a
  python main.py --audio hearing.wav --mode both --whisper-model medium
  python main.py --audio call.mp3 --mode llm \\
                 --llm-model meta-llama/Llama-3.3-70B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from a2n import extractive
from a2n.docx_export import markdown_to_docx
from a2n.io_utils import save_json, save_text
from a2n.transcribe import transcribe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local, API-free audio-to-notes pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", required=True, help="Path to the input audio file.")
    p.add_argument("--title", default="Audio Notes", help="Title for the notes.")
    p.add_argument(
        "--mode",
        choices=["extractive", "llm", "both"],
        default="extractive",
        help="Note-generation mode.",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper size: tiny|base|small|medium|large|turbo.",
    )
    p.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HF model id for --mode llm/both.",
    )
    p.add_argument("--language", default=None, help="Force a language code, e.g. en.")
    p.add_argument("--output-dir", default="output", help="Where to write outputs.")
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization for the LLM (needs much more VRAM).",
    )
    p.add_argument(
        "--no-docx",
        action="store_true",
        help="Skip writing the .docx file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Audio file not found: {audio_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: local transcription -------------------------------------
    print(f"[1/3] Transcribing locally with Whisper-{args.whisper_model} ...")
    result = transcribe(
        str(audio_path), model_name=args.whisper_model, language=args.language
    )
    transcript = extractive.clean_text(result.get("text", ""))
    segments = result.get("segments", [])

    save_text(out_dir / "transcript.txt", transcript)
    save_text(
        out_dir / "timestamped_transcript.txt",
        extractive.build_timestamped_transcript(segments) if segments else transcript,
    )
    save_json(out_dir / "raw_output.json", result)

    if not transcript.strip():
        sys.exit("Transcription produced no text. Check the audio file and ffmpeg.")

    # ---- Stage 2: note generation -----------------------------------------
    wrote_md = None

    if args.mode in ("extractive", "both"):
        print("[2/3] Generating extractive (rule-based) notes ...")
        md = extractive.generate_markdown_notes(
            title=args.title,
            transcript=transcript,
            segments=segments,
            whisper_model=args.whisper_model,
            audio_file=audio_path.name,
        )
        path = out_dir / "notes_extractive.md"
        save_text(path, md)
        wrote_md = path
        if not args.no_docx:
            markdown_to_docx(md, str(out_dir / "notes_extractive.docx"))

    if args.mode in ("llm", "both"):
        print(f"[2/3] Loading local LLM ({args.llm_model}) -- this can take a while ...")
        from a2n.llm_notes import generate_notes_llm, load_llm, notes_to_markdown

        tokenizer, model = load_llm(
            model_id=args.llm_model, load_in_4bit=not args.no_4bit
        )
        print("       Generating notes with the local LLM ...")
        body = generate_notes_llm(transcript, tokenizer, model)
        md = notes_to_markdown(args.title, body, audio_path.name, args.llm_model)
        path = out_dir / "notes_llm.md"
        save_text(path, md)
        wrote_md = path
        if not args.no_docx:
            markdown_to_docx(md, str(out_dir / "notes_llm.docx"))

    # ---- Stage 3: report ---------------------------------------------------
    print("[3/3] Done. Outputs written to:", out_dir.resolve())
    for f in sorted(out_dir.iterdir()):
        print("   -", f.name)
    if wrote_md:
        print("\nPreview of notes:\n")
        print(wrote_md.read_text(encoding="utf-8")[:1200])


if __name__ == "__main__":
    main()
