#!/usr/bin/env python
"""
Lausanne - orchestrator.

Sequences the pipeline. The single most important property of this file is
what it does NOT do: it never imports torch, faster-whisper, pyannote, or any
CUDA-linked library. It shells out to each stage as a fresh interpreter and
waits for it to exit.

Why that matters on this hardware. PyTorch does not reliably return CUDA
memory to the OS within a process, and Ollama will happily keep a 5 GB model
resident between calls. With 8 GB of VRAM and 16 GB of system RAM, holding
Whisper's allocations while pyannote loads, and then loading an 8B model on
top, does not OOM cleanly on Windows - the WDDM driver spills to shared system
memory and the pipeline silently runs about fifty times slower. Process exit
is the only eviction primitive that is actually reliable here.

The design has two free side effects worth having: crash recovery (every
stage's output is on disk, so a failure resumes rather than restarts), and
independent testability of every stage.

Usage:
    python run.py doctor
    python run.py data/audio/hearing.wav
    python run.py data/audio/hearing.wav --mode abstractive --tier local
    python run.py data/audio/hearing.wav --mode all
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
from typing import List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.config import load_config, Config
from src.util import gpu
from src.schemas import ASRResult, DiarizationResult, NoteSet
from src.util.io import ensure_dir, read_json, run_dir, write_json
from src.util.logging import get_logger
from src.util.ollama_client import OllamaClient
from src.util import provenance as prov

log = get_logger("run")

REPO = pathlib.Path(__file__).resolve().parent


def stage_env() -> dict:
    """
    Environment for child stages.

    OLLAMA_* are set defensively: if the user has not set them persistently,
    a second resident model is the most likely cause of a mysterious slowdown.
    """
    env = dict(os.environ)
    env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    env.setdefault("OLLAMA_KEEP_ALIVE", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Reduces fragmentation in the ASR/diarization stages.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run_stage(name: str, argv: List[str], skip_if: Optional[pathlib.Path] = None,
              force: bool = False) -> bool:
    """
    Run one stage in a fresh interpreter. Returns True on success.

    `skip_if` enables resume: if the artifact already exists and --force was
    not passed, the stage is skipped. On a 16 GB machine a full rerun of ASR
    to fix a summarisation prompt is an expensive mistake to make twice.
    """
    if skip_if is not None and skip_if.exists() and not force:
        log.info("[%s] artifact exists, skipping (%s)", name, skip_if.name)
        return True

    cmd = [sys.executable, "-u"] + argv
    log.info("[%s] launching subprocess", name)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=stage_env())
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        log.error("[%s] FAILED with exit code %d after %.1fs",
                  name, proc.returncode, elapsed)
        return False

    log.info("[%s] ok (%.1fs)", name, elapsed)
    gpu.log_state(f"after {name}")
    return True


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def doctor(cfg: Config) -> int:
    """Pre-flight every external dependency and print an actionable report."""
    print("\n" + "=" * 68)
    print("  LAUSANNE ENVIRONMENT CHECK")
    print("=" * 68 + "\n")

    ok = True

    print(f"Python           : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 9):
        print("  ✗ Python 3.9+ required")
        ok = False

    state = gpu.query()
    if state:
        print(f"GPU              : {state.name}")
        print(f"  VRAM           : {state.total_mb} MB total, "
              f"{state.free_mb} MB free")
        problems = cfg.memory.check()
        for p in problems:
            print(f"  ✗ budget       : {p}")
            ok = False
        if not problems:
            print(f"  ✓ budget       : usable {cfg.memory.usable_vram_mb} MB "
                  f"covers all stage ceilings")
    else:
        print("GPU              : none detected (nvidia-smi unavailable)")
        print("  ! pipeline will run on CPU - expect ~10-20x slower")

    print()
    for label, module, hint in (
        ("faster-whisper", "faster_whisper", "pip install faster-whisper"),
        ("pyannote.audio", "pyannote.audio", "pip install pyannote.audio"),
        ("torch", "torch", "see docs/SETUP_WINDOWS.md for the CUDA wheel"),
        ("python-docx", "docx", "pip install python-docx"),
        ("pydantic", "pydantic", "pip install pydantic"),
        ("requests", "requests", "pip install requests"),
    ):
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✓ {label:<16} {ver}")
        except ImportError:
            print(f"  ✗ {label:<16} MISSING  ->  {hint}")
            ok = False

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ torch CUDA     {torch.version.cuda} "
                  f"({torch.cuda.get_device_name(0)})")
        else:
            print("  ✗ torch CUDA     not available - you installed the CPU "
                  "wheel; see docs/SETUP_WINDOWS.md")
            ok = False
    except ImportError:
        pass

    print()
    token = any(os.environ.get(v) for v in
                ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    if token:
        print("  ✓ HF_TOKEN       set")
    else:
        print("  ✗ HF_TOKEN       NOT SET - pyannote models are gated.")
        print("      1. Accept: https://huggingface.co/pyannote/"
              "speaker-diarization-community-1")
        print("      2. Token:  https://hf.co/settings/tokens  (read scope)")
        print("      3. setx HF_TOKEN <token>   then open a NEW terminal")
        ok = False

    print()
    tier = cfg.tier()
    client = OllamaClient(host=tier.host)
    if client.is_up():
        print(f"  ✓ Ollama         reachable at {tier.host}")
        models = client.list_models()
        print(f"      pulled: {', '.join(models) if models else '(none)'}")
        for needed, why in ((tier.model, f"tier '{tier.name}' generator"),
                            (cfg.judge.model, "evaluation judge")):
            if client.has_model(needed):
                print(f"  ✓ {needed:<16} present ({why})")
            else:
                print(f"  ✗ {needed:<16} MISSING ({why})  ->  "
                      f"ollama pull {needed}")
                ok = False
        res = client.resident()
        if res:
            names = ", ".join(m.get("name", "?") for m in res)
            print(f"  ! resident now  : {names}")
            print("      Run `ollama stop <model>` before a pipeline run, or "
                  "set OLLAMA_KEEP_ALIVE=0")
        else:
            print("  ✓ nothing resident in VRAM")
    else:
        print(f"  ✗ Ollama         NOT reachable at {tier.host}")
        print("      Start it from the tray, or run `ollama serve`")
        ok = False

    for var in ("OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE"):
        val = os.environ.get(var)
        if val is None:
            print(f"  ! {var} unset (recommended: "
                  f"{'1' if var.endswith('MODELS') else '0'})")
        else:
            print(f"  ✓ {var}={val}")

    print("\n" + "=" * 68)
    print("  READY" if ok else "  NOT READY - resolve the ✗ items above")
    print("=" * 68 + "\n")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def pipeline(args, cfg: Config) -> int:
    audio = pathlib.Path(args.audio).resolve()
    if not audio.exists():
        log.error("audio not found: %s", audio)
        return 2

    work = run_dir(cfg.work_dir, audio)
    out = ensure_dir(pathlib.Path(cfg.output_dir) / work.name)

    transcript = work / "transcript.json"
    speakers = work / "speakers.json"
    merged = work / "merged.json"

    # DP1: classification is declared up front and defaults to the safe value.
    # A user who forgets the flag gets the strict guarantee, not the loose one.
    material = args.material
    tier_name = args.tier or cfg.active_tier
    tier = cfg.tier(tier_name)

    # DP1: refuse BEFORE any inference, not after the transcript is already
    # sitting in someone else's request log.
    if args.mode != "extractive-rule":
        try:
            prov.guard_tier(material, tier_name, tier.host)
        except prov.TierViolation as exc:
            log.error("%s", exc)
            return 3

    manifest = prov.new_manifest(work.name, str(audio), material)

    log.info("audio : %s", audio)
    log.info("work  : %s", work)
    log.info("output: %s", out)
    log.info("class : %s | tier: %s", material.upper(), tier_name)
    gpu.log_state("start")

    if not run_stage(
        "asr",
        ["src/stages/asr.py", str(audio), "--out", str(transcript)],
        skip_if=transcript, force=args.force,
    ):
        return 1
    _record(manifest, "asr", transcript, ASRResult, cfg)

    if not args.no_diarize:
        diar_argv = ["src/stages/diarize.py", str(audio), "--out", str(speakers)]
        if args.min_speakers:
            diar_argv += ["--min-speakers", str(args.min_speakers)]
        if args.max_speakers:
            diar_argv += ["--max-speakers", str(args.max_speakers)]
        if not run_stage("diarize", diar_argv, skip_if=speakers, force=args.force):
            log.warning(
                "diarization failed - continuing without speaker attribution. "
                "Notes will still be produced; every utterance becomes "
                "SPEAKER_00."
            )
        else:
            _record(manifest, "diarize", speakers, DiarizationResult, cfg)

    merge_argv = ["src/stages/merge.py", "--transcript", str(transcript),
                  "--out", str(merged)]
    if speakers.exists():
        merge_argv += ["--speakers", str(speakers)]
    if not run_stage("merge", merge_argv, skip_if=merged, force=args.force):
        return 1
    prov.record_stage(
        manifest, "merge", model=None, device="cpu", endpoint="",
        artifact=str(merged),
    )

    modes = (
        ["extractive-rule", "extractive-llm", "abstractive"]
        if args.mode == "all" else [args.mode]
    )

    produced = []
    for mode in modes:
        notes_path = work / f"notes_{mode}.json"
        if mode == "extractive-rule":
            argv = ["src/stages/extract.py", "--merged", str(merged),
                    "--out", str(notes_path)]
        else:
            argv = ["src/stages/summarize.py", "--merged", str(merged),
                    "--mode", mode, "--out", str(notes_path),
                    "--tier", tier_name]
            if args.model:
                argv += ["--model", args.model]

        if not run_stage(f"notes:{mode}", argv, skip_if=notes_path,
                         force=args.force):
            log.error("mode %s failed", mode)
            if len(modes) == 1:
                return 1
            continue

        # The rule-based mode uses no model and no endpoint at all; recording
        # it as such is the point, since it is the only mode whose output could
        # not have left the host even in principle.
        endpoint = "" if mode == "extractive-rule" else tier.host
        _record(
            manifest, f"notes:{mode}", notes_path, NoteSet, cfg,
            endpoint=endpoint,
        )

        if not run_stage(
            f"render:{mode}",
            ["src/stages/render.py", "--merged", str(merged),
             "--notes", str(notes_path), "--outdir", str(out)],
        ):
            return 1
        produced.append(mode)

    prov.attest(manifest)
    write_json(work / "provenance.json", manifest)
    attestation_text = prov.render(manifest)
    (out / "attestation.txt").write_text(attestation_text, encoding="utf-8")

    gpu.log_state("finish")
    print("\n" + attestation_text)
    print("\n" + "-" * 68)
    print(f"  Done. Modes produced: {', '.join(produced) or '(none)'}")
    print(f"  Output: {out}")
    print(f"  Attestation: {out / 'attestation.txt'}")
    print("-" * 68 + "\n")
    return 0 if produced else 1


def _record(manifest, stage, artifact, model_cls, cfg, endpoint=None):
    """
    Read a completed stage's artefact and add it to the manifest.

    Reads rather than assumes: the manifest should describe what the stage
    actually did, including when it fell back to a different model or device
    than the config requested.
    """
    try:
        obj = read_json(artifact, model_cls)
    except Exception as exc:
        log.warning("could not record provenance for %s: %s", stage, exc)
        return
    prov.record_stage(
        manifest, stage,
        model=getattr(obj, "model", None),
        tier=getattr(obj, "tier", None),
        device=getattr(obj, "device", None),
        endpoint=endpoint if endpoint is not None else "",
        elapsed_seconds=getattr(obj, "elapsed_seconds", None),
        artifact=str(artifact),
    )


def attest_command(args) -> int:
    """Verify a previous run's manifest against the files on disk."""
    target = pathlib.Path(args.verify)
    if target.is_dir():
        target = target / "provenance.json"

    ok, problems = prov.verify(target)

    with open(target, "r", encoding="utf-8") as fh:
        import json as _json
        manifest = prov.RunManifest.model_validate(_json.load(fh))
    print("\n" + prov.render(manifest))

    print("\n  VERIFICATION AGAINST FILES ON DISK")
    print("  " + "-" * 66)
    if ok:
        print("  PASS - every artefact still hashes to the manifest value.")
    else:
        print("  FAIL")
        for p in problems:
            print(f"    - {p}")
    print()
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lausanne - local legal audio-to-notes pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("audio", nargs="?", help="audio file, or 'doctor'")
    ap.add_argument("--mode", default="extractive-llm",
                    choices=["extractive-rule", "extractive-llm",
                             "abstractive", "all"])
    ap.add_argument("--tier", default=None, help="local | hosted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even if artifacts exist")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--material", default="privileged", choices=["privileged", "public"],
        help="Confidentiality class of the input. Defaults to privileged, "
             "which forbids any non-local inference endpoint.",
    )
    ap.add_argument(
        "--verify", default=None, metavar="PATH",
        help="Verify a previous run's provenance.json (pass the file or its "
             "work directory) and exit.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.verify:
        return attest_command(args)

    if args.audio in (None, "doctor"):
        return doctor(cfg)

    return pipeline(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
