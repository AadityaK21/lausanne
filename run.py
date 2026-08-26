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
import re
import subprocess
import sys
import time
from typing import List, Optional, Tuple

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


# Which import each stage needs, and how to get it. Checked with find_spec so
# the orchestrator never actually imports a CUDA-linked library itself.
STAGE_REQUIREMENTS = {
    "asr": ("faster_whisper", "pip install faster-whisper",
            "speech recognition"),
    "diarize": ("pyannote.audio", "pip install pyannote.audio",
                "speaker identification"),
    "render": ("docx", "pip install python-docx", "Word export"),
}


def interpreter_tag() -> str:
    """The CPython ABI tag this interpreter loads compiled extensions for."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def abi_mismatches() -> List[Tuple[str, str]]:
    """
    Installed packages whose compiled wheels target a different Python.

    A venv is supposed to make this impossible, and it does not. `pip` on
    PATH resolves to whichever interpreter owns that pip.exe, but it honours
    VIRTUAL_ENV when choosing where to *write*. Run bare `pip install` inside
    an activated venv while a different Python's pip is first on PATH, and the
    wheels are chosen for that other interpreter and installed here. Nothing
    warns you. The venv ends up holding a mixture, and the failure surfaces
    much later as an unreadable C-extension error from whichever module
    happens to be imported first.

    Detected by reading each distribution's WHEEL metadata rather than by
    importing anything, so this stays cheap enough to run on every doctor
    invocation.

    Three wheel shapes, and only one of them can be wrong. A tag is
    `<python>-<abi>-<platform>`:

        py3-none-any            pure Python. Always fine.
        cp37-abi3-win_amd64     built against the stable ABI. Forward
                                compatible, so it is fine on any interpreter
                                at or above the version named - which is why
                                a cp37 wheel is correct on Python 3.12 and
                                flagging it would be a false alarm.
        cp312-cp312-win_amd64   locked to one interpreter version. This is the
                                only shape that can mismatch.

    A distribution is compatible if ANY of its tags is, since a wheel may
    declare several.
    """
    import re
    import sysconfig

    ours = sys.version_info[:2]

    def tag_ok(tag: str) -> Optional[bool]:
        """True/False if the tag decides compatibility, None if irrelevant."""
        parts = tag.split("-")
        if len(parts) < 2:
            return None
        py, abi = parts[0], parts[1]
        if abi == "none":
            return True                      # pure Python
        if abi == "abi3":
            # Stable ABI: usable on the named version and everything after it.
            m = re.fullmatch(r"cp3(\d+)", py)
            return True if not m else (3, int(m.group(1))) <= ours
        m = re.fullmatch(r"cp3(\d+)", abi.split("t")[0])
        if not m:
            return None                      # pypy, graalpy, unrecognised
        return (3, int(m.group(1))) == ours

    roots = {
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
    }
    found: dict = {}

    for root in roots:
        if not root:
            continue
        base = pathlib.Path(root)
        if not base.is_dir():
            continue
        for info in base.glob("*.dist-info"):
            try:
                text = (info / "WHEEL").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            tags = re.findall(r"^Tag:\s*(\S+)", text, re.M)
            verdicts = [tag_ok(t) for t in tags]
            decided = [v for v in verdicts if v is not None]
            if not decided or any(decided):
                continue
            name = info.name.split("-")[0]
            found.setdefault(name, tags[0].split("-")[1])

    return sorted(found.items())


def import_probe(module: str, timeout: int = 180) -> Tuple[bool, str]:
    """
    Can `module` actually be imported? Answered in a throwaway subprocess.

    In-process would be one line and would be wrong. The orchestrator's
    defining property is that it never loads a model-bearing library: this
    process must not hold torch, CUDA, or CTranslate2, because the entire
    memory design depends on the only things touching the GPU being
    short-lived children (docs/MEMORY.md). An import here to "just check"
    would put a CUDA context in the one process that must not have one, and on
    Windows would additionally pin the CUDA DLLs for the run's lifetime.

    A subprocess answers exactly the same question and takes the consequences
    with it when it exits. That it costs a second is the point rather than a
    regret - process isolation is the mechanism this project is built on, and
    the preflight check should not be the one place that quietly opts out.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True, text=True, timeout=timeout, env=stage_env(),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"could not run the import probe: {exc}"

    if proc.returncode == 0:
        return True, ""

    # The last non-empty traceback line is the exception; the rest is noise.
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    return False, (lines[-1] if lines else f"exit {proc.returncode}")


def missing_requirement(stage: str) -> Optional[str]:
    """
    Return an actionable message if a stage's dependency is absent or broken.

    Two distinct failures needing different advice, and `find_spec` answers
    only the first. It answers the second wrongly: a package whose compiled
    extension cannot load still has a perfectly good spec, so an earlier
    version of this reported an ABI-mismatched install as present and let the
    subprocess die later with a NumPy C-extension traceback that named no fix.
    """
    entry = STAGE_REQUIREMENTS.get(stage)
    if entry is None:
        return None
    module, install, purpose = entry

    ok, err = import_probe(module)
    if ok:
        return None

    # "No module named 'faster_whisper'" is absence; "No module named 'numpy.
    # _core._multiarray_umath'" from the same probe is a broken install, and
    # telling the user to reinstall faster-whisper would send them round the
    # loop again.
    top = module.split(".")[0]
    absent = (
        "ModuleNotFoundError" in err
        and re.search(rf"No module named '{re.escape(top)}'", err) is not None
    )

    if absent:
        return (
            f"{stage}: {module!r} is not installed in this interpreter, so "
            f"{purpose} cannot run.\n"
            f"    Fix:  python -m pip install {install.split('install ')[1]}\n"
            f"    Note: use 'python -m pip', not bare 'pip' - a bare pip can "
            f"install into a different interpreter than the one running this.\n"
            f"    Interpreter in use: {sys.executable}"
        )

    bad = abi_mismatches()
    lines = [
        f"{stage}: {module!r} is installed but will not import, so {purpose} "
        f"cannot run.",
        f"    Error: {err[:200]}",
    ]
    if bad:
        lines += [
            f"    Cause: this environment is Python {interpreter_tag()}, but "
            f"{len(bad)} package(s) were built for another version:",
            "      " + ", ".join(f"{n} ({t})" for n, t in bad[:8]),
            "    This happens when bare 'pip' resolves to a different Python "
            "than the venv.",
            "    Fix: rebuild the environment with ONE interpreter and use "
            "'python -m pip' throughout.",
            "    See the 'Mixed Python versions' section of "
            "docs/SETUP_WINDOWS.md.",
        ]
    else:
        lines.append(
            "    Fix: python -m pip install --force-reinstall "
            f"{module.replace('_', '-')}"
        )
    lines.append(f"    Interpreter in use: {sys.executable}")
    return "\n".join(lines)


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
    """
    Report what this installation can actually do.

    Grouped by capability rather than as one pass/fail list, because the
    dependencies are genuinely optional and independent. An earlier version
    printed NOT READY when speaker identification was absent, which is wrong
    and discouraging: transcription and note generation work perfectly well
    without it, and telling someone their setup is broken when it is merely
    partial sends them off installing several gigabytes they may not need.
    """
    import importlib.util

    def have(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    def version(module: str) -> str:
        try:
            mod = __import__(module)
            return getattr(mod, "__version__", "?")
        except Exception:
            return "?"

    print("\n" + "=" * 68)
    print("  LAUSANNE ENVIRONMENT CHECK")
    print("=" * 68 + "\n")

    print(f"  Python  : {sys.version.split()[0]}  ({interpreter_tag()})")
    print(f"  Running : {sys.executable}")
    if not (hasattr(sys, "real_prefix") or sys.base_prefix != sys.prefix):
        print("  ! not in a virtual environment - packages install globally")

    # An environment holding wheels built for two different Pythons is the one
    # failure that makes every other line of this report untrustworthy: the
    # packages are present, their versions read correctly, and they cannot be
    # imported. Report it first and in full, because the error it eventually
    # produces names a C extension rather than a cause.
    bad = abi_mismatches()
    if bad:
        print()
        print("  " + "!" * 64)
        print(f"  MIXED PYTHON VERSIONS - this environment is broken")
        print("  " + "!" * 64)
        print(f"    This venv runs Python {sys.version.split()[0]} "
              f"({interpreter_tag()}), but {len(bad)} installed package(s)")
        print("    were compiled for a different one:")
        for name, tag in bad[:12]:
            print(f"      {name:<28} built for {tag}")
        if len(bad) > 12:
            print(f"      ... and {len(bad) - 12} more")
        print()
        print("    Cause: bare 'pip' resolves to whichever Python owns the")
        print("    first pip.exe on PATH, but writes into the ACTIVE venv. Run")
        print("    it inside a venv built on a different version and you get")
        print("    exactly this mixture, silently.")
        print()
        print("    Fix: rebuild the environment on one interpreter -")
        print("      deactivate")
        print("      Remove-Item -Recurse -Force .venv")
        print("      py -3.12 -m venv .venv")
        print("      .\\.venv\\Scripts\\Activate.ps1")
        print("      python -m pip install --upgrade pip")
        print("    then install with 'python -m pip', never bare 'pip'.")
        print("    See docs/SETUP_WINDOWS.md, 'Mixed Python versions'.")

    state = gpu.query()
    if state:
        print(f"  GPU     : {state.name}")
        print(f"            {state.free_mb} MB free of {state.total_mb} MB")
        for p in cfg.memory.check():
            print(f"    x budget: {p}")
    else:
        print("  GPU     : none detected - stages will run on CPU")

    # ---- capability groups
    core_mods = [("faster_whisper", "faster-whisper"),
                 ("pydantic", "pydantic"), ("requests", "requests"),
                 ("yaml", "PyYAML"), ("docx", "python-docx")]
    core_missing = [pkg for mod, pkg in core_mods if not have(mod)]

    print("\n  " + "-" * 64)
    print("  TRANSCRIPTION AND RULE-BASED NOTES")
    for mod, pkg in core_mods:
        mark = "OK " if have(mod) else "-- "
        ver = f"  {version(mod)}" if have(mod) else ""
        print(f"    [{mark}] {pkg}{ver}")
    if core_missing:
        print(f"    NOT AVAILABLE -> python -m pip install "
              f"-r requirements-minimal.txt")
    else:
        print("    AVAILABLE")

    # cuDNN discoverability. Reported here because the failure it causes -
    # "Could not locate cudnn_ops64_9.dll" - surfaces during decoding, several
    # minutes into a run, and names a file rather than a cause.
    if sys.platform == "win32":
        found, where, needed = gpu.cudnn_status()
        if found:
            print(f"    [OK ] cuDNN/cuBLAS found in {where}")
        else:
            print("    [-- ] cuDNN/cuBLAS not found - GPU transcription will")
            print(f"           fall back to CPU. Missing: {', '.join(needed)}")
            print("           These ship inside the CUDA build of PyTorch:")
            print("             python -m pip install torch torchaudio "
                  "--index-url https://download.pytorch.org/whl/cu126")
            print("           No PATH editing is needed - the stage registers")
            print("           torch's DLL directory itself at startup.")

    diar_ok = have("pyannote.audio") and have("torch")
    token = any(os.environ.get(v) for v in
                ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    print("\n  SPEAKER IDENTIFICATION  (optional)")
    print(f"    [{'OK ' if have('pyannote.audio') else '-- '}] pyannote.audio")
    print(f"    [{'OK ' if have('torch') else '-- '}] torch")
    print(f"    [{'OK ' if token else '-- '}] HF_TOKEN")
    if diar_ok and token:
        print("    AVAILABLE")
    else:
        print("    NOT AVAILABLE -> run with --no-diarize; every utterance")
        print("      becomes SPEAKER_00. See docs/SETUP_WINDOWS.md steps 2 and 4.")

    tier = cfg.tier()
    client = OllamaClient(host=tier.host)
    ollama_up = client.is_up()
    models = client.list_models() if ollama_up else []
    gen_ok = ollama_up and client.has_model(tier.model)
    judge_ok = ollama_up and client.has_model(cfg.judge.model)

    print("\n  MODEL-SELECTED AND ABSTRACTIVE NOTES  (optional)")
    print(f"    [{'OK ' if ollama_up else '-- '}] Ollama at {tier.host}")
    if models:
        print(f"          pulled: {', '.join(models)}")
    print(f"    [{'OK ' if gen_ok else '-- '}] {tier.model}  (generator)")
    if gen_ok:
        print("    AVAILABLE")
    elif ollama_up:
        print(f"    NOT AVAILABLE -> ollama pull {tier.model}")
    else:
        print("    NOT AVAILABLE -> start Ollama from the tray")

    # Cloud-tagged models are a confidentiality hazard, not a convenience.
    cloud = [m for m in models if prov.is_cloud_model(m)]
    if cloud:
        print("\n    ! CLOUD MODELS PRESENT: " + ", ".join(cloud))
        print("      These execute on the provider's servers even though they")
        print("      are reached at a local URL. A privileged run will refuse")
        print("      them; do not select one for material you would not publish.")

    print("\n  EVALUATION HARNESS  (optional)")
    print(f"    [{'OK ' if judge_ok else '-- '}] {cfg.judge.model}  (judge)")
    if judge_ok:
        print("    AVAILABLE")
    else:
        print(f"    NOT AVAILABLE -> ollama pull {cfg.judge.model}")
        print("      Only needed to score abstractive faithfulness. WER, DER,")
        print("      and verbatim rate need no model at all.")

    # Model residency.
    #
    # An earlier version of this check read OLLAMA_MAX_LOADED_MODELS and
    # OLLAMA_KEEP_ALIVE out of os.environ and warned when they were unset.
    # That was measuring the wrong process. Those variables are read by the
    # Ollama *server* when it starts - typically from the tray, long before
    # this shell existed - so their value here says nothing about how the
    # server is behaving, and setting them in this shell does not change it.
    # Users correctly set them, restarted nothing, and were told they had
    # failed.
    #
    # The server's actual state is observable through /api/ps, so ask it.
    if ollama_up:
        resident = client.resident()
        print("\n  MODEL RESIDENCY")
        if resident:
            names = ", ".join(
                m.get("name", "?") for m in resident
            )
            mb = sum(m.get("size_vram", 0) for m in resident) / (1024 * 1024)
            print(f"    {len(resident)} model(s) resident in VRAM: {names}")
            if mb:
                print(f"    holding roughly {mb:.0f} MB")
            print("    The ASR stage needs that memory. It will be freed")
            print("    before a run - every LLM call this project makes sets")
            print("    keep_alive=0 explicitly, which overrides the server")
            print("    default - but anything you loaded by hand stays until")
            print("    evicted:  ollama stop <model>")
        else:
            print("    nothing resident - all of VRAM is available to ASR")

        env_set = [
            v for v in ("OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE")
            if os.environ.get(v) is not None
        ]
        if env_set:
            print(f"    ({', '.join(env_set)} set in this shell; note the")
            print("     server only reads these when IT starts, so restart")
            print("     Ollama from the tray if you just set them.)")

    # ---- what you can actually run
    print("\n" + "=" * 68)
    if bad:
        print("  Rebuild the environment before anything else - the package")
        print("  mixture above means nothing here can be trusted to import.")
        print("=" * 68 + "\n")
        return 1
    if core_missing:
        print("  Install the core set before anything else:")
        print("    python -m pip install -r requirements-minimal.txt")
        print("=" * 68 + "\n")
        return 1

    print("  YOU CAN RUN:")
    diar_flag = "" if (diar_ok and token) else " --no-diarize"
    print(f"    python app\\server.py")
    print(f"    python run.py <audio> --mode extractive-rule{diar_flag}")
    if gen_ok:
        print(f"    python run.py <audio> --mode extractive-llm{diar_flag}")
        print(f"    python run.py <audio> --mode abstractive{diar_flag}")
    print("=" * 68 + "\n")
    return 0


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
            prov.guard_tier(material, tier_name, tier.host, tier.model)
        except prov.TierViolation as exc:
            log.error("%s", exc)
            return 3

    manifest = prov.new_manifest(work.name, str(audio), material)

    log.info("audio : %s", audio)
    log.info("work  : %s", work)
    log.info("output: %s", out)
    log.info("class : %s | tier: %s", material.upper(), tier_name)
    gpu.log_state("start")

    problem = missing_requirement("asr")
    if problem and not transcript.exists():
        log.error("%s", problem)
        return 4

    asr_argv = ["src/stages/asr.py", str(audio), "--out", str(transcript)]
    if args.device:
        asr_argv += ["--device", args.device]
    if args.asr_model:
        asr_argv += ["--model", args.asr_model]
    if not run_stage("asr", asr_argv, skip_if=transcript, force=args.force):
        return 1
    _record(manifest, "asr", transcript, ASRResult, cfg)

    diar_problem = missing_requirement("diarize")
    if not args.no_diarize and diar_problem and not speakers.exists():
        log.warning(
            "%s\n    Continuing without speaker attribution; pass "
            "--no-diarize to silence this.", diar_problem,
        )
        args.no_diarize = True

    if not args.no_diarize:
        diar_argv = ["src/stages/diarize.py", str(audio), "--out", str(speakers)]
        if args.device:
            diar_argv += ["--device", args.device]
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

    # ---- Optional: speaker-label bias probe -------------------------------
    # Opt-in because it costs one full selection pass per condition. It is the
    # only measurement in the project that asks whether the notes are
    # even-handed rather than whether they are correct, so it belongs on the
    # main entry point rather than buried in the eval directory - but it is not
    # something to run by default on every hearing.
    if getattr(args, "bias_probe", False):
        if "extractive-llm" not in produced:
            log.warning(
                "--bias-probe needs the extractive-llm mode; skipping "
                "(re-run with --mode extractive-llm)"
            )
        else:
            bias_path = work / "bias.json"
            run_stage(
                "bias-probe",
                ["src/eval/bias.py", "--merged", str(merged),
                 "--out", str(bias_path), "--tier", tier_name],
                skip_if=bias_path, force=args.force,
            )
            if bias_path.exists():
                _record(manifest, "bias-probe", bias_path, None, cfg,
                        endpoint=tier.host)

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

    `model_cls` may be None for an artefact that is not a single stage
    contract - the bias probe writes a list of conditions. The manifest still
    hashes the file and records the endpoint, because provenance covers every
    artefact a run produced, not only the ones that happen to be one model.
    """
    obj = None
    if model_cls is not None:
        try:
            obj = read_json(artifact, model_cls)
        except Exception as exc:
            log.warning("could not record provenance for %s: %s", stage, exc)
            return
    elif not pathlib.Path(artifact).exists():
        log.warning("could not record provenance for %s: no artefact", stage)
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
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip speaker identification (no Hugging Face token "
                         "needed)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="force a device; CPU needs no CUDA install")
    ap.add_argument("--asr-model", default=None,
                    help="override the Whisper model, e.g. small or base")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even if artifacts exist")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--material", default="privileged", choices=["privileged", "public"],
        help="Confidentiality class of the input. Defaults to privileged, "
             "which forbids any non-local inference endpoint.",
    )
    ap.add_argument(
        "--bias-probe", action="store_true",
        help="After the notes are produced, re-run selection over transcripts "
             "whose text is identical and whose speaker labels have been "
             "anonymised and permuted, and report whether who is speaking "
             "changes what gets selected. Costs one selection pass per "
             "condition.",
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
