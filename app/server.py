#!/usr/bin/env python
"""
Lausanne - local application server.

A single-file HTTP server built on the Python standard library. No web
framework, deliberately.

Two reasons. A research artefact that requires FastAPI, uvicorn, and their
transitive dependency tree to reproduce stops reproducing the moment one of
those makes a breaking release; `http.server` has been in the standard library
since Python 2 and will outlive this project. And the target machine is already
carrying a fragile CUDA/cuDNN installation, where every additional dependency is
another thing that can silently break the pipeline.

DP1 is enforced here structurally, not by convention: the server binds to the
loopback interface and refuses to bind anything else. A tool whose entire
premise is that privileged audio never leaves the machine must not be one
mistyped flag away from serving that audio to a network.

Usage:
    python app/server.py                 # real pipeline
    python app/server.py --demo          # synthetic fixtures, no models needed
    python app/server.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from email.parser import BytesParser
from email.policy import default as email_default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                                  # noqa: E402
from src.schemas import MergedTranscript, NoteSet                    # noqa: E402
from src.util import provenance as prov                              # noqa: E402
from src.util.io import ensure_dir, read_json, run_dir, write_json    # noqa: E402
from src.util.logging import get_logger                              # noqa: E402

log = get_logger("app")

STATIC = pathlib.Path(__file__).resolve().parent / "static"

# The only interfaces this server will bind. See module docstring.
ALLOWED_BINDS = {"127.0.0.1", "localhost", "::1"}

STAGES = [
    ("asr", "Transcribing"),
    ("diarize", "Identifying speakers"),
    ("merge", "Reconciling"),
    ("notes", "Selecting notes"),
    ("render", "Rendering"),
]


# --------------------------------------------------------------------------
# Job state
# --------------------------------------------------------------------------


class Job:
    """
    One pipeline run. Holds progress state and an event fan-out for the UI.

    Deliberately a plain object guarded by a lock rather than anything clever:
    there is exactly one job at a time, because the whole architecture exists
    to keep one model resident at a time.
    """

    def __init__(self, job_id: str, audio: pathlib.Path, mode: str,
                 material: str, tier: str, demo: bool):
        self.id = job_id
        self.audio = audio
        self.mode = mode
        self.material = material
        self.tier = tier
        self.demo = demo

        self.status = "queued"          # queued | running | done | failed | cancelled
        self.stage = ""
        self.stage_label = ""
        self.progress = 0.0
        self.message = ""
        self.error: Optional[str] = None
        self.started = time.time()
        self.finished: Optional[float] = None
        self.log_lines: List[str] = []
        self.work: Optional[pathlib.Path] = None

        self._lock = threading.Lock()
        self._subscribers: List[queue.Queue] = []
        self._cancel = threading.Event()

    # ---- event fan-out

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        q.put(self.snapshot())
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self) -> None:
        snap = self.snapshot()
        with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(snap)
                except queue.Full:
                    # A stalled browser tab must not stall the pipeline.
                    pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
            "mode": self.mode,
            "material": self.material,
            "tier": self.tier,
            "demo": self.demo,
            "audio_name": self.audio.name,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "log": self.log_lines[-40:],
        }

    def set(self, *, stage: str = None, label: str = None,
            progress: float = None, message: str = None,
            status: str = None) -> None:
        if stage is not None:
            self.stage = stage
        if label is not None:
            self.stage_label = label
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        if status is not None:
            self.status = status
        self._emit()

    def log_line(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self.log_lines.append(line)
            self._emit()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()


# --------------------------------------------------------------------------
# Pipeline execution
# --------------------------------------------------------------------------


class Runner:
    def __init__(self, cfg, demo: bool):
        self.cfg = cfg
        self.demo = demo
        self.job: Optional[Job] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def busy(self) -> bool:
        return self.job is not None and self.job.status in ("queued", "running")

    def start(self, audio: pathlib.Path, mode: str, material: str,
              tier: str) -> Job:
        with self._lock:
            if self.busy():
                raise RuntimeError("a run is already in progress")
            job = Job(uuid.uuid4().hex[:12], audio, mode, material, tier,
                      self.demo)
            self.job = job
            self._thread = threading.Thread(
                target=self._execute, args=(job,), daemon=True
            )
            self._thread.start()
            return job

    # ---- the actual run

    def _execute(self, job: Job) -> None:
        try:
            job.set(status="running", progress=0.02, message="Preparing")

            # DP1: refuse before any inference, exactly as the CLI does. The
            # UI must not become a way around the guard.
            tier = self.cfg.tier(job.tier)
            if job.mode != "extractive-rule":
                prov.guard_tier(job.material, job.tier, tier.host, tier.model)

            work = run_dir(self.cfg.work_dir, job.audio)
            job.work = work
            out = ensure_dir(pathlib.Path(self.cfg.output_dir) / work.name)

            manifest = prov.new_manifest(work.name, str(job.audio), job.material)

            transcript = work / "transcript.json"
            speakers = work / "speakers.json"
            merged = work / "merged.json"
            notes = work / f"notes_{job.mode}.json"

            if self.demo:
                self._demo_run(job, work, out, manifest, transcript,
                               speakers, merged, notes, tier)
            else:
                self._real_run(job, work, out, manifest, transcript,
                               speakers, merged, notes, tier)

            if job.cancelled:
                job.set(status="cancelled", message="Cancelled")
                return

            prov.attest(manifest)
            write_json(work / "provenance.json", manifest)
            (out / "attestation.txt").write_text(
                prov.render(manifest), encoding="utf-8"
            )

            job.finished = time.time()
            job.set(status="done", progress=1.0, stage="", 
                    message="Complete")
            log.info("job %s complete in %.1fs", job.id, job.elapsed
                     if hasattr(job, "elapsed") else 0)

        except prov.TierViolation as exc:
            job.finished = time.time()
            job.error = str(exc)
            job.set(status="failed", message="Refused: confidentiality guard")
            log.warning("job %s refused: %s", job.id, exc)
        except Exception as exc:  # noqa: BLE001
            job.finished = time.time()
            job.error = f"{type(exc).__name__}: {exc}"
            job.set(status="failed", message="Failed")
            log.error("job %s failed: %s", job.id, exc, exc_info=True)

    def _stage_cmd(self, job: Job, name: str, label: str, argv: List[str],
                   base: float, span: float) -> bool:
        """Run one stage as a subprocess, streaming its stderr into the UI."""
        if job.cancelled:
            return False
        job.set(stage=name, label=label, progress=base, message=label)

        env = dict(os.environ)
        env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
        env.setdefault("OLLAMA_KEEP_ALIVE", "0")
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc = subprocess.Popen(
            [sys.executable, "-u"] + argv, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        ticks = 0
        tail: List[str] = []
        for line in proc.stdout:
            if job.cancelled:
                proc.terminate()
                return False
            job.log_line(line)
            # Keep the end of the stage's output. "exit 1" on its own tells the
            # user nothing; the cause is always in the last few lines, and
            # making them dig through a log pane for it is a failure of the
            # interface, not of their patience.
            tail.append(line.rstrip())
            if len(tail) > 12:
                tail.pop(0)
            # No stage reports true percentage, so creep toward the next
            # boundary on output rather than inventing a fake number.
            ticks += 1
            job.set(progress=min(base + span * 0.9,
                                 base + span * (1 - 0.985 ** ticks)))
        proc.wait()
        if proc.returncode != 0:
            detail = "\n".join(
                t for t in tail
                if t.strip() and not t.strip().startswith("Traceback")
            )
            raise RuntimeError(
                f"The {label.lower()} stage failed (exit {proc.returncode}).\n\n"
                f"{detail or '(no output captured)'}"
            )
        job.set(progress=base + span)
        return True

    def _real_run(self, job, work, out, manifest, transcript, speakers,
                  merged, notes, tier) -> None:
        self._stage_cmd(
            job, "asr", "Transcribing",
            ["src/stages/asr.py", str(job.audio), "--out", str(transcript)],
            0.02, 0.40,
        )
        prov.record_stage(manifest, "asr", endpoint="", artifact=str(transcript))

        try:
            self._stage_cmd(
                job, "diarize", "Identifying speakers",
                ["src/stages/diarize.py", str(job.audio), "--out", str(speakers)],
                0.42, 0.22,
            )
            prov.record_stage(manifest, "diarize", endpoint="",
                              artifact=str(speakers))
        except RuntimeError as exc:
            job.log_line(f"diarization unavailable ({exc}); continuing without "
                         f"speaker attribution")

        merge_argv = ["src/stages/merge.py", "--transcript", str(transcript),
                      "--out", str(merged)]
        if speakers.exists():
            merge_argv += ["--speakers", str(speakers)]
        self._stage_cmd(job, "merge", "Reconciling", merge_argv, 0.64, 0.04)
        prov.record_stage(manifest, "merge", endpoint="", artifact=str(merged))

        if job.mode == "extractive-rule":
            argv = ["src/stages/extract.py", "--merged", str(merged),
                    "--out", str(notes)]
            endpoint = ""
        else:
            argv = ["src/stages/summarize.py", "--merged", str(merged),
                    "--mode", job.mode, "--out", str(notes),
                    "--tier", job.tier]
            endpoint = tier.host
        self._stage_cmd(job, "notes", "Selecting notes", argv, 0.68, 0.26)
        prov.record_stage(manifest, f"notes:{job.mode}", endpoint=endpoint,
                          artifact=str(notes))

        self._stage_cmd(
            job, "render", "Rendering",
            ["src/stages/render.py", "--merged", str(merged),
             "--notes", str(notes), "--outdir", str(out)],
            0.94, 0.05,
        )

    def _demo_run(self, job, work, out, manifest, transcript, speakers,
                  merged, notes, tier) -> None:
        """
        Run the real CPU stages over synthetic fixtures.

        This is not a mock of the UI. Merge, extraction, and rendering are the
        production code paths; only the two GPU stages are replaced by
        pre-built fixtures. It exists so the interface can be exercised and
        tested before CUDA is installed, and so CI can cover it.
        """
        sys.path.insert(0, str(ROOT / "tests"))
        from test_pipeline import build_fixtures  # noqa: PLC0415

        asr, diar, _gold = build_fixtures()

        job.set(stage="asr", label="Transcribing", progress=0.05,
                message="Transcribing (demo fixture)")
        job.log_line("demo mode: using synthetic ASR fixture, no model loaded")
        for i in range(6):
            if job.cancelled:
                return
            time.sleep(0.18)
            job.set(progress=0.05 + 0.30 * (i + 1) / 6)
        write_json(transcript, asr)
        prov.record_stage(manifest, "asr", model=asr.model, device="demo",
                          endpoint="", artifact=str(transcript))

        job.set(stage="diarize", label="Identifying speakers", progress=0.38)
        job.log_line("demo mode: using synthetic diarization fixture")
        for i in range(4):
            if job.cancelled:
                return
            time.sleep(0.15)
            job.set(progress=0.38 + 0.20 * (i + 1) / 4)
        write_json(speakers, diar)
        prov.record_stage(manifest, "diarize", model=diar.model, device="demo",
                          endpoint="", artifact=str(speakers))

        # From here the real code runs.
        merge_argv = ["src/stages/merge.py", "--transcript", str(transcript),
                      "--speakers", str(speakers), "--out", str(merged)]
        self._stage_cmd(job, "merge", "Reconciling", merge_argv, 0.60, 0.06)
        prov.record_stage(manifest, "merge", endpoint="", artifact=str(merged))

        if job.mode == "extractive-rule":
            argv = ["src/stages/extract.py", "--merged", str(merged),
                    "--out", str(notes)]
            self._stage_cmd(job, "notes", "Selecting notes", argv, 0.66, 0.26)
            prov.record_stage(manifest, f"notes:{job.mode}", endpoint="",
                              artifact=str(notes))
        else:
            job.log_line(
                f"demo mode: {job.mode} needs a language model, which is not "
                f"loaded. Falling back to extractive-rule so the interface "
                f"still has notes to show."
            )
            job.mode = "extractive-rule"
            notes = work / "notes_extractive-rule.json"
            argv = ["src/stages/extract.py", "--merged", str(merged),
                    "--out", str(notes)]
            self._stage_cmd(job, "notes", "Selecting notes", argv, 0.66, 0.26)
            prov.record_stage(manifest, "notes:extractive-rule", endpoint="",
                              artifact=str(notes))

        self._stage_cmd(
            job, "render", "Rendering",
            ["src/stages/render.py", "--merged", str(merged),
             "--notes", str(notes), "--outdir", str(out)],
            0.92, 0.06,
        )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "Lausanne"
    protocol_version = "HTTP/1.1"

    runner: Runner = None       # injected
    cfg = None                  # injected

    def log_message(self, fmt, *args):  # quieter default logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    # ---- helpers

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ---- GET

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            return self._serve_static("index.html")
        if route.startswith("/static/"):
            return self._serve_static(route[len("/static/"):])
        if route == "/api/config":
            return self._api_config()
        if route == "/api/state":
            job = self.runner.job
            return self._json(job.snapshot() if job else {"status": "idle"})
        if route == "/api/events":
            return self._api_events()
        if route == "/api/result":
            return self._api_result(parse_qs(parsed.query))
        if route == "/api/audio":
            return self._api_audio(parse_qs(parsed.query))
        if route == "/api/download":
            return self._api_download(parse_qs(parsed.query))
        return self._error("not found", 404)

    def _serve_static(self, rel: str) -> None:
        # Contain path traversal: resolve and confirm the result is inside
        # the static directory before opening anything.
        target = (STATIC / unquote(rel)).resolve()
        try:
            target.relative_to(STATIC.resolve())
        except ValueError:
            return self._error("forbidden", 403)
        if not target.exists() or not target.is_file():
            return self._error("not found", 404)

        data = target.read_bytes()
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _api_config(self) -> None:
        cfg = self.cfg
        self._json({
            "demo": self.runner.demo,
            "tiers": {
                name: {
                    "model": t.model,
                    "host": t.host,
                    "local": prov.is_local_endpoint(t.host),
                    "notes": t.notes,
                }
                for name, t in cfg.tiers.items()
            },
            "active_tier": cfg.active_tier,
            "modes": [
                {
                    "id": "extractive-rule",
                    "name": "Rule-based extraction",
                    "sub": "No model. Frequency scoring and cue patterns.",
                    "faithful": "guaranteed",
                },
                {
                    "id": "extractive-llm",
                    "name": "Model-selected extraction",
                    "sub": "A local model chooses lines; it can only return ids.",
                    "faithful": "guaranteed",
                },
                {
                    "id": "abstractive",
                    "name": "Abstractive",
                    "sub": "The model writes prose. Fluent, and can fabricate.",
                    "faithful": "not guaranteed",
                },
            ],
        })

    def _api_events(self) -> None:
        """Server-sent events. Keeps the UI live without polling."""
        job = self.runner.job
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        if job is None:
            self._sse({"status": "idle"})
            return

        q = job.subscribe()
        try:
            while True:
                try:
                    snap = q.get(timeout=15)
                except queue.Empty:
                    # Comment frame keeps proxies and browsers from timing out.
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    continue
                if not self._sse(snap):
                    return
                if snap.get("status") in ("done", "failed", "cancelled"):
                    return
        finally:
            job.unsubscribe(q)

    def _sse(self, payload: Dict[str, Any]) -> bool:
        try:
            self.wfile.write(
                f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            )
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _api_result(self, params: Dict[str, List[str]]) -> None:
        job = self.runner.job
        if job is None or job.work is None:
            return self._error("no completed run", 404)

        mode = (params.get("mode") or [job.mode])[0]
        merged_path = job.work / "merged.json"
        notes_path = job.work / f"notes_{mode}.json"

        if not merged_path.exists():
            return self._error("run has not produced a transcript yet", 404)
        if not notes_path.exists():
            return self._error(f"mode {mode} has not been run", 404)

        merged = read_json(merged_path, MergedTranscript)
        notes = read_json(notes_path, NoteSet)

        attest_path = job.work / "provenance.json"
        attestation = None
        if attest_path.exists():
            with open(attest_path, "r", encoding="utf-8") as fh:
                attestation = json.load(fh)

        available = sorted(
            p.name[len("notes_"):-len(".json")]
            for p in job.work.glob("notes_*.json")
        )

        self._json({
            "run": job.work.name,
            "mode": mode,
            # The interface must be able to state, unmissably, that the
            # transcript on screen did not come from the user's file.
            "demo": self.runner.demo,
            "available_modes": available,
            "audio_name": job.audio.name,
            "duration": merged.audio_duration,
            "speakers": merged.speaker_labels,
            "utterances": [u.model_dump(mode="json") for u in merged.utterances],
            "notes": notes.model_dump(mode="json"),
            "attestation": attestation,
        })

    def _api_audio(self, params: Dict[str, List[str]]) -> None:
        """
        Serve the run's audio with byte-range support.

        Range handling is what makes the citation-playback feature work: the
        browser seeks to a timestamp rather than downloading an hour of audio
        to hear twelve seconds of it.
        """
        job = self.runner.job
        if job is None or not job.audio.exists():
            return self._error("no audio for this run", 404)

        path = job.audio
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "audio/mpeg"

        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                partial = True

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _api_download(self, params: Dict[str, List[str]]) -> None:
        job = self.runner.job
        if job is None or job.work is None:
            return self._error("no completed run", 404)
        name = (params.get("file") or [""])[0]
        out_dir = (pathlib.Path(self.cfg.output_dir) / job.work.name).resolve()
        target = (out_dir / name).resolve()
        try:
            target.relative_to(out_dir)
        except ValueError:
            return self._error("forbidden", 403)
        if not target.exists():
            return self._error("not found", 404)

        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header(
            "Content-Disposition", f'attachment; filename="{target.name}"'
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- POST

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/run":
            return self._api_run()
        if route == "/api/upload":
            return self._api_upload()
        if route == "/api/cancel":
            job = self.runner.job
            if job:
                job.cancel()
            return self._json({"ok": True})
        return self._error("not found", 404)

    def _api_upload(self) -> None:
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._error("expected multipart/form-data")

        raw = b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + self._body()
        msg = BytesParser(policy=email_default).parsebytes(raw)

        uploads = ensure_dir(ROOT / "data" / "audio")
        saved = None
        for part in msg.iter_parts():
            filename = part.get_filename()
            if not filename:
                continue
            safe = "".join(
                c if c.isalnum() or c in "-_. " else "_" for c in filename
            ).strip() or "upload.wav"
            target = uploads / safe
            target.write_bytes(part.get_payload(decode=True) or b"")
            saved = target
            break

        if saved is None:
            return self._error("no file in upload")
        log.info("uploaded %s (%d bytes)", saved.name, saved.stat().st_size)
        self._json({"path": str(saved), "name": saved.name,
                    "bytes": saved.stat().st_size})

    def _api_run(self) -> None:
        try:
            payload = json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            return self._error("invalid JSON")

        audio = pathlib.Path(payload.get("audio", "")).expanduser()
        if not audio.exists():
            return self._error(f"audio not found: {audio}", 404)

        mode = payload.get("mode", "extractive-llm")
        material = payload.get("material", prov.PRIVILEGED)
        tier = payload.get("tier", self.cfg.active_tier)

        if material not in prov.VALID_CLASSES:
            return self._error(f"material must be one of {prov.VALID_CLASSES}")

        try:
            job = self.runner.start(audio.resolve(), mode, material, tier)
        except RuntimeError as exc:
            return self._error(str(exc), 409)
        self._json(job.snapshot())


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def serve(host: str, port: int, demo: bool, open_browser: bool,
          config_path: Optional[str]) -> int:
    if host not in ALLOWED_BINDS:
        print(
            f"\nRefusing to bind {host!r}.\n\n"
            f"This server may only bind the loopback interface "
            f"({', '.join(sorted(ALLOWED_BINDS))}). Lausanne exists to keep "
            f"privileged audio on one machine, and a tool built on that promise "
            f"must not be one mistyped flag away from serving that audio to a "
            f"network.\n\n"
            f"If you genuinely need remote access, put it behind an SSH tunnel "
            f"you control rather than changing this.\n",
            file=sys.stderr,
        )
        return 2

    cfg = load_config(config_path)
    Handler.runner = Runner(cfg, demo)
    Handler.cfg = cfg

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    url = f"http://{host}:{port}/"
    print("\n" + "=" * 66)
    print("  LAUSANNE" + ("  (demo mode - no models required)" if demo else ""))
    print("=" * 66)
    print(f"\n  Open:  {url}")
    print(f"  Bound: {host} only - not reachable from any other machine")
    if demo:
        print("\n  Demo mode runs the real merge, extraction, and rendering")
        print("  code over synthetic fixtures. The two GPU stages are")
        print("  replaced by pre-built data, so no models are needed.")
    print("\n  Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping.\n")
    finally:
        httpd.server_close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Lausanne local app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7842)
    ap.add_argument("--demo", action="store_true",
                    help="run over synthetic fixtures; no models required")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    return serve(args.host, args.port, args.demo, not args.no_browser,
                 args.config)


if __name__ == "__main__":
    raise SystemExit(main())
