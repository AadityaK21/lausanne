"""
End-to-end test of the local application, with no models and no network.

Starts the real server on an ephemeral loopback port, drives it through a
complete run in demo mode, and checks the API contract the interface depends
on. Also asserts the two properties the app must not lose: that it refuses to
bind a non-loopback interface, and that it enforces the confidentiality guard
rather than offering a way around it.

Run:
    python tests/test_app.py
"""

from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL = "  \033[92mPASS\033[0m", "  \033[91mFAIL\033[0m"
_results = []


def check(label, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {label}" + (f"  — {detail}" if detail else ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.headers, r.read()


def post_json(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main() -> int:
    print("\n=== bind guard (DP1) ===")
    proc = subprocess.run(
        [sys.executable, "app/server.py", "--host", "0.0.0.0",
         "--port", str(free_port()), "--no-browser"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    check("refuses to bind 0.0.0.0", proc.returncode == 2,
          f"exit {proc.returncode}")
    check("explains why, not just 'invalid'",
          "privileged audio" in proc.stderr,
          proc.stderr.strip().splitlines()[0] if proc.stderr else "")

    # A synthetic input file. Demo mode replaces the GPU stages with fixtures,
    # so the bytes never need to be real audio.
    audio_dir = ROOT / "data" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio = audio_dir / "apptest_hearing.mp3"
    audio.write_bytes(b"ID3\x03\x00\x00\x00" + b"\x00" * 8000)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "app/server.py", "--demo", "--port", str(port),
         "--no-browser"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for the port to answer.
        ready = False
        for _ in range(80):
            try:
                get(f"{base}/api/config")
                ready = True
                break
            except Exception:
                time.sleep(0.25)
        check("server starts on loopback", ready, base)
        if not ready:
            out = server.stdout.read() if server.stdout else ""
            print(out[:2000])
            return 1

        print("\n=== static and config ===")
        status, _, body = get(f"{base}/")
        check("serves the interface", status == 200 and b"Lausanne" in body,
              f"{len(body)} bytes")
        check("no external asset hosts in the page",
              b"fonts.googleapis" not in body and b"cdn." not in body
              and b"http://" not in body.replace(b"http://127.0.0.1", b""),
              "offline-safe")

        status, _, body = get(f"{base}/api/config")
        cfg = json.loads(body)
        check("config lists three modes", len(cfg["modes"]) == 3,
              ", ".join(m["id"] for m in cfg["modes"]))
        check("config marks which modes guarantee verbatim",
              sum(1 for m in cfg["modes"] if m["faithful"] == "guaranteed") == 2)
        check("demo flag is reported", cfg["demo"] is True)

        print("\n=== path traversal ===")
        try:
            status, _, _ = get(f"{base}/static/..%2f..%2frun.py")
        except urllib.error.HTTPError as e:
            status = e.code
        check("static route refuses traversal", status in (403, 404),
              f"status {status}")

        print("\n=== confidentiality guard through the API (DP1) ===")
        status, resp = post_json(f"{base}/api/run", {
            "audio": str(audio), "mode": "abstractive",
            "material": "privileged", "tier": "hosted",
        })
        # The guard fires inside the job thread, so the POST is accepted and
        # the job then fails; either shape is acceptable as long as no work
        # against a remote endpoint proceeds.
        refused = False
        for _ in range(40):
            _, _, b = get(f"{base}/api/state")
            st = json.loads(b)
            if st.get("status") in ("failed", "done", "cancelled"):
                refused = st.get("status") == "failed" and \
                    "PRIVILEGED" in (st.get("error") or "")
                break
            time.sleep(0.2)
        check("privileged + hosted tier is refused", refused,
              "the UI cannot route around the guard")

        print("\n=== full run in demo mode ===")
        status, resp = post_json(f"{base}/api/run", {
            "audio": str(audio), "mode": "extractive-rule",
            "material": "privileged", "tier": "local",
        })
        check("run accepted", status == 200 and "id" in resp,
              resp.get("id", resp.get("error", "")))

        final = None
        for _ in range(300):
            _, _, b = get(f"{base}/api/state")
            st = json.loads(b)
            if st["status"] in ("done", "failed", "cancelled"):
                final = st
                break
            time.sleep(0.25)
        check("run completes", final and final["status"] == "done",
              (final or {}).get("error") or (final or {}).get("status"))
        if not final or final["status"] != "done":
            print((final or {}).get("log", []))
            return 1

        print("\n=== result contract ===")
        _, _, b = get(f"{base}/api/result")
        res = json.loads(b)
        check("returns utterances", len(res["utterances"]) > 10,
              f"{len(res['utterances'])} lines")
        check("returns speakers", len(res["speakers"]) >= 4,
              ", ".join(res["speakers"][:3]) + " …")
        check("returns notes", bool(res["notes"]["summary"]),
              f"{len(res['notes']['summary'])} summary lines")

        items = (res["notes"]["summary"] + res["notes"]["key_points"]
                 + res["notes"]["decisions"] + res["notes"]["action_items"])
        check("every note carries a citation",
              all(i["citations"] for i in items), f"{len(items)} items")
        check("citations resolve to real utterances",
              all(c["uid"] in {u["uid"] for u in res["utterances"]}
                  for i in items for c in i["citations"]),
              "the click-to-verify link cannot dangle")
        check("extractive notes are all marked verbatim",
              all(i["verbatim"] for i in items))

        print("\n=== attestation surfaced to the UI ===")
        att = res.get("attestation")
        check("attestation present", att is not None)
        check("attestation passes for a local run",
              att and att["attestation_ok"] is True)
        check("no data egress recorded",
              att and att["data_egress"] == [], str(att.get("data_egress")))
        check("input is hashed", bool(att and att.get("input_sha256")),
              (att.get("input_sha256") or "")[:16] if att else "")

        print("\n=== demo mode discloses that it substituted data ===")
        # A user who uploads their own recording in demo mode gets a synthetic
        # transcript. Displaying that without saying so is the exact failure
        # this project exists to prevent, so the disclosure is a tested
        # property, not a nicety.
        check("result payload flags demo", res.get("demo") is True)
        _, _, page = get(f"{base}/")
        check("interface warns before the run is started",
              b"will <b>not</b> be transcribed" in page,
              "warning sits next to the Run button")
        check("interface can state it is not the user's recording",
              b"This is not your recording." in page)
        check("demo pill is styled as a warning, not a neutral label",
              b'class="pill risk" id="demoPill"' in page)

        print("\n=== audio range requests (citation playback) ===")
        status, headers, body = get(f"{base}/api/audio")
        check("serves audio", status == 200, f"{len(body)} bytes")
        check("advertises range support",
              headers.get("Accept-Ranges") == "bytes")
        status, headers, body = get(f"{base}/api/audio",
                                    {"Range": "bytes=100-199"})
        check("honours a byte range", status == 206 and len(body) == 100,
              f"status {status}, {len(body)} bytes")
        check("reports content range",
              (headers.get("Content-Range") or "").startswith("bytes 100-199/"),
              headers.get("Content-Range", ""))

        print("\n=== downloads ===")
        status, headers, body = get(
            f"{base}/api/download?file=notes_extractive-rule.md")
        check("serves the rendered notes", status == 200 and len(body) > 200,
              f"{len(body)} bytes")
        try:
            status, _, _ = get(f"{base}/api/download?file=../../run.py")
        except urllib.error.HTTPError as e:
            status = e.code
        check("download route refuses traversal", status in (403, 404),
              f"status {status}")

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        audio.unlink(missing_ok=True)

    total, passed = len(_results), sum(_results)
    print(f"\n{'=' * 60}\n  {passed}/{total} checks passed\n{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
