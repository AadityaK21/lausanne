"""
Build the gold evaluation corpus from Oyez.

Oyez publishes US Supreme Court oral argument audio aligned to the official
transcripts, with per-turn speaker attribution and timings. That single source
gives us gold references for *both* things this pipeline needs to be measured
on:

    reference transcript  -> word error rate, including legal-term WER
    reference speaker turns -> diarization error rate

This is real legal speech - overlapping interruptions from the bench, dense
citation, Latin terminology, unfamiliar party names, variable microphone
quality - which is exactly the acoustic and lexical setting Section 8.4 says
the prototype was never tested in. It is also public, so nothing privileged is
involved and the high-resource tier can be run against it without violating
the confidentiality argument in Section 8.3.

Licence: Oyez data is CC-BY-NC (Oyez, Inc., with Cornell LII, Chicago-Kent,
and Justia). Fine for academic evaluation; cite it, do not redistribute the
audio commercially.

Usage:
    python src/eval/fetch_oyez.py --term 2019 --limit 5
    python src/eval/fetch_oyez.py --case 2019/17-1618
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.util.io import ensure_dir, write_json   # noqa: E402
from src.util.logging import get_logger          # noqa: E402

log = get_logger("oyez")

API = "https://api.oyez.org"
HEADERS = {
    "User-Agent": "lausanne-legal-ai-eval/1.0 (academic research)",
    "Accept": "application/json",
}


def _get(url: str, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning("GET %s failed (%s); retry in %ds", url, exc, wait)
            time.sleep(wait)
    return None


def list_cases(term: int) -> List[Dict[str, Any]]:
    url = f"{API}/cases?per_page=0&filter=term:{term}"
    data = _get(url)
    return data if isinstance(data, list) else []


def case_detail(term: str, docket: str) -> Dict[str, Any]:
    return _get(f"{API}/cases/{term}/{docket}")


def parse_transcript(audio_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten Oyez's nested transcript into gold turns.

    Structure:
        transcript.sections[].turns[].speaker.name
        transcript.sections[].turns[].text_blocks[].{start, stop, text}

    Timings live on the text_blocks; the turn's own start/stop is sometimes
    absent, so we derive it from the blocks it contains.
    """
    transcript = audio_obj.get("transcript") or {}
    sections = transcript.get("sections") or []

    turns: List[Dict[str, Any]] = []
    for section in sections:
        for turn in section.get("turns") or []:
            speaker = (turn.get("speaker") or {})
            name = (
                speaker.get("name")
                or speaker.get("last_name")
                or speaker.get("identifier")
                or "UNKNOWN"
            )
            blocks = turn.get("text_blocks") or []
            texts, starts, stops = [], [], []
            for b in blocks:
                txt = (b.get("text") or "").strip()
                if not txt:
                    continue
                texts.append(txt)
                if b.get("start") is not None:
                    starts.append(float(b["start"]))
                if b.get("stop") is not None:
                    stops.append(float(b["stop"]))
            if not texts:
                continue
            start = min(starts) if starts else turn.get("start")
            stop = max(stops) if stops else turn.get("stop")
            if start is None or stop is None:
                continue
            turns.append({
                "start": float(start),
                "stop": float(stop),
                "speaker": str(name).strip(),
                "text": " ".join(texts),
            })

    turns.sort(key=lambda t: t["start"])
    return {
        "title": transcript.get("title") or audio_obj.get("title"),
        "turns": turns,
        "reference_text": " ".join(t["text"] for t in turns),
        "duration": max((t["stop"] for t in turns), default=0.0),
        "speakers": sorted({t["speaker"] for t in turns}),
    }


def best_audio_url(audio_obj: Dict[str, Any]) -> Optional[str]:
    media = audio_obj.get("media_file") or []
    # Prefer mp3: universally decodable by ffmpeg, and smaller than ogg here.
    for want in ("audio/mpeg", "mp3"):
        for m in media:
            if want in str(m.get("mime", "")).lower() or str(
                m.get("href", "")
            ).endswith(".mp3"):
                return m.get("href")
    return media[0].get("href") if media else None


def download(url: str, dest: pathlib.Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("audio already present: %s", dest.name)
        return True
    log.info("downloading %s", url)
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
                    total += len(chunk)
            tmp.replace(dest)
        log.info("  %.1f MB -> %s", total / 1e6, dest.name)
        return True
    except requests.RequestException as exc:
        log.error("download failed: %s", exc)
        return False


def fetch_case(
    term: str, docket: str, audio_dir: pathlib.Path, gold_dir: pathlib.Path,
    skip_audio: bool = False,
) -> Optional[str]:
    detail = case_detail(term, docket)
    if not detail:
        log.warning("no detail for %s/%s", term, docket)
        return None

    name = detail.get("name", "unknown")
    args = detail.get("oral_argument_audio") or []
    if not args:
        log.info("skip %s (%s): no oral argument audio", docket, name)
        return None

    for idx, entry in enumerate(args):
        href = entry.get("href")
        if not href:
            continue
        audio_obj = _get(href)
        if not audio_obj:
            continue

        parsed = parse_transcript(audio_obj)
        if len(parsed["turns"]) < 10:
            log.info("skip %s: transcript too short (%d turns)",
                     docket, len(parsed["turns"]))
            continue

        slug = f"{term}_{docket.replace('/', '-')}" + (f"_{idx}" if idx else "")
        audio_url = best_audio_url(audio_obj)

        audio_path = audio_dir / f"{slug}.mp3"
        if not skip_audio and audio_url:
            if not download(audio_url, audio_path):
                continue

        gold = {
            "id": slug,
            "case_name": name,
            "term": term,
            "docket": docket,
            "audio_path": str(audio_path),
            "audio_url": audio_url,
            "source": "Oyez (CC-BY-NC)",
            "duration": parsed["duration"],
            "speakers": parsed["speakers"],
            "num_speakers": len(parsed["speakers"]),
            "turns": parsed["turns"],
            "reference_text": parsed["reference_text"],
            # Gold action items / decisions are NOT provided by Oyez. They
            # require expert annotation; see docs/ANNOTATION.md for the
            # protocol and the seed file this script leaves behind.
            "gold_action_items": [],
            "gold_decisions": [],
            "annotation_status": "unannotated",
        }
        write_json(gold_dir / f"{slug}.json", gold)
        log.info(
            "gold: %s | %s | %d turns, %d speakers, %.1f min",
            slug, name, len(parsed["turns"]), len(parsed["speakers"]),
            parsed["duration"] / 60,
        )
        return slug
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Oyez gold corpus")
    ap.add_argument("--term", type=int, default=None)
    ap.add_argument("--case", default=None, help="e.g. 2019/17-1618")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--gold-dir", default="data/gold")
    ap.add_argument("--skip-audio", action="store_true",
                    help="fetch transcripts only (fast, for schema work)")
    ap.add_argument("--max-minutes", type=float, default=90.0,
                    help="skip arguments longer than this")
    args = ap.parse_args()

    audio_dir = ensure_dir(args.audio_dir)
    gold_dir = ensure_dir(args.gold_dir)

    if args.case:
        term, docket = args.case.split("/", 1)
        slug = fetch_case(term, docket, audio_dir, gold_dir, args.skip_audio)
        return 0 if slug else 1

    if args.term is None:
        ap.error("pass --term or --case")

    cases = list_cases(args.term)
    log.info("term %d: %d cases listed", args.term, len(cases))

    fetched = 0
    for case in cases:
        if fetched >= args.limit:
            break
        docket = case.get("docket_number")
        term = str(case.get("term") or args.term)
        if not docket:
            continue
        try:
            slug = fetch_case(term, docket, audio_dir, gold_dir, args.skip_audio)
        except Exception as exc:
            log.warning("case %s failed: %s", docket, exc)
            continue
        if slug:
            fetched += 1
        time.sleep(0.5)  # be polite to a nonprofit's API

    log.info("fetched %d cases into %s", fetched, gold_dir)
    if fetched:
        print(
            f"\nNext:\n"
            f"  python run.py {audio_dir}/<slug>.mp3 --mode all\n"
            f"  python src/eval/report.py --gold {gold_dir} --work work\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
