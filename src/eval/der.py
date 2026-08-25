"""
Diarization error rate against Oyez gold speaker turns.

DER = (missed speech + false alarm + speaker confusion) / total reference speech

Implemented frame-wise on a 10 ms grid. That is coarser than a proper
interval-based scorer but agrees with it to well within the differences we
care about, and it is short enough to audit by eye - which matters more here
than the last decimal place.

Two conventions worth stating, because DER numbers are meaningless without
them:

  * A forgiveness collar of 250 ms is applied around every reference boundary.
    This is the standard NIST convention and exists because human annotators
    do not agree on boundary placement to better than that.

  * Overlapping speech IS scored. Many published DERs quietly exclude it,
    which flatters any system on courtroom audio specifically, because
    interruption from the bench is constant. Excluding it here would hide the
    hardest and most legally relevant part of the problem. `der_no_overlap` is
    also reported so numbers can be compared with papers that exclude it.

Usage:
    python src/eval/der.py --gold data/gold/X.json --hyp work/X/speakers.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from itertools import permutations
from typing import Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.schemas import DERResult, DiarizationResult, SpeakerTurn  # noqa: E402
from src.util.io import read_json, read_raw, write_json            # noqa: E402
from src.util.logging import get_logger                            # noqa: E402

log = get_logger("der")

FRAME = 0.010          # 10 ms
COLLAR = 0.250         # 250 ms either side of a reference boundary


def _to_frames(
    turns: Sequence[Tuple[float, float, str]], n_frames: int
) -> List[Set[str]]:
    grid: List[Set[str]] = [set() for _ in range(n_frames)]
    for start, stop, label in turns:
        a = max(0, int(start / FRAME))
        b = min(n_frames, int(stop / FRAME) + 1)
        for f in range(a, b):
            grid[f].add(label)
    return grid


def _collar_mask(
    turns: Sequence[Tuple[float, float, str]], n_frames: int
) -> List[bool]:
    """True where the frame is scored (i.e. NOT inside a collar)."""
    mask = [True] * n_frames
    half = int(COLLAR / FRAME)
    for start, stop, _ in turns:
        for boundary in (start, stop):
            c = int(boundary / FRAME)
            for f in range(max(0, c - half), min(n_frames, c + half + 1)):
                mask[f] = False
    return mask


def _best_mapping(
    ref_grid: List[Set[str]], hyp_grid: List[Set[str]],
    ref_labels: List[str], hyp_labels: List[str],
) -> Dict[str, str]:
    """
    Map hypothesis labels to reference labels to maximise frame agreement.

    Exhaustive when the label counts are small (which they are for a court
    proceeding - typically 3-8 voices), greedy otherwise.
    """
    overlap: Dict[Tuple[str, str], int] = {}
    for r_set, h_set in zip(ref_grid, hyp_grid):
        for r in r_set:
            for h in h_set:
                overlap[(r, h)] = overlap.get((r, h), 0) + 1

    if not overlap:
        return {}

    if len(hyp_labels) <= 8 and len(ref_labels) <= 8:
        best_map, best_score = {}, -1
        longer, shorter = (
            (ref_labels, hyp_labels) if len(ref_labels) >= len(hyp_labels)
            else (hyp_labels, ref_labels)
        )
        ref_is_longer = len(ref_labels) >= len(hyp_labels)
        for perm in permutations(longer, len(shorter)):
            if ref_is_longer:
                candidate = {h: r for h, r in zip(shorter, perm)}
            else:
                candidate = {h: r for h, r in zip(perm, shorter)}
            s = sum(overlap.get((r, h), 0) for h, r in candidate.items())
            if s > best_score:
                best_score, best_map = s, candidate
        return best_map

    # Greedy fallback.
    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    for (r, h), _c in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if h in mapping or r in used:
            continue
        mapping[h] = r
        used.add(r)
    return mapping


def compute(
    ref_turns: Sequence[Tuple[float, float, str]],
    hyp_turns: Sequence[Tuple[float, float, str]],
    score_overlap: bool = True,
) -> Tuple[DERResult, float]:
    duration = max(
        max((t[1] for t in ref_turns), default=0.0),
        max((t[1] for t in hyp_turns), default=0.0),
    )
    n_frames = int(duration / FRAME) + 1

    ref_grid = _to_frames(ref_turns, n_frames)
    hyp_grid = _to_frames(hyp_turns, n_frames)
    scored = _collar_mask(ref_turns, n_frames)

    ref_labels = sorted({t[2] for t in ref_turns})
    hyp_labels = sorted({t[2] for t in hyp_turns})
    mapping = _best_mapping(ref_grid, hyp_grid, ref_labels, hyp_labels)
    log.info("speaker mapping: %s", mapping or "(none found)")

    total = missed = false_alarm = confusion = 0
    total_no_ovl = missed_no = fa_no = conf_no = 0

    for f in range(n_frames):
        if not scored[f]:
            continue
        ref_set = ref_grid[f]
        hyp_set = {mapping.get(h, f"__unmapped_{h}") for h in hyp_grid[f]}

        n_ref, n_hyp = len(ref_set), len(hyp_set)
        correct = len(ref_set & hyp_set)

        frame_missed = max(0, n_ref - max(n_hyp, correct))
        frame_fa = max(0, n_hyp - n_ref)
        frame_conf = min(n_ref, n_hyp) - correct

        total += n_ref
        missed += frame_missed
        false_alarm += frame_fa
        confusion += frame_conf

        if n_ref <= 1:
            total_no_ovl += n_ref
            missed_no += frame_missed
            fa_no += frame_fa
            conf_no += frame_conf

    def rate(x: int, denom: int) -> float:
        return (x / denom) if denom else 0.0

    der = rate(missed + false_alarm + confusion, total)
    der_no_overlap = rate(missed_no + fa_no + conf_no, total_no_ovl)

    result = DERResult(
        total_speech_seconds=total * FRAME,
        missed_detection=rate(missed, total),
        false_alarm=rate(false_alarm, total),
        speaker_confusion=rate(confusion, total),
        der=der,
        reference_speakers=len(ref_labels),
        hypothesis_speakers=len(hyp_labels),
    )
    log.info(
        "DER %.2f%%  (miss %.2f%% / FA %.2f%% / conf %.2f%%)  |  "
        "no-overlap DER %.2f%%  |  speakers ref=%d hyp=%d",
        der * 100, result.missed_detection * 100, result.false_alarm * 100,
        result.speaker_confusion * 100, der_no_overlap * 100,
        len(ref_labels), len(hyp_labels),
    )
    return result, der_no_overlap


def main() -> int:
    ap = argparse.ArgumentParser(description="Diarization error rate")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = read_raw(args.gold)
    ref_turns = [
        (float(t["start"]), float(t["stop"]), str(t["speaker"]))
        for t in gold["turns"]
    ]

    diar = read_json(args.hyp, DiarizationResult)
    turns: List[SpeakerTurn] = diar.exclusive_turns or diar.turns
    hyp_turns = [(t.start, t.end, t.speaker) for t in turns]

    result, der_no_overlap = compute(ref_turns, hyp_turns)

    if args.out:
        payload = result.model_dump(mode="json")
        payload["der_no_overlap"] = der_no_overlap
        payload["used_exclusive"] = bool(diar.exclusive_turns)
        write_json(args.out, payload)
        log.info("wrote %s", args.out)
    else:
        print()
        print(f"  DER                {result.der * 100:6.2f}%")
        print(f"  DER (no overlap)   {der_no_overlap * 100:6.2f}%")
        print(f"  missed speech      {result.missed_detection * 100:6.2f}%")
        print(f"  false alarm        {result.false_alarm * 100:6.2f}%")
        print(f"  speaker confusion  {result.speaker_confusion * 100:6.2f}%")
        print(f"  speakers           ref={result.reference_speakers} "
              f"hyp={result.hypothesis_speakers}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
