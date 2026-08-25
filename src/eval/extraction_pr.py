"""
Precision / recall of action-item and decision detection.

Section 8.4 asks for "the precision/recall of the rule-based action-item and
decision extraction" against expert-annotated gold. This module computes it,
for all three note modes, so the rule-based cue patterns can be compared
against model-based selection on the same footing.

Matching is temporal, not textual, and that is a deliberate choice. Gold spans
are annotated against the *official* transcript's timeline; predictions live
on the ASR's own segmentation, which will not agree on sentence boundaries.
Requiring string equality would score segmentation disagreement as extraction
error. A predicted item counts as a hit when it overlaps a gold span by at
least `--iou` of that span's duration.

Because Oyez ships no such annotations, `make_annotation_template` writes a
skeleton for a human to fill in. Numbers reported without that step are
meaningless, and the harness says so rather than emitting a zero.

Usage:
    python src/eval/extraction_pr.py --gold data/gold/X.json \\
        --notes work/X/notes_extractive-rule.json
    python src/eval/extraction_pr.py --gold data/gold/X.json --make-template
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.schemas import ExtractionPRResult, NoteItem, NoteSet     # noqa: E402
from src.util.io import read_json, read_raw, write_json           # noqa: E402
from src.util.logging import get_logger                           # noqa: E402

log = get_logger("extract-pr")

Span = Tuple[float, float]


def _span_of(item: NoteItem) -> Span | None:
    if not item.citations:
        return None
    return (
        min(c.start for c in item.citations),
        max(c.end for c in item.citations),
    )


def _overlap_fraction(pred: Span, gold: Span) -> float:
    """
    Overlap normalised by the SHORTER of the two spans.

    Normalising by the gold span alone was the obvious first implementation
    and it is wrong here, because the two sides are annotated at different
    granularities. Gold spans are whole speaking turns - a justice's question
    can run ninety seconds - while a prediction is a single sentence inside
    such a turn. Dividing by the gold duration means a correctly identified
    ruling scores 0.15 against its own turn and is counted as a miss, which
    measures the segmentation mismatch rather than the extraction.

    Using min(pred, gold) asks the question we actually mean: do these two
    spans point at the same moment in the proceeding?
    """
    lo = max(pred[0], gold[0])
    hi = min(pred[1], gold[1])
    if hi <= lo:
        return 0.0
    shorter = min(pred[1] - pred[0], gold[1] - gold[0])
    return (hi - lo) / shorter if shorter > 0 else 0.0


def score_category(
    predictions: Sequence[NoteItem],
    gold_spans: Sequence[Dict],
    category: str,
    threshold: float = 0.5,
) -> ExtractionPRResult:
    gold: List[Span] = [
        (float(g["start"]), float(g["stop"])) for g in gold_spans
    ]
    preds: List[Span] = []
    for item in predictions:
        s = _span_of(item)
        if s is not None:
            preds.append(s)

    matched_gold = set()
    tp = 0
    for p in preds:
        best_i, best_frac = None, 0.0
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            frac = _overlap_fraction(p, g)
            if frac > best_frac:
                best_frac, best_i = frac, i
        # One-to-one matching: a single prediction cannot claim two gold items,
        # and two predictions over the same gold item cost a false positive.
        if best_i is not None and best_frac >= threshold:
            matched_gold.add(best_i)
            tp += 1

    fp = len(preds) - tp
    fn = len(gold) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    log.info(
        "%s | P %.3f R %.3f F1 %.3f  (TP %d FP %d FN %d)",
        category, precision, recall, f1, tp, fp, fn,
    )
    return ExtractionPRResult(
        category=category,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def make_annotation_template(gold: Dict, out_path: pathlib.Path) -> None:
    """
    Emit a skeleton for expert annotation.

    Every gold turn is listed with its timing so an annotator can mark the
    ones that carry an obligation or a ruling. This is the step that makes the
    P/R numbers real, and it is the one that cannot be automated - which is
    itself worth stating in the paper.
    """
    lines = [
        "# Annotation template",
        f"# case: {gold.get('case_name')}  ({gold.get('id')})",
        "#",
        "# For each turn below, prefix with:",
        "#   A  - imposes an obligation, deadline, filing, production, follow-up",
        "#   D  - records a ruling, grant, denial, agreement, stipulation",
        "#   AD - both",
        "#   (leave blank otherwise)",
        "#",
        "# Then run:  python src/eval/extraction_pr.py --gold <gold.json> "
        "--apply-annotations <this file>",
        "#",
    ]
    for i, t in enumerate(gold.get("turns", [])):
        text = t["text"].replace("\n", " ")
        if len(text) > 240:
            text = text[:237] + "..."
        lines.append(
            f"[  ] {i:>4} {t['start']:>8.2f}-{t['stop']:>8.2f} "
            f"{t['speaker']}: {text}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote annotation template: %s (%d turns)",
             out_path, len(gold.get("turns", [])))


def apply_annotations(gold: Dict, template_path: pathlib.Path) -> Dict:
    """Read a filled-in template back into the gold file."""
    actions, decisions = [], []
    turns = gold.get("turns", [])
    for line in template_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("["):
            continue
        tag = line[1:3].strip().upper()
        if not tag:
            continue
        try:
            idx = int(line[4:].split()[0])
        except (ValueError, IndexError):
            continue
        if idx >= len(turns):
            continue
        turn = turns[idx]
        entry = {
            "start": turn["start"], "stop": turn["stop"],
            "speaker": turn["speaker"], "text": turn["text"],
            "turn_index": idx,
        }
        if "A" in tag:
            actions.append(entry)
        if "D" in tag:
            decisions.append(entry)

    gold["gold_action_items"] = actions
    gold["gold_decisions"] = decisions
    gold["annotation_status"] = "annotated"
    log.info("applied: %d action items, %d decisions",
             len(actions), len(decisions))
    return gold


def main() -> int:
    ap = argparse.ArgumentParser(description="Extraction precision/recall")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--apply-annotations", default=None)
    args = ap.parse_args()

    gold_path = pathlib.Path(args.gold)
    gold = read_raw(gold_path)

    if args.make_template:
        make_annotation_template(
            gold, gold_path.with_suffix(".annotate.txt")
        )
        return 0

    if args.apply_annotations:
        gold = apply_annotations(gold, pathlib.Path(args.apply_annotations))
        write_json(gold_path, gold)
        log.info("updated %s", gold_path)
        return 0

    if not args.notes:
        ap.error("pass --notes, or --make-template / --apply-annotations")

    if gold.get("annotation_status") != "annotated":
        log.error(
            "gold file %s is not annotated. Action items and decisions cannot "
            "be scored without expert labels.\n"
            "  1. python src/eval/extraction_pr.py --gold %s --make-template\n"
            "  2. fill in the .annotate.txt file\n"
            "  3. python src/eval/extraction_pr.py --gold %s "
            "--apply-annotations <file>",
            gold_path, gold_path, gold_path,
        )
        return 2

    notes = read_json(args.notes, NoteSet)

    results = [
        score_category(
            notes.action_items, gold.get("gold_action_items", []),
            "action_items", args.iou,
        ),
        score_category(
            notes.decisions, gold.get("gold_decisions", []),
            "decisions", args.iou,
        ),
    ]

    payload = {
        "mode": notes.mode.value,
        "model": notes.model,
        "iou_threshold": args.iou,
        "results": [r.model_dump(mode="json") for r in results],
    }

    if args.out:
        write_json(args.out, payload)
        log.info("wrote %s", args.out)
    else:
        print()
        print(f"  mode: {notes.mode.value}")
        print(f"  {'category':<14} {'P':>7} {'R':>7} {'F1':>7}   TP/FP/FN")
        for r in results:
            print(f"  {r.category:<14} {r.precision:>7.3f} {r.recall:>7.3f} "
                  f"{r.f1:>7.3f}   {r.true_positives}/{r.false_positives}/"
                  f"{r.false_negatives}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
