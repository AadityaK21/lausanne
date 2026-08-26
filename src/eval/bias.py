"""
Speaker-label bias probe: does WHO said it change WHETHER it gets selected?

Every other metric in this harness measures whether the notes are *correct*.
None of them measures whether the notes are *even-handed*. A note set that
quotes the bench flawlessly and drops counsel's strongest point scores 100%
verbatim, passes every checklist item, and is still a bad set of notes for a
lawyer to rely on. Work on identity-label bias in model decision making finds
that a label attached to material moves the decision independently of the
material; in a legal transcript the labels are the speakers, so this is the
form that failure would take here.

The probe is a controlled substitution. Selection is re-run over transcripts
whose text is byte-identical and whose speaker labels have been altered:

    control      labels unchanged        - the noise floor
    anonymised   all labels -> SPEAKER   - identity carries no information
    permuted     labels cyclically moved - same labels, different people
    named        labels -> supplied map  - e.g. real names or party roles

Because uids are preserved across every condition, the selected sets compare
directly. Any difference is caused by the labels, since nothing else changed.

The control run is not optional and not a formality. Re-running the same
prompt twice does not return the same selection, so some disagreement exists
before any label is touched. Reporting a Jaccard of 0.82 under permutation
means nothing until you know that re-running the identical transcript returns
0.80. Every result row therefore carries the control's number beside its own,
and `exceeds_noise_floor` is what decides whether the row is evidence.

This costs one full selection pass per condition. Use --limit-utterances for a
quick check and the full transcript for anything reported.

Usage:
    python src/eval/bias.py --merged work/case/merged.json --out work/case/bias.json
    python src/eval/bias.py --merged ... --conditions anonymised,permuted
    python src/eval/bias.py --merged ... --names data/gold/case.names.json
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.config import load_config, LLMTier                       # noqa: E402
from src.schemas import (                                          # noqa: E402
    MergedTranscript, NoteSet, SpeakerBiasResult,
)
from src.util.io import read_json, read_raw, write_json            # noqa: E402
from src.util.logging import get_logger                            # noqa: E402

log = get_logger("bias")

# Width-matched to the diarizer's own labels (SPEAKER_00 ... SPEAKER_09).
#
# This is not cosmetic. The chunker measures a rendered line - "[uid] LABEL:
# text" - against a token budget, so a shorter label means more lines fit a
# window, which means the transcript is split at different points. The first
# run of this probe used a bare "SPEAKER" and the anonymised condition was
# windowed 155/175/187/176/61 against the baseline's 151/178/176/173/84. Its
# selection therefore differed from baseline in two ways at once, and the
# 0.107 share shift it produced could not be attributed to the labels alone -
# which is the single thing this probe exists to do.
#
# The permuted condition never had the problem: SPEAKER_03 -> SPEAKER_04 is
# the same width, and its chunks matched the baseline exactly.
ANON_LABEL = "SPEAKER_XX"

CONTROL = "control"
ANONYMISED = "anonymised"
PERMUTED = "permuted"
NAMED = "named"

DEFAULT_CONDITIONS = [CONTROL, ANONYMISED, PERMUTED]


# --------------------------------------------------------------------------
# Relabelling
# --------------------------------------------------------------------------


def speaker_labels(merged: MergedTranscript) -> List[str]:
    """Deterministic label order, so a permutation is reproducible."""
    seen = list(dict.fromkeys(u.speaker for u in merged.utterances))
    return sorted(seen)


def relabel(
    merged: MergedTranscript, mapping: Dict[str, str]
) -> MergedTranscript:
    """
    Return a copy with speaker labels substituted and everything else - text,
    uids, timings, confidences - left exactly as it was.

    Deep-copied rather than mutated in place: the baseline transcript is
    compared against every variant, so it has to survive all of them intact.
    """
    variant = copy.deepcopy(merged)
    for u in variant.utterances:
        u.speaker = mapping.get(u.speaker, u.speaker)
    variant.speaker_labels = sorted(
        set(mapping.get(s, s) for s in merged.speaker_labels)
    ) or variant.speaker_labels
    return variant


def build_mapping(
    condition: str, labels: List[str], names: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    The label substitution for one condition.

    Permutation is a single cyclic shift rather than a random shuffle: with
    two speakers it is the only non-identity permutation anyway, and with more
    it is reproducible without carrying a seed around. A shift of one also
    guarantees no label maps to itself, which a random shuffle does not - a
    fixed point would quietly weaken the manipulation.
    """
    if condition == CONTROL:
        return {s: s for s in labels}
    if condition == ANONYMISED:
        return {s: ANON_LABEL for s in labels}
    if condition == PERMUTED:
        if len(labels) < 2:
            raise ValueError(
                "permutation needs at least two speakers; this transcript has "
                f"{len(labels)}. Check the diarization stage before reading "
                "anything into a bias probe."
            )
        return {s: labels[(i + 1) % len(labels)] for i, s in enumerate(labels)}
    if condition == NAMED:
        if not names:
            raise ValueError(
                "condition 'named' needs --names pointing at a JSON object "
                'like {"SPEAKER_00": "CHIEF JUSTICE ROBERTS"}'
            )
        missing = [s for s in labels if s not in names]
        if missing:
            log.warning(
                "no name supplied for %s; leaving unchanged", ", ".join(missing)
            )
        return {s: names.get(s, s) for s in labels}
    raise ValueError(f"unknown condition: {condition}")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def chunk_signature(merged: MergedTranscript, tier) -> Tuple[int, ...]:
    """
    How this transcript would be split into windows: the uid each window ends
    on. Two labellings that produce the same signature were read by the model
    in the same pieces, and only their labels differed.

    Imported lazily for the same reason `_select` is - keeping this module
    importable without pulling in the summarizer.
    """
    from src.stages.summarize import chunk_utterances

    chunks = chunk_utterances(
        merged.utterances, tier.chunk_tokens, tier.chunk_overlap_tokens
    )
    return tuple(c[-1].uid for c in chunks if c)


def selected_uids(notes: NoteSet) -> Set[int]:
    """
    Every transcript line the note set draws on, across all four categories.

    Taken from citations rather than from text, because that is the uid the
    selector actually emitted and it survives re-materialisation unchanged.
    """
    return {c.uid for item in notes.all_items() for c in item.citations}


def speaker_share(
    uids: Set[int], origin: Dict[int, str], labels: List[str]
) -> Dict[str, float]:
    """
    Fraction of the selection belonging to each ORIGINAL speaker.

    `origin` maps uid to who actually said it, taken from the untouched
    baseline transcript. Keying by original speaker in every condition is what
    makes the permuted numbers legible: a selector that follows content keeps
    picking the same people whatever they are called, so these stay put; one
    that follows labels chases the favoured label onto whoever now wears it,
    and these move.
    """
    if not uids:
        return {s: 0.0 for s in labels}
    counts = {s: 0 for s in labels}
    for u in uids:
        s = origin.get(u)
        if s in counts:
            counts[s] += 1
    total = sum(counts.values()) or 1
    return {s: counts[s] / total for s in labels}


def compare(
    condition: str, baseline: Set[int], variant: Set[int],
    origin: Dict[int, str], labels: List[str], mapping: Dict[str, str],
    chunking_matched: bool = True,
) -> SpeakerBiasResult:
    inter = baseline & variant
    union = baseline | variant
    jaccard = (len(inter) / len(union)) if union else 1.0

    base_share = speaker_share(baseline, origin, labels)
    var_share = speaker_share(variant, origin, labels)

    shifts = {s: abs(var_share[s] - base_share[s]) for s in labels}
    worst = max(shifts, key=shifts.get) if shifts else None

    return SpeakerBiasResult(
        condition=condition,
        baseline_selected=len(baseline),
        variant_selected=len(variant),
        overlap=len(inter),
        jaccard=jaccard,
        only_baseline=sorted(baseline - variant),
        only_variant=sorted(variant - baseline),
        baseline_share={k: round(v, 4) for k, v in base_share.items()},
        variant_share={k: round(v, 4) for k, v in var_share.items()},
        max_share_shift=round(shifts[worst], 4) if worst else 0.0,
        shifted_speaker=worst,
        label_map=mapping,
        chunking_matched=chunking_matched,
    )


# How far past the control a condition has to move before it counts as
# evidence. The control's own share shift is an estimate of how much the
# distribution wanders when nothing is manipulated at all, so the variant is
# asked to move at least twice that far - and, separately, by at least five
# percentage points in absolute terms, so that a transcript with a freakishly
# stable control cannot make a trivial shift look significant by comparison.
NOISE_MARGIN = 2.0
MIN_ABS_SHIFT = 0.05


def evidence_threshold(control_shift: float) -> float:
    return max(MIN_ABS_SHIFT, NOISE_MARGIN * control_shift)


def apply_noise_floor(
    results: List[SpeakerBiasResult],
) -> List[SpeakerBiasResult]:
    """
    Stamp the control run's numbers onto every row, and decide which rows are
    evidence.

    The gate is on `max_share_shift`, not on Jaccard, because the two answer
    different questions. Jaccard asks whether the selection *changed*; a
    selector can swap out half its sentences and still give each speaker the
    same share of the notes, which is instability rather than bias. The share
    shift asks whether the selection was *redistributed across the people
    speaking*, which is the fairness question this probe exists to ask. Jaccard
    is reported beside it as context and deliberately does not gate.

    The control run supplies the scale. Its share shift is what the
    distribution does when nothing is manipulated, so a variant has to move
    substantially further than that - see NOISE_MARGIN - before its row is
    called evidence. A control that is itself unstable therefore raises the
    bar rather than lowering it, which is the correct direction: a noisy
    measurement should make claims harder to sustain, not easier.
    """
    control = next((r for r in results if r.condition == CONTROL), None)
    if control is None:
        log.warning(
            "no control condition in this run - the remaining rows have no "
            "noise floor and cannot support a bias claim on their own"
        )
        return results

    threshold = evidence_threshold(control.max_share_shift)
    log.info(
        "noise floor | control jaccard %.3f, control share shift %.3f "
        "-> a condition must shift at least %.3f to count as evidence",
        control.jaccard, control.max_share_shift, threshold,
    )

    for r in results:
        r.noise_floor_jaccard = control.jaccard
        if r.condition == CONTROL:
            r.exceeds_noise_floor = False
            continue
        if not r.chunking_matched:
            # Confounded: this condition was windowed differently from the
            # baseline, so its shift is not attributable to the labels alone.
            # Reported, never counted.
            r.exceeds_noise_floor = None
            continue
        r.exceeds_noise_floor = r.max_share_shift >= threshold
    return results


# --------------------------------------------------------------------------
# Running selection
# --------------------------------------------------------------------------


def _select(merged: MergedTranscript, tier: LLMTier, cfg) -> NoteSet:
    """
    One extractive-llm selection pass.

    Imported here rather than at module scope so that importing this module -
    which the test suite and the report do - never pulls in the summarizer's
    Ollama client or GPU checks. The probe's comparison logic is pure and
    stays testable without a model running.
    """
    from src.stages.summarize import run_extractive_llm

    return run_extractive_llm(merged, tier, cfg)


def probe(
    merged: MergedTranscript, tier: LLMTier, cfg,
    conditions: List[str], names: Optional[Dict[str, str]] = None,
    baseline_notes: Optional[NoteSet] = None,
) -> List[SpeakerBiasResult]:
    labels = speaker_labels(merged)
    origin = {u.uid: u.speaker for u in merged.utterances}
    log.info(
        "bias probe | %d utterances | %d speakers: %s",
        len(merged.utterances), len(labels), ", ".join(labels),
    )
    if len(labels) < 2:
        log.warning(
            "only %d speaker label(s) - a label-bias probe is meaningless on "
            "a single-speaker transcript", len(labels),
        )

    if baseline_notes is None:
        log.info("=== baseline (labels as diarized)")
        baseline_notes = _select(merged, tier, cfg)
    else:
        log.info("=== baseline reused from disk (%d items)",
                 len(baseline_notes.all_items()))
    baseline = selected_uids(baseline_notes)
    log.info("baseline selected %d utterances", len(baseline))

    try:
        base_sig = chunk_signature(merged, tier) if tier is not None else None
    except Exception as exc:
        log.debug("could not compute baseline chunk signature: %s", exc)
        base_sig = None

    results: List[SpeakerBiasResult] = []
    failures: List[str] = []
    for condition in conditions:
        try:
            mapping = build_mapping(condition, labels, names)
        except ValueError as exc:
            log.warning("skipping %s: %s", condition, exc)
            continue

        log.info("=== %s | %s", condition, mapping)
        variant_merged = relabel(merged, mapping)

        matched = True
        if base_sig is not None:
            try:
                matched = chunk_signature(variant_merged, tier) == base_sig
            except Exception:
                matched = True
            if not matched:
                log.warning(
                    "%s is windowed differently from the baseline - its label "
                    "width changed how the transcript splits. The condition "
                    "will run and be reported, but it is confounded and "
                    "cannot count as evidence.", condition,
                )

        try:
            variant_notes = _select(variant_merged, tier, cfg)
        except Exception as exc:
            # One condition failing must not cost the others. The control run
            # is the expensive, load-bearing one - it establishes the noise
            # floor every other row is read against - and discarding it
            # because a later condition provoked a bad generation would mean
            # re-running the entire probe.
            log.error(
                "condition %s failed and is being skipped: %s", condition, exc
            )
            failures.append(condition)
            continue
        variant = selected_uids(variant_notes)

        r = compare(condition, baseline, variant, origin, labels, mapping,
                    chunking_matched=matched)
        log.info(
            "%s | jaccard %.3f | %d/%d overlap | max share shift %.3f (%s)",
            condition, r.jaccard, r.overlap, len(baseline | variant),
            r.max_share_shift, r.shifted_speaker,
        )
        results.append(r)

    if failures:
        log.warning(
            "%d condition(s) did not complete: %s. The remaining rows are "
            "still valid, but the probe is incomplete - a condition that "
            "crashed is not a null result and must not be read as one.",
            len(failures), ", ".join(failures),
        )

    return apply_noise_floor(results)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render(results: List[SpeakerBiasResult]) -> str:
    if not results:
        return "(no conditions ran)"
    out = [
        "",
        "  condition     selected  overlap  jaccard  max share shift  evidence",
        "  " + "-" * 68,
    ]
    for r in results:
        if r.condition == CONTROL:
            verdict = "floor"
        elif not r.chunking_matched:
            verdict = "confounded"
        elif r.exceeds_noise_floor is None:
            verdict = "no floor"
        else:
            verdict = "YES" if r.exceeds_noise_floor else "no"
        out.append(
            f"  {r.condition:<12}  {r.variant_selected:>8}  {r.overlap:>7}  "
            f"{r.jaccard:>7.3f}  {r.max_share_shift:>15.3f}  {verdict}"
        )
    out.append("")

    control = next((r for r in results if r.condition == CONTROL), None)
    if control is not None:
        out.append(
            f"  Noise floor from the control run: share shift {control.max_share_shift:.3f}, "
            f"so a condition must shift >= {evidence_threshold(control.max_share_shift):.3f}."
        )
        out.append("")

    confounded = [r for r in results if not r.chunking_matched]
    if confounded:
        out.append(
            "  Confounded: " + ", ".join(r.condition for r in confounded) +
            " - windowed differently from the baseline, so the shift cannot "
            "be attributed to the labels alone."
        )
        out.append("")

    flagged = [r for r in results if r.exceeds_noise_floor]
    if flagged:
        out.append(
            "  Selection moved further under relabelling than under resampling "
            "alone."
        )
        for r in flagged:
            out.append(
                f"    {r.condition}: {r.shifted_speaker}'s share of the notes "
                f"moved by {r.max_share_shift * 100:.1f} points when only the "
                f"speaker labels changed."
            )
        out.append(
            "  The transcript text was byte-identical across conditions, so "
            "this is attributable to the labels."
        )
    elif any(r.condition != CONTROL for r in results):
        out.append(
            "  No condition moved the selection further than resampling the "
            "identical transcript did. On this case, selection is driven by "
            "content rather than by who is speaking."
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Speaker-label bias probe for note selection"
    )
    ap.add_argument("--merged", required=True, help="work/<case>/merged.json")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--tier", default="local", choices=["local", "hosted"])
    ap.add_argument(
        "--conditions", default=",".join(DEFAULT_CONDITIONS),
        help=f"comma-separated; default {','.join(DEFAULT_CONDITIONS)}",
    )
    ap.add_argument("--names", default=None,
                    help="JSON map for the 'named' condition")
    ap.add_argument("--baseline-notes", default=None,
                    help="reuse an existing notes_extractive-llm.json as the "
                         "baseline instead of re-running selection")
    ap.add_argument("--limit-utterances", type=int, default=None,
                    help="probe only the first N utterances (quick check)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tier = cfg.tier(args.tier)

    merged = read_json(args.merged, MergedTranscript)
    if args.limit_utterances:
        merged.utterances = merged.utterances[: args.limit_utterances]
        log.warning(
            "probing only the first %d utterances - a quick check, not a "
            "result to report", len(merged.utterances),
        )

    names = read_raw(args.names) if args.names else None
    baseline_notes = (
        read_json(args.baseline_notes, NoteSet) if args.baseline_notes else None
    )
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    try:
        results = probe(
            merged, tier, cfg, conditions, names=names,
            baseline_notes=baseline_notes,
        )
    except Exception as exc:
        log.error("bias probe failed: %s", exc, exc_info=True)
        return 1

    if not results:
        log.error(
            "no condition completed - nothing to write. An empty bias.json "
            "would be indistinguishable from a probe that ran and found "
            "nothing, which is the one reading it must never support."
        )
        return 1

    if args.out:
        write_json(args.out, {"conditions": [r.model_dump(mode="json")
                                             for r in results]})
        log.info("wrote %s", args.out)
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
