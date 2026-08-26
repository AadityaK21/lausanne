"""
Aggregate every metric into the tables the paper needs.

Produces results.json (machine-readable) and results.md (paste-ready), with
one row per (case, mode) and a summary block averaging across cases.

The table this is built to produce is the one that settles the survey's
central tension for this system:

    mode              informativeness   verbatim   judged support   cost
    extractive-rule   low               100%       n/a              0 s
    extractive-llm    high              100%       n/a              ~3 min
    abstractive       high              ~0%        measured         ~3 min

If extractive-llm holds coverage close to abstractive while keeping the
verbatim guarantee, the argument for using free generation on privileged
material gets considerably weaker - and that is a claim with a number behind
it rather than an assertion.

Usage:
    python src/eval/report.py --gold data/gold --work work
    python src/eval/report.py --gold data/gold --work work --judge
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.config import load_config                                  # noqa: E402
from src.eval import der as der_mod                                 # noqa: E402
from src.eval import extraction_pr as pr_mod                        # noqa: E402
from src.eval import judge as judge_mod                             # noqa: E402
from src.eval import wer as wer_mod                                 # noqa: E402
from src.schemas import (                                            # noqa: E402
    ASRResult, DiarizationResult, MergedTranscript, NoteSet,
)
from src.util.io import ensure_dir, read_json, read_raw, write_json  # noqa: E402
from src.util.logging import get_logger                              # noqa: E402

log = get_logger("report")

MODES = ["extractive-rule", "extractive-llm", "abstractive"]


def _safe(fn, label: str, default=None):
    try:
        return fn()
    except Exception as exc:
        log.warning("%s failed: %s", label, exc)
        return default


def coverage(notes: NoteSet, merged: MergedTranscript) -> Dict[str, float]:
    """
    Cheap informativeness proxies, computed without a model.

    `content_coverage` is the fraction of the transcript's distinct content
    words that appear somewhere in the notes; `temporal_spread` is how much of
    the proceeding's timeline the cited material spans. A summariser that
    picks eight good sentences from the first five minutes scores badly on the
    second, which is the failure mode frequency-based extraction actually has.
    """
    import re

    def words(text: str) -> set:
        return {
            w for w in re.findall(r"[a-z][a-z'\-]{3,}", text.lower())
        }

    source = words(" ".join(u.text for u in merged.utterances))
    note_text = " ".join(i.text for i in notes.all_items())
    covered = words(note_text)

    spans = [
        c.start for i in notes.all_items() for c in i.citations
    ]
    spread = 0.0
    if spans and merged.audio_duration > 0:
        spread = (max(spans) - min(spans)) / merged.audio_duration

    return {
        "content_coverage": len(source & covered) / len(source) if source else 0.0,
        "temporal_spread": spread,
        "note_words": len(note_text.split()),
        "compression_ratio": (
            len(note_text.split())
            / max(1, sum(len(u.text.split()) for u in merged.utterances))
        ),
    }


def evaluate_case(
    gold_path: pathlib.Path, work_root: pathlib.Path, cfg,
    do_judge: bool, injection_test: bool, judge_limit: Optional[int],
) -> Optional[Dict[str, Any]]:
    gold = read_raw(gold_path)
    slug = gold["id"]
    work = work_root / slug
    if not work.exists():
        log.warning("no work dir for %s (run: python run.py %s --mode all)",
                    slug, gold.get("audio_path"))
        return None

    row: Dict[str, Any] = {
        "id": slug,
        "case_name": gold.get("case_name"),
        "duration_min": gold.get("duration", 0.0) / 60,
        "reference_speakers": gold.get("num_speakers"),
        "annotated": gold.get("annotation_status") == "annotated",
    }

    # ---- ASR
    tpath = work / "transcript.json"
    if tpath.exists():
        asr = read_json(tpath, ASRResult)
        w = _safe(
            lambda: wer_mod.score(gold["reference_text"], asr.full_text()),
            f"WER {slug}",
        )
        if w:
            row["wer"] = w.wer
            row["legal_term_wer"] = w.legal_term_wer
            row["legal_term_count"] = w.legal_term_count
            row["substitution_rate"] = w.substitution_rate
            row["legal_substitution_rate"] = w.legal_substitution_rate
            # Keep the legal-term errors specifically. `top_errors` is the
            # global top-k with the worst legal pairs appended after it, so
            # slicing the front of the list kept the frequent-but-dull
            # function-word deletions and discarded every entry the
            # "Legal-term error examples" section exists to show - which is
            # why that section rendered empty on a run that had plenty.
            legal_errs = [e for e in w.top_errors if e.get("legal")]
            row["top_errors"] = (legal_errs or w.top_errors)[:12]
        row["asr_model"] = asr.model
        row["asr_rtf"] = asr.rtf
        row["asr_seconds"] = asr.elapsed_seconds

    # ---- Diarization
    spath = work / "speakers.json"
    if spath.exists():
        diar = read_json(spath, DiarizationResult)
        turns = diar.exclusive_turns or diar.turns
        d = _safe(
            lambda: der_mod.compute(
                [(float(t["start"]), float(t["stop"]), str(t["speaker"]))
                 for t in gold["turns"]],
                [(t.start, t.end, t.speaker) for t in turns],
            ),
            f"DER {slug}",
        )
        if d:
            der_res, der_no_ovl = d
            row["der"] = der_res.der
            row["der_no_overlap"] = der_no_ovl
            row["missed_speech"] = der_res.missed_detection
            row["false_alarm"] = der_res.false_alarm
            row["speaker_confusion"] = der_res.speaker_confusion
            row["hyp_speakers"] = der_res.hypothesis_speakers
        row["diar_model"] = diar.model
        row["diar_seconds"] = diar.elapsed_seconds

    # ---- Note modes
    mpath = work / "merged.json"
    if not mpath.exists():
        return row
    merged = read_json(mpath, MergedTranscript)
    row["utterances"] = len(merged.utterances)

    row["modes"] = {}
    for mode in MODES:
        npath = work / f"notes_{mode}.json"
        if not npath.exists():
            continue
        notes = read_json(npath, NoteSet)
        entry: Dict[str, Any] = {
            "model": notes.model,
            "tier": notes.tier,
            "seconds": notes.elapsed_seconds,
            "llm_calls": notes.llm_calls,
            "validation_warnings": len(notes.validation_warnings),
            "n_summary": len(notes.summary),
            "n_key_points": len(notes.key_points),
            "n_decisions": len(notes.decisions),
            "n_action_items": len(notes.action_items),
        }
        entry.update(coverage(notes, merged))

        if do_judge:
            f = _safe(
                lambda: judge_mod.evaluate(
                    merged, notes, cfg.judge,
                    injection_test=injection_test, limit=judge_limit,
                ),
                f"judge {slug}/{mode}",
            )
            if f:
                entry["verbatim_rate"] = f.verbatim_rate
                entry["judged_claims"] = f.judged_claims
                entry["support_rate"] = f.support_rate
                entry["injections_flagged"] = f.injection_attempts_detected
                # Calibration. Carried alongside the support rate rather than
                # folded into it, so a reader can see how much of the score
                # rests on judgements the panel actually settled.
                entry["undecided_claims"] = f.undecided_claims
                entry["abstention_rate"] = f.abstention_rate
                entry["panel_agreement"] = f.mean_item_agreement
                entry["support_rate_decided"] = f.support_rate_decided
        else:
            # The verbatim check needs no model, so always report it.
            index = merged.by_uid()
            import re as _re
            full = _re.sub(
                r"\s+", " ", " ".join(u.text for u in merged.utterances)
            ).strip()
            items = notes.all_items()
            vb = sum(
                1 for i in items
                if judge_mod.is_verbatim(i.text, index, full)
            )
            entry["verbatim_rate"] = (vb / len(items)) if items else 0.0

        if row["annotated"]:
            for cat, gold_key in (
                ("action_items", "gold_action_items"),
                ("decisions", "gold_decisions"),
            ):
                pred = (
                    notes.action_items if cat == "action_items"
                    else notes.decisions
                )
                r = _safe(
                    lambda: pr_mod.score_category(
                        pred, gold.get(gold_key, []), cat
                    ),
                    f"P/R {slug}/{mode}/{cat}",
                )
                if r:
                    entry[f"{cat}_precision"] = r.precision
                    entry[f"{cat}_recall"] = r.recall
                    entry[f"{cat}_f1"] = r.f1

        row["modes"][mode] = entry

    # ---- Speaker-label bias probe
    # Read rather than run: the probe costs one full selection pass per
    # condition, which does not belong inside a reporting command. Produce it
    # with src/eval/bias.py and this picks it up.
    bpath = work / "bias.json"
    if bpath.exists():
        try:
            row["bias"] = read_raw(bpath).get("conditions", [])
        except Exception as exc:
            log.warning("could not read %s: %s", bpath, exc)

    return row


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def _num(v: Optional[float], places: int = 2) -> str:
    return f"{v:.{places}f}" if v is not None else "—"


def render_markdown(rows: List[Dict[str, Any]], judged: bool) -> str:
    out: List[str] = []
    out.append("# Evaluation Results")
    out.append("")
    out.append(
        f"Corpus: {len(rows)} US Supreme Court oral arguments (Oyez, CC-BY-NC). "
        f"Total audio: "
        f"{sum(r.get('duration_min', 0) for r in rows):.0f} minutes."
    )
    out.append("")

    # ---- Table 1: transcription
    out.append("## Table 1 — Transcription and diarization")
    out.append("")
    out.append(
        "| Case | Min | WER | Sub. rate | Legal sub. rate | Ratio | DER | "
        "Speakers ref/hyp | ASR RTF |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sub, lsub = r.get("substitution_rate"), r.get("legal_substitution_rate")
        ratio = (lsub / sub) if (sub and lsub) else None
        out.append(
            f"| {r.get('case_name', r['id'])[:34]} "
            f"| {r.get('duration_min', 0):.0f} "
            f"| {_pct(r.get('wer'))} | {_pct(sub)} | {_pct(lsub)} "
            f"| {_num(ratio, 1) + '×' if ratio else '—'} "
            f"| {_pct(r.get('der'))} "
            f"| {r.get('reference_speakers', '—')}/{r.get('hyp_speakers', '—')} "
            f"| {_num(r.get('asr_rtf'), 3)} |"
        )
    mean_wer = _mean([r.get("wer") for r in rows])
    mean_sub = _mean([r.get("substitution_rate") for r in rows])
    mean_lsub = _mean([r.get("legal_substitution_rate") for r in rows])
    mean_ratio = (mean_lsub / mean_sub) if (mean_sub and mean_lsub) else None
    out.append(
        f"| **mean** |  | **{_pct(mean_wer)}** | **{_pct(mean_sub)}** "
        f"| **{_pct(mean_lsub)}** "
        f"| **{_num(mean_ratio, 1) + '×' if mean_ratio else '—'}** "
        f"| **{_pct(_mean([r.get('der') for r in rows]))}** |  "
        f"| **{_num(_mean([r.get('asr_rtf') for r in rows]), 3)}** |"
    )
    out.append("")
    out.append(
        "> **Why the substitution rate is the column to read.** A verbatim "
        "court transcript records false starts, repetitions and "
        "interruptions; the recogniser cleans them up. Scored against each "
        "other, deletions dominate — on this corpus they were the large "
        "majority of all errors — and they measure a transcription-convention "
        "disagreement rather than a recognition failure. Substitutions are "
        "where the model wrote a *different word*, which is the error class "
        "that matters when the word is a party's name."
    )
    out.append("")
    if mean_ratio and mean_ratio > 1.2:
        out.append(
            f"> Legally load-bearing tokens — case names, citations, statutory "
            f"references, Latin terms — are **{mean_ratio:.1f}× more likely to "
            f"be substituted** than the average token. Aggregate WER conceals "
            f"this completely: measured with deletions included, the same "
            f"corpus shows legal terms as no worse than average."
        )
        out.append("")

    # ---- Table 2: note modes
    out.append("## Table 2 — Note generation modes")
    out.append("")
    # Built as a column list rather than by appending to a formatted string.
    # The previous version bolted the judged columns on with header[:-1], which
    # ate the trailing pipe and left the header one separator short of its own
    # row - invisible with two extra columns, and a broken table with four.
    cols = [
        "Mode", "Model", "Verbatim", "Content cov.", "Temporal spread",
        "Compression", "Invalid uids", "Sec",
    ]
    if judged:
        cols += ["Judged support", "Undecided", "Panel agr.", "Injections"]

    def _row(cells: List[str]) -> str:
        assert len(cells) == len(cols), (
            f"row has {len(cells)} cells but the table has {len(cols)} columns"
        )
        return "| " + " | ".join(cells) + " |"

    out.append(_row(cols))
    out.append("|" + "---|" * len(cols))

    for mode in MODES:
        entries = [r["modes"][mode] for r in rows
                   if r.get("modes", {}).get(mode)]
        if not entries:
            continue
        cells = [
            f"`{mode}`",
            entries[0].get("model") or "none",
            _pct(_mean([e.get("verbatim_rate") for e in entries])),
            _pct(_mean([e.get("content_coverage") for e in entries])),
            _pct(_mean([e.get("temporal_spread") for e in entries])),
            _pct(_mean([e.get("compression_ratio") for e in entries])),
            _num(_mean([e.get("validation_warnings") for e in entries]), 1),
            _num(_mean([e.get("seconds") for e in entries]), 1),
        ]
        if judged:
            cells += [
                _pct(_mean([e.get("support_rate") for e in entries])),
                _pct(_mean([e.get("abstention_rate") for e in entries])),
                _num(_mean([e.get("panel_agreement") for e in entries]), 2),
                _num(_mean([e.get("injections_flagged") for e in entries]), 1),
            ]
        out.append(_row(cells))
    out.append("")

    if judged:
        abst = _mean([
            e.get("abstention_rate")
            for r in rows for e in r.get("modes", {}).values()
        ])
        out.append(
            "> `Judged support` counts an undecided claim against the system. "
            "`Undecided` is the share of claims where the self-consistency "
            "panel split and no verdict was recorded, and `Panel agr.` is the "
            "mean fraction of the panel agreeing on a checklist item. Both "
            "exist because model-based scorers are confident well past their "
            "accuracy, and a majority vote hides that: 3–0 and 2–1 both emerge "
            "as \"supported\". Reporting the abstention rate makes the "
            "instability visible instead of averaging it away."
        )
        if abst is not None and abst > 0.3:
            out.append("")
            out.append(
                f"> **The judge abstained on {abst * 100:.0f}% of claims.** At "
                f"that rate the support column is weak evidence. Raise "
                f"`judge.self_consistency_runs` and re-run before drawing a "
                f"conclusion from it."
            )
        out.append("")
    out.append(
        "> `Verbatim` is the fraction of note lines that are byte-exact spans "
        "of the transcript. It is verified without a model and is therefore "
        "not subject to judge error. For the two extractive modes it is a "
        "structural guarantee; anything below 100% there is a pipeline defect, "
        "not a model failure."
    )
    out.append("")

    # ---- Table 3: extraction P/R
    annotated = [r for r in rows if r.get("annotated")]
    out.append("## Table 3 — Action item and decision extraction")
    out.append("")
    if not annotated:
        out.append(
            "*No gold file carries expert annotations yet, so precision and "
            "recall cannot be reported. Run "
            "`python src/eval/extraction_pr.py --gold <file> --make-template`, "
            "annotate, then re-run this report. Reporting zeros here instead "
            "would be worse than reporting nothing.*"
        )
    else:
        out.append(
            "| Mode | Action P | Action R | Action F1 | "
            "Decision P | Decision R | Decision F1 |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for mode in MODES:
            entries = [r["modes"][mode] for r in annotated
                       if r.get("modes", {}).get(mode)]
            if not entries:
                continue
            out.append(
                f"| `{mode}` "
                f"| {_num(_mean([e.get('action_items_precision') for e in entries]), 3)} "
                f"| {_num(_mean([e.get('action_items_recall') for e in entries]), 3)} "
                f"| {_num(_mean([e.get('action_items_f1') for e in entries]), 3)} "
                f"| {_num(_mean([e.get('decisions_precision') for e in entries]), 3)} "
                f"| {_num(_mean([e.get('decisions_recall') for e in entries]), 3)} "
                f"| {_num(_mean([e.get('decisions_f1') for e in entries]), 3)} |"
            )
    out.append("")

    # ---- Table 4: speaker-label bias
    biased_rows = [r for r in rows if r.get("bias")]
    out.append("## Table 4 — Speaker-label bias in note selection")
    out.append("")
    if not biased_rows:
        out.append(
            "*No bias probe has been run. Produce one with "
            "`python src/eval/bias.py --merged work/<case>/merged.json "
            "--out work/<case>/bias.json` and re-run this report. Every other "
            "table in this document measures whether the notes are correct; "
            "this is the only one that asks whether they are even-handed, and "
            "a note set can score perfectly everywhere else while quoting one "
            "side of a case and ignoring the other.*"
        )
    else:
        out.append(
            "Selection re-run over transcripts whose text is byte-identical "
            "and whose speaker labels have been altered. `control` leaves the "
            "labels alone and measures how much the selection moves under "
            "resampling; nothing below that floor is evidence. Substitute "
            "labels are width-matched to the originals so that every "
            "condition is split into the same windows — otherwise a condition "
            "differs from the baseline in two ways at once and its shift "
            "cannot be attributed to the labels."
        )
        out.append("")
        out.append(
            "| Case | Condition | Jaccard vs baseline | Max share shift | "
            "Most-shifted speaker | Above noise floor |"
        )
        out.append("|---|---|---|---|---|---|")
        for r in biased_rows:
            for c in r["bias"]:
                cond = c.get("condition", "?")
                evidence = c.get("exceeds_noise_floor")
                mark = (
                    "— (floor)" if cond == "control"
                    else "confounded" if not c.get("chunking_matched", True)
                    else "n/a" if evidence is None
                    else "**yes**" if evidence else "no"
                )
                out.append(
                    f"| {r.get('case_name', r['id'])[:28]} "
                    f"| `{cond}` "
                    f"| {_num(c.get('jaccard'), 3)} "
                    f"| {_num(c.get('max_share_shift'), 3)} "
                    f"| {c.get('shifted_speaker') or '—'} "
                    f"| {mark} |"
                )
        out.append("")

        flagged = [
            (r, c) for r in biased_rows for c in r["bias"]
            if c.get("exceeds_noise_floor")
        ]
        if flagged:
            out.append(
                "> **Selection is label-sensitive.** In the rows marked above, "
                "changing nothing but the speaker labels redistributed the "
                "notes across the people speaking by more than resampling the "
                "identical transcript did:"
            )
            for r, c in flagged[:6]:
                out.append(
                    f"> - {r.get('case_name', r['id'])[:40]} (`"
                    f"{c.get('condition')}`): {c.get('shifted_speaker')}'s "
                    f"share moved {float(c.get('max_share_shift', 0)) * 100:.1f} "
                    f"points."
                )
            out.append(
                "> This is invisible to WER, DER, verbatim rate and judged "
                "support alike, all of which are satisfied by notes that "
                "faithfully quote one party and omit the other."
            )
        else:
            out.append(
                "> No condition moved the selection further than resampling "
                "the identical transcript did. On this corpus, what gets "
                "selected is driven by content rather than by who is speaking. "
                "This is a negative result, and it is reported because the "
                "measurement is only worth anything if a null outcome is "
                "reported as readily as a positive one."
            )
        out.append("")

    # ---- Qualitative
    out.append("## Legal-term error examples")
    out.append("")
    seen = set()
    shown = 0
    for r in rows:
        for e in r.get("top_errors", []):
            if not e.get("legal"):
                continue
            key = (e["reference"], e["hypothesis"])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                f"- `{e['reference']}` → `{e['hypothesis']}` "
                f"({e['count']}×, {r.get('case_name', r['id'])[:40]})"
            )
            shown += 1
            if shown >= 20:
                break
        if shown >= 20:
            break
    if not shown:
        out.append("*(none recorded — run the ASR stage first)*")
    out.append("")

    out.append("---")
    out.append("")
    out.append(
        "*All local-tier numbers produced on a single consumer laptop "
        "(RTX 4060 8 GB, 16 GB RAM). No audio or transcript left the machine. "
        "Hosted-tier rows, where present, were produced on rented GPU against "
        "public court audio only.*"
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate evaluation report")
    ap.add_argument("--gold", default="data/gold")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="results")
    ap.add_argument("--judge", action="store_true",
                    help="run the LLM judge on abstractive claims (slow)")
    ap.add_argument("--injection-test", action="store_true")
    ap.add_argument("--judge-limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gold_dir = pathlib.Path(args.gold)
    work_root = pathlib.Path(args.work)
    out_dir = ensure_dir(args.out)

    gold_files = sorted(
        p for p in gold_dir.glob("*.json") if not p.name.endswith(".annotate.json")
    )
    if not gold_files:
        log.error(
            "no gold files in %s. Run: python src/eval/fetch_oyez.py "
            "--term 2019 --limit 5", gold_dir,
        )
        return 2

    rows = []
    for gp in gold_files:
        log.info("=== %s", gp.name)
        row = evaluate_case(
            gp, work_root, cfg, args.judge, args.injection_test,
            args.judge_limit,
        )
        if row:
            rows.append(row)

    if not rows:
        log.error("nothing evaluated - run the pipeline first")
        return 1

    write_json(out_dir / "results.json", {"cases": rows})
    md = render_markdown(rows, judged=args.judge)
    (out_dir / "results.md").write_text(md, encoding="utf-8")

    print("\n" + md + "\n")
    log.info("wrote %s/results.md and %s/results.json", out_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
