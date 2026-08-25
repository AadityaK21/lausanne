"""
Stage 4b - LLM note generation, in two modes.

EXTRACTIVE_LLM
    The model reads a numbered transcript and returns *only integer uids*.
    Its response schema contains no free-text field for claims at all, so the
    output is re-materialised from the transcript by uid lookup. Fabrication
    is impossible in the same structural sense as the rule-based mode, but the
    *selection* is made by a language model rather than by word-frequency
    heuristics. This is the configuration the survey's Section 6 argues for
    and, as far as the author is aware, is not what existing local
    meeting-notes tools do.

ABSTRACTIVE
    Free generation with forced per-claim uid citations. Fluent, and subject
    to exactly the hallucination risk Section 6.1 documents. Included because
    you cannot argue that extraction is safer without measuring the thing it
    is safer than.

Map-reduce over chunks, because a one-hour hearing is 15-20k tokens and the
local tier runs an 8k window.

Run standalone:
    python src/stages/summarize.py --merged work/h/merged.json \\
        --mode extractive-llm --out work/h/notes_llm.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.config import load_config, LLMTier                    # noqa: E402
from src.schemas import (                                       # noqa: E402
    AbstractiveResponse, Citation, MergedTranscript, NoteItem, NoteMode,
    NoteSet, SelectionResponse, Utterance,
)
from src.util import gpu                                        # noqa: E402
from src.util.io import read_json, write_json                   # noqa: E402
from src.util.logging import get_logger                         # noqa: E402
from src.util.ollama_client import OllamaClient, preflight      # noqa: E402

log = get_logger("summarize")

PROMPT_DIR = pathlib.Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str, fallback: str) -> str:
    path = PROMPT_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Cheap token estimate. Deliberately conservative (over-estimates) so a
    chunk never silently overflows the context and gets truncated - a
    truncated chunk produces a summary of half a hearing with no error.
    """
    return max(1, int(len(text) / 3.5))


def render_line(u: Utterance) -> str:
    return f"[{u.uid}] {u.speaker}: {u.text}"


def chunk_utterances(
    utterances: List[Utterance], budget_tokens: int, overlap_tokens: int
) -> List[List[Utterance]]:
    """Split into overlapping windows that fit the model's context budget."""
    chunks: List[List[Utterance]] = []
    current: List[Utterance] = []
    current_tokens = 0

    for u in utterances:
        cost = estimate_tokens(render_line(u))
        if current and current_tokens + cost > budget_tokens:
            chunks.append(current)
            # Carry back enough tail context that a sentence split across a
            # boundary is still interpretable in the next window.
            back: List[Utterance] = []
            back_tokens = 0
            for prev in reversed(current):
                c = estimate_tokens(render_line(prev))
                if back_tokens + c > overlap_tokens:
                    break
                back.insert(0, prev)
                back_tokens += c
            current, current_tokens = list(back), back_tokens
        current.append(u)
        current_tokens += cost

    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_MAP_SELECT_FALLBACK = """\
You are assisting a lawyer who is reviewing the record of a legal proceeding.

Below is a numbered, speaker-attributed excerpt of a transcript. Each line is
formatted as:

    [uid] SPEAKER: text

Your task is to SELECT lines. You do not write anything. You return uids.

Choose:
- summary_uids: up to {n_summary} lines that together convey what this excerpt
  is about. Prefer lines that state substance over procedure or pleasantries.
- key_point_uids: up to {n_key} additional lines carrying an important fact,
  argument, concession, or question.
- decision_uids: lines where something was RULED, GRANTED, DENIED, SUSTAINED,
  OVERRULED, AGREED, STIPULATED, or otherwise settled.
- action_item_uids: lines imposing an obligation, deadline, filing, production,
  or follow-up on a party.
- keywords: up to {n_kw} short topical terms that appear in the excerpt.

Rules:
- Only return uids that appear in the excerpt below. Never invent a uid.
- A uid may appear in more than one list if genuinely applicable.
- If a category has nothing in this excerpt, return an empty list. Do not pad.

EXCERPT:
{excerpt}
"""

_REDUCE_SELECT_FALLBACK = """\
You are consolidating notes for a legal proceeding.

Below are candidate lines selected from across the whole transcript, in
chronological order, formatted as:

    [uid] SPEAKER: text

Select the final set. Return uids only.

- summary_uids: exactly the {n_summary} lines that best summarise the WHOLE
  proceeding, in chronological order.
- key_point_uids: up to {n_key} further significant lines.
- decision_uids: every line that records something settled or ruled.
- action_item_uids: every line that imposes an obligation or deadline.
- keywords: up to {n_kw} topical terms.

Only return uids present below. Never invent a uid. Drop redundant lines that
repeat a point already covered.

CANDIDATES:
{excerpt}
"""

_MAP_ABSTRACTIVE_FALLBACK = """\
You are a legal assistant producing notes from a transcript excerpt.

Each transcript line is formatted as:

    [uid] SPEAKER: text

Write concise notes about this excerpt. For EVERY claim you write, list the
uids of the transcript lines that support it in supporting_uids. A claim with
no supporting uid is not permitted.

Produce:
- summary: 2-4 claims describing what happened in this excerpt.
- key_points: important facts, arguments, or concessions.
- decisions: anything ruled, granted, denied, agreed, or stipulated.
- action_items: obligations, deadlines, filings, or follow-ups.
- keywords: short topical terms.

Do not speculate about anything not stated in the excerpt. Do not add legal
analysis, citations, or context from your own knowledge.

EXCERPT:
{excerpt}
"""

_REDUCE_ABSTRACTIVE_FALLBACK = """\
Consolidate these per-section notes into one set of notes for the whole
proceeding. Keep the supporting_uids attached to each claim. Merge duplicates,
drop redundancy, preserve chronological sense.

Produce at most {n_summary} summary claims and {n_key} key points; keep all
distinct decisions and action items. Add nothing that is not already present
in the notes below.

NOTES:
{excerpt}
"""


# --------------------------------------------------------------------------
# Validation - the mechanism that makes extraction faithful
# --------------------------------------------------------------------------


def validate_uids(
    uids: List[int], index: Dict[int, Utterance], label: str
) -> Tuple[List[int], List[str]]:
    """
    Drop uids that do not exist, and report them.

    These warnings are a measurement, not just hygiene: the rate at which a
    model emits non-existent uids is a direct, cheap proxy for its tendency to
    fabricate, and it is reported in the evaluation.
    """
    valid, warnings, seen = [], [], set()
    for uid in uids:
        if uid in seen:
            continue
        seen.add(uid)
        if uid in index:
            valid.append(uid)
        else:
            warnings.append(f"{label}: model emitted non-existent uid {uid}")
    return valid, warnings


def items_from_uids(uids: List[int], index: Dict[int, Utterance]) -> List[NoteItem]:
    """Re-materialise text from the transcript. This is where faithfulness is won."""
    items = []
    for uid in sorted(uids):
        u = index[uid]
        items.append(
            NoteItem(
                text=u.text,
                citations=[
                    Citation(uid=u.uid, start=u.start, end=u.end, speaker=u.speaker)
                ],
                verbatim=True,
            )
        )
    return items


# --------------------------------------------------------------------------
# Extractive-LLM mode
# --------------------------------------------------------------------------


def run_extractive_llm(
    merged: MergedTranscript, tier: LLMTier, cfg
) -> NoteSet:
    client = OllamaClient(host=tier.host, keep_alive=tier.keep_alive)
    preflight(client, tier.model)
    client.unload_all()
    gpu.assert_headroom(cfg.memory.llm_ceiling_mb, "summarize")

    index = merged.by_uid()
    chunks = chunk_utterances(
        merged.utterances, tier.chunk_tokens, tier.chunk_overlap_tokens
    )
    log.info(
        "extractive-llm | %s | %d utterances -> %d chunks",
        tier.model, len(merged.utterances), len(chunks),
    )

    map_tmpl = load_prompt("map_select.txt", _MAP_SELECT_FALLBACK)
    reduce_tmpl = load_prompt("reduce_select.txt", _REDUCE_SELECT_FALLBACK)

    t0 = time.perf_counter()
    warnings: List[str] = []
    calls = 0
    pooled: Dict[str, List[int]] = {
        "summary": [], "key_points": [], "decisions": [], "action_items": []
    }
    keywords: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        excerpt = "\n".join(render_line(u) for u in chunk)
        prompt = map_tmpl.format(
            excerpt=excerpt, n_summary=4, n_key=5, n_kw=8
        )
        log.info("  map %d/%d (%d lines)", i, len(chunks), len(chunk))
        resp = client.chat_structured(
            tier.model,
            [{"role": "user", "content": prompt}],
            SelectionResponse,
            temperature=tier.temperature,
            num_ctx=tier.context_tokens,
            num_predict=tier.num_predict,
        )
        calls += 1

        allowed = {u.uid for u in chunk}
        for field, key in (
            ("summary_uids", "summary"),
            ("key_point_uids", "key_points"),
            ("decision_uids", "decisions"),
            ("action_item_uids", "action_items"),
        ):
            raw = getattr(resp, field)
            valid, warns = validate_uids(raw, index, f"map{i}.{field}")
            # Also reject uids that exist globally but not in the window we
            # showed the model - that is a different failure than inventing an
            # id, and worth counting separately.
            out_of_window = [u for u in valid if u not in allowed]
            if out_of_window:
                warns.append(
                    f"map{i}.{field}: {len(out_of_window)} uids outside the "
                    f"presented window"
                )
            pooled[key].extend([u for u in valid if u in allowed])
            warnings.extend(warns)
        keywords.extend(resp.keywords)

    # Reduce over the union of everything selected, if it is large enough to
    # need consolidating.
    candidate_uids = sorted(set(sum(pooled.values(), [])))
    if len(chunks) > 1 and candidate_uids:
        excerpt = "\n".join(render_line(index[u]) for u in candidate_uids)
        if estimate_tokens(excerpt) > tier.chunk_tokens:
            log.warning(
                "candidate set too large for one reduce call; truncating to "
                "the highest-value window"
            )
            excerpt = "\n".join(
                render_line(index[u]) for u in candidate_uids
            )[: tier.chunk_tokens * 3]

        prompt = reduce_tmpl.format(
            excerpt=excerpt, n_summary=8, n_key=10, n_kw=15
        )
        log.info("  reduce over %d candidates", len(candidate_uids))
        resp = client.chat_structured(
            tier.model,
            [{"role": "user", "content": prompt}],
            SelectionResponse,
            temperature=tier.temperature,
            num_ctx=tier.context_tokens,
            num_predict=tier.num_predict,
        )
        calls += 1

        allowed = set(candidate_uids)
        final: Dict[str, List[int]] = {}
        for field, key in (
            ("summary_uids", "summary"),
            ("key_point_uids", "key_points"),
            ("decision_uids", "decisions"),
            ("action_item_uids", "action_items"),
        ):
            valid, warns = validate_uids(getattr(resp, field), index, f"reduce.{field}")
            warnings.extend(warns)
            kept = [u for u in valid if u in allowed]
            # A reduce that returns nothing for a category it was given
            # candidates for is more likely a model failure than a genuine
            # empty; fall back to the pooled map result.
            final[key] = kept if kept else pooled[key]
        pooled = final
        if resp.keywords:
            keywords = resp.keywords

    client.unload(tier.model)

    # De-duplicate: a uid in summary should not repeat in key_points.
    summary_ids = list(dict.fromkeys(pooled["summary"]))
    key_ids = [u for u in dict.fromkeys(pooled["key_points"]) if u not in summary_ids]

    elapsed = time.perf_counter() - t0
    seen_kw, uniq_kw = set(), []
    for k in keywords:
        kl = k.strip().lower()
        if kl and kl not in seen_kw:
            seen_kw.add(kl)
            uniq_kw.append(k.strip())

    log.info(
        "extractive-llm done: %d summary, %d key, %d decisions, %d actions, "
        "%d calls, %.1fs, %d warnings",
        len(summary_ids), len(key_ids), len(pooled["decisions"]),
        len(pooled["action_items"]), calls, elapsed, len(warnings),
    )

    return NoteSet(
        mode=NoteMode.EXTRACTIVE_LLM,
        model=tier.model,
        tier=tier.name,
        summary=items_from_uids(summary_ids, index),
        key_points=items_from_uids(key_ids, index),
        decisions=items_from_uids(list(dict.fromkeys(pooled["decisions"])), index),
        action_items=items_from_uids(
            list(dict.fromkeys(pooled["action_items"])), index
        ),
        keywords=uniq_kw[:15],
        validation_warnings=warnings,
        elapsed_seconds=elapsed,
        llm_calls=calls,
    )


# --------------------------------------------------------------------------
# Abstractive mode
# --------------------------------------------------------------------------


def _sections_to_items(
    sections, index: Dict[int, Utterance], label: str
) -> Tuple[List[NoteItem], List[str]]:
    items, warnings = [], []
    for sec in sections:
        claim = (sec.claim or "").strip()
        if not claim:
            continue
        valid, warns = validate_uids(sec.supporting_uids, index, label)
        warnings.extend(warns)
        if not valid:
            warnings.append(f"{label}: uncited claim {claim[:70]!r}")
        cites = [
            Citation(
                uid=index[u].uid, start=index[u].start,
                end=index[u].end, speaker=index[u].speaker,
            )
            for u in valid
        ]
        # Mark verbatim only if the generated claim happens to be an exact
        # transcript span. Almost never true, and that is the point.
        exact = any(claim == index[u].text for u in valid)
        items.append(NoteItem(text=claim, citations=cites, verbatim=exact))
    return items, warnings


def run_abstractive(merged: MergedTranscript, tier: LLMTier, cfg) -> NoteSet:
    client = OllamaClient(host=tier.host, keep_alive=tier.keep_alive)
    preflight(client, tier.model)
    client.unload_all()
    gpu.assert_headroom(cfg.memory.llm_ceiling_mb, "summarize")

    index = merged.by_uid()
    chunks = chunk_utterances(
        merged.utterances, tier.chunk_tokens, tier.chunk_overlap_tokens
    )
    log.info(
        "abstractive | %s | %d utterances -> %d chunks",
        tier.model, len(merged.utterances), len(chunks),
    )

    map_tmpl = load_prompt("map_abstractive.txt", _MAP_ABSTRACTIVE_FALLBACK)
    reduce_tmpl = load_prompt("reduce_abstractive.txt", _REDUCE_ABSTRACTIVE_FALLBACK)

    t0 = time.perf_counter()
    warnings: List[str] = []
    calls = 0
    partials: List[AbstractiveResponse] = []

    for i, chunk in enumerate(chunks, 1):
        excerpt = "\n".join(render_line(u) for u in chunk)
        log.info("  map %d/%d (%d lines)", i, len(chunks), len(chunk))
        resp = client.chat_structured(
            tier.model,
            [{"role": "user", "content": map_tmpl.format(excerpt=excerpt)}],
            AbstractiveResponse,
            temperature=tier.temperature,
            num_ctx=tier.context_tokens,
            num_predict=tier.num_predict,
        )
        calls += 1
        partials.append(resp)

    if len(partials) == 1:
        final = partials[0]
    else:
        lines = []
        for p in partials:
            for group, name in (
                (p.summary, "SUMMARY"), (p.key_points, "KEY"),
                (p.decisions, "DECISION"), (p.action_items, "ACTION"),
            ):
                for sec in group:
                    uids = ",".join(str(u) for u in sec.supporting_uids)
                    lines.append(f"{name} | uids=[{uids}] | {sec.claim}")
        excerpt = "\n".join(lines)
        log.info("  reduce over %d partial claims", len(lines))
        final = client.chat_structured(
            tier.model,
            [{
                "role": "user",
                "content": reduce_tmpl.format(
                    excerpt=excerpt, n_summary=6, n_key=10
                ),
            }],
            AbstractiveResponse,
            temperature=tier.temperature,
            num_ctx=tier.context_tokens,
            num_predict=tier.num_predict,
        )
        calls += 1
        keywords = []
        for p in partials:
            keywords.extend(p.keywords)
        if not final.keywords:
            final.keywords = keywords

    client.unload(tier.model)

    summary, w1 = _sections_to_items(final.summary, index, "summary")
    key_points, w2 = _sections_to_items(final.key_points, index, "key_points")
    decisions, w3 = _sections_to_items(final.decisions, index, "decisions")
    actions, w4 = _sections_to_items(final.action_items, index, "action_items")
    warnings.extend(w1 + w2 + w3 + w4)

    elapsed = time.perf_counter() - t0
    log.info(
        "abstractive done: %d claims total, %d calls, %.1fs, %d warnings",
        len(summary) + len(key_points) + len(decisions) + len(actions),
        calls, elapsed, len(warnings),
    )

    seen, uniq = set(), []
    for k in final.keywords:
        kl = k.strip().lower()
        if kl and kl not in seen:
            seen.add(kl)
            uniq.append(k.strip())

    return NoteSet(
        mode=NoteMode.ABSTRACTIVE,
        model=tier.model,
        tier=tier.name,
        summary=summary,
        key_points=key_points,
        decisions=decisions,
        action_items=actions,
        keywords=uniq[:15],
        validation_warnings=warnings,
        elapsed_seconds=elapsed,
        llm_calls=calls,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM note generation")
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode", default="extractive-llm",
        choices=["extractive-llm", "abstractive"],
    )
    ap.add_argument("--tier", default=None, help="local | hosted")
    ap.add_argument("--model", default=None, help="override the tier's model")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tier = cfg.tier(args.tier)
    if args.model:
        tier.model = args.model

    merged = read_json(args.merged, MergedTranscript)

    try:
        if args.mode == "extractive-llm":
            notes = run_extractive_llm(merged, tier, cfg)
        else:
            notes = run_abstractive(merged, tier, cfg)
    except Exception as exc:
        log.error("summarization failed: %s", exc, exc_info=True)
        return 1

    write_json(args.out, notes)
    log.info("wrote %s", args.out)
    gpu.log_state("summarize exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
