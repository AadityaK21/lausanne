"""
Stage 4a - Rule-based extractive notes. Zero neural models.

This is the survey's Section 8.2 extractive mode, reimplemented over
speaker-attributed utterances. It is fully deterministic, runs in
milliseconds on CPU, and cannot fabricate: every emitted line is a byte-exact
utterance from the transcript, carried with its uid and timestamp.

It serves two purposes. It is a usable note mode in its own right - and it is
the faithfulness *floor* against which the two LLM modes are measured. A
model-based summariser that cannot beat this on informativeness while also
matching its faithfulness has not earned the extra 5 GB of VRAM.

Run standalone:
    python src/stages/extract.py --merged work/h/merged.json --out work/h/notes_rule.json
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Set

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.config import load_config, ExtractConfig             # noqa: E402
from src.schemas import (                                      # noqa: E402
    Citation, MergedTranscript, NoteItem, NoteMode, NoteSet, Utterance,
)
from src.util.io import read_json, write_json                  # noqa: E402
from src.util.logging import get_logger                        # noqa: E402

log = get_logger("extract")


STOPWORDS: Set[str] = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with you your yours yourself yourselves
would can may might must shall going get got go know think say said says like
one two also well right okay yes mr mrs ms dr thank thanks please sorry
question answer honor court counsel case matter really actually kind sort
""".split())


def _content_words(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z'\-]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def _compile(cues) -> List[re.Pattern]:
    return [re.compile(c, re.IGNORECASE) for c in cues]


def _cite(u: Utterance) -> Citation:
    return Citation(uid=u.uid, start=u.start, end=u.end, speaker=u.speaker)


def _item(u: Utterance, cue: str | None = None) -> NoteItem:
    """Every item is verbatim by construction - the text IS the utterance."""
    return NoteItem(text=u.text, citations=[_cite(u)], verbatim=True, cue=cue)


def score_utterances(
    utterances: List[Utterance], cfg: ExtractConfig
) -> Dict[int, float]:
    """
    Score by mean corpus frequency of content words, as in Section 8.2, with
    two corrections that matter on courtroom audio:

      * TF is damped by log, so one repeated term cannot dominate.
      * The score is normalised by sentence length, otherwise the ranking
        simply returns the longest sentences.
    """
    freq: Counter = Counter()
    for u in utterances:
        freq.update(_content_words(u.text))

    if not freq:
        return {u.uid: 0.0 for u in utterances}

    max_freq = max(freq.values())
    scores: Dict[int, float] = {}
    for u in utterances:
        words = _content_words(u.text)
        if len(words) < cfg.min_sentence_words:
            scores[u.uid] = 0.0
            continue
        total = sum(math.log1p(freq[w]) / math.log1p(max_freq) for w in words)
        scores[u.uid] = total / math.sqrt(len(words))
    return scores


def match_cues(
    utterances: List[Utterance], patterns: List[re.Pattern]
) -> List[NoteItem]:
    out: List[NoteItem] = []
    for u in utterances:
        for pat in patterns:
            m = pat.search(u.text)
            if m:
                out.append(_item(u, cue=m.group(0).lower()))
                break
    return out


def extract(merged: MergedTranscript, cfg: ExtractConfig) -> NoteSet:
    t0 = time.perf_counter()
    utterances = merged.utterances
    scores = score_utterances(utterances, cfg)

    ranked = sorted(utterances, key=lambda u: scores[u.uid], reverse=True)

    # Scale selection with transcript length. Fixed counts are fine for an
    # hour of audio and absurd for two minutes: 8 summary lines plus 10 key
    # points drawn from 24 utterances reproduces most of the transcript and
    # calls it a summary. Caps at ~20% / ~25% of available utterances keep the
    # output a summary at every input size, while the configured values remain
    # the ceiling for long proceedings.
    n = len(utterances)
    n_summary = max(1, min(cfg.summary_sentences, int(n * 0.20)))
    n_key = max(1, min(cfg.key_points, int(n * 0.25)))

    # Preserve source order in the output, per Section 8.2 - a summary whose
    # sentences are reordered misrepresents the sequence of a proceeding.
    summary_pool = sorted(ranked[:n_summary], key=lambda u: u.uid)
    chosen = {u.uid for u in summary_pool}
    key_pool = sorted(
        [u for u in ranked if u.uid not in chosen][:n_key],
        key=lambda u: u.uid,
    )

    action_items = match_cues(utterances, _compile(cfg.action_cues))
    decisions = match_cues(utterances, _compile(cfg.decision_cues))

    freq: Counter = Counter()
    for u in utterances:
        freq.update(_content_words(u.text))
    keywords = [w for w, _ in freq.most_common(cfg.keywords)]

    elapsed = time.perf_counter() - t0
    log.info(
        "rule-based: %d summary, %d key points, %d decisions, %d action items "
        "(%.3fs)",
        len(summary_pool), len(key_pool), len(decisions), len(action_items),
        elapsed,
    )

    return NoteSet(
        mode=NoteMode.EXTRACTIVE_RULE,
        model=None,
        tier="none",
        summary=[_item(u) for u in summary_pool],
        key_points=[_item(u) for u in key_pool],
        decisions=decisions,
        action_items=action_items,
        keywords=keywords,
        elapsed_seconds=elapsed,
        llm_calls=0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule-based extractive notes")
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    merged = read_json(args.merged, MergedTranscript)

    try:
        notes = extract(merged, cfg.extract)
    except Exception as exc:
        log.error("extraction failed: %s", exc, exc_info=True)
        return 1

    write_json(args.out, notes)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
