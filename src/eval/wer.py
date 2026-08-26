"""
Word error rate, with a legal-term breakdown.

Two design notes.

1. No jiwer dependency. WER is a Levenshtein alignment and we need the
   alignment itself - not just the score - in order to attribute each error to
   a reference token and ask whether that token was legally load-bearing. Most
   WER libraries return the score and throw the alignment away.

2. Anchor-based segmentation. A one-hour argument is ~10,000 reference tokens.
   A full DP matrix is 10^8 cells, which in Python is unusable. Instead we
   find n-gram anchors that occur exactly once in both sequences, take the
   longest increasing subsequence of those anchor pairs, and run exact DP only
   in the gaps between anchors. On real ASR output the gaps are short, so this
   is exact in practice and runs in seconds rather than hours.

The headline number this module exists to produce is `legal_term_wer`. Overall
WER around 5-8% is normal for Whisper on courtroom audio and looks reassuring.
The error rate restricted to case names, citations, statutory references, and
Latin terms is typically several times higher, and those are precisely the
tokens whose corruption Section 8.4 warns propagates downstream. A summary
built on a transcript that misheard the case name is wrong in a way no
summarisation method can repair.

Usage:
    python src/eval/wer.py --gold data/gold/2019_17-1618.json \\
        --hyp work/2019_17-1618/transcript.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.schemas import ASRResult, WERResult                       # noqa: E402
from src.util.io import read_json, read_raw, write_json            # noqa: E402
from src.util.logging import get_logger                            # noqa: E402
from src.util.textnorm import legal_token_indices, tokenize        # noqa: E402

log = get_logger("wer")

MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"
Op = Tuple[str, int, int]      # (kind, ref_index, hyp_index); -1 when absent


# --------------------------------------------------------------------------
# Exact DP on short sequences
# --------------------------------------------------------------------------


def _align_exact(ref: Sequence[str], hyp: Sequence[str]) -> List[Op]:
    n, m = len(ref), len(hyp)
    if n == 0:
        return [(INS, -1, j) for j in range(m)]
    if m == 0:
        return [(DEL, i, -1) for i in range(n)]

    # cost[i][j] = edits to turn ref[:i] into hyp[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 0=diag 1=del 2=ins

    for i in range(1, n + 1):
        cost[i][0] = i
        back[i][0] = 1
    for j in range(1, m + 1):
        cost[0][j] = j
        back[0][j] = 2

    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = cost[i], cost[i - 1]
        brow = back[i]
        for j in range(1, m + 1):
            d = prev[j - 1] + (0 if ri == hyp[j - 1] else 1)
            de = prev[j] + 1
            ins = row[j - 1] + 1
            best = d
            b = 0
            if de < best:
                best, b = de, 1
            if ins < best:
                best, b = ins, 2
            row[j] = best
            brow[j] = b

    ops: List[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == 0:
            kind = MATCH if ref[i - 1] == hyp[j - 1] else SUB
            ops.append((kind, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or back[i][j] == 1):
            ops.append((DEL, i - 1, -1))
            i -= 1
        else:
            ops.append((INS, -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


# --------------------------------------------------------------------------
# Anchor-based segmentation for long sequences
# --------------------------------------------------------------------------


def _unique_ngram_positions(seq: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Counter = Counter()
    first: Dict[Tuple[str, ...], int] = {}
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        counts[g] += 1
        if g not in first:
            first[g] = i
    return {g: first[g] for g, c in counts.items() if c == 1}


def _lis_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Longest increasing subsequence over hyp positions, pairs pre-sorted by ref."""
    if not pairs:
        return []
    tails: List[int] = []
    tail_idx: List[int] = []
    parent = [-1] * len(pairs)

    for k, (_r, h) in enumerate(pairs):
        pos = bisect_left(tails, h)
        if pos == len(tails):
            tails.append(h)
            tail_idx.append(k)
        else:
            tails[pos] = h
            tail_idx[pos] = k
        parent[k] = tail_idx[pos - 1] if pos > 0 else -1

    out: List[Tuple[int, int]] = []
    k = tail_idx[-1]
    while k != -1:
        out.append(pairs[k])
        k = parent[k]
    out.reverse()
    return out


def align(
    ref: Sequence[str], hyp: Sequence[str], ngram: int = 5,
    max_exact: int = 1200,
) -> List[Op]:
    """
    Align two token sequences, exactly for short input and anchor-segmented
    for long input.
    """
    if len(ref) * len(hyp) <= max_exact * max_exact:
        return _align_exact(ref, hyp)

    ref_ng = _unique_ngram_positions(ref, ngram)
    hyp_ng = _unique_ngram_positions(hyp, ngram)
    shared = [(ref_ng[g], hyp_ng[g]) for g in ref_ng.keys() & hyp_ng.keys()]
    shared.sort()
    anchors = _lis_pairs(shared)

    # Keep anchors far enough apart that the DP gaps stay small, but not so
    # sparse that a gap explodes.
    filtered: List[Tuple[int, int]] = []
    last_r = last_h = -10 ** 9
    for r, h in anchors:
        if r - last_r >= ngram and h - last_h >= ngram:
            filtered.append((r, h))
            last_r, last_h = r, h
    anchors = filtered

    if not anchors:
        log.warning(
            "no shared %d-gram anchors; falling back to exact DP (may be slow)",
            ngram,
        )
        return _align_exact(ref, hyp)

    log.info(
        "anchor alignment: %d anchors over %d ref / %d hyp tokens",
        len(anchors), len(ref), len(hyp),
    )

    ops: List[Op] = []
    r_cur = h_cur = 0
    for r, h in anchors:
        if r > r_cur or h > h_cur:
            gap = _align_exact(ref[r_cur:r], hyp[h_cur:h])
            for kind, ri, hi in gap:
                ops.append((
                    kind,
                    r_cur + ri if ri >= 0 else -1,
                    h_cur + hi if hi >= 0 else -1,
                ))
        for k in range(ngram):
            ops.append((MATCH, r + k, h + k))
        r_cur, h_cur = r + ngram, h + ngram

    if r_cur < len(ref) or h_cur < len(hyp):
        tail = _align_exact(ref[r_cur:], hyp[h_cur:])
        for kind, ri, hi in tail:
            ops.append((
                kind,
                r_cur + ri if ri >= 0 else -1,
                h_cur + hi if hi >= 0 else -1,
            ))
    return ops


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(
    reference_raw: str, hypothesis_raw: str, top_k: int = 25
) -> WERResult:
    ref = tokenize(reference_raw)
    hyp = tokenize(hypothesis_raw)
    legal_idx = legal_token_indices(reference_raw)

    log.info(
        "scoring: %d reference tokens (%d legally load-bearing), %d hypothesis",
        len(ref), len(legal_idx), len(hyp),
    )

    ops = align(ref, hyp)

    subs = dels = ins = 0
    legal_errors = 0
    legal_subs = 0
    legal_seen = 0
    error_pairs: Counter = Counter()
    legal_error_pairs: Counter = Counter()

    for kind, ri, hi in ops:
        is_legal = ri >= 0 and ri in legal_idx
        if is_legal and kind in (MATCH, SUB, DEL):
            legal_seen += 1

        if kind == SUB:
            subs += 1
            error_pairs[(ref[ri], hyp[hi])] += 1
            if is_legal:
                legal_errors += 1
                legal_subs += 1
                legal_error_pairs[(ref[ri], hyp[hi])] += 1
        elif kind == DEL:
            dels += 1
            error_pairs[(ref[ri], "<deleted>")] += 1
            if is_legal:
                legal_errors += 1
                legal_error_pairs[(ref[ri], "<deleted>")] += 1
        elif kind == INS:
            ins += 1

    n_ref = len(ref)
    wer = (subs + dels + ins) / n_ref if n_ref else 0.0
    legal_wer = (legal_errors / legal_seen) if legal_seen else None
    sub_rate = (subs / n_ref) if n_ref else None
    legal_sub_rate = (legal_subs / legal_seen) if legal_seen else None

    top = [
        {
            "reference": r, "hypothesis": h, "count": c,
            "legal": bool(legal_error_pairs.get((r, h))),
        }
        for (r, h), c in error_pairs.most_common(top_k)
    ]
    # Always surface the worst legal-term errors even if they are not in the
    # global top-k; those are the ones that matter for the discussion.
    for (r, h), c in legal_error_pairs.most_common(10):
        if not any(t["reference"] == r and t["hypothesis"] == h for t in top):
            top.append({
                "reference": r, "hypothesis": h, "count": c, "legal": True
            })

    result = WERResult(
        reference_words=n_ref,
        hypothesis_words=len(hyp),
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        wer=wer,
        legal_term_wer=legal_wer,
        legal_term_count=legal_seen,
        substitution_rate=sub_rate,
        legal_substitution_rate=legal_sub_rate,
        top_errors=top,
    )
    log.info(
        "WER %.2f%%  (S=%d D=%d I=%d)  |  legal-term WER %s over %d tokens",
        wer * 100, subs, dels, ins,
        f"{legal_wer * 100:.2f}%" if legal_wer is not None else "n/a",
        legal_seen,
    )
    if legal_wer is not None and wer > 0:
        log.info(
            "legal terms are %.1fx more likely to be wrong than the average "
            "token", legal_wer / wer if wer else 0.0,
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="WER with legal-term breakdown")
    ap.add_argument("--gold", required=True, help="gold JSON from fetch_oyez")
    ap.add_argument("--hyp", required=True, help="transcript.json from ASR")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = read_raw(args.gold)
    asr = read_json(args.hyp, ASRResult)

    result = score(gold["reference_text"], asr.full_text())

    if args.out:
        write_json(args.out, result)
        log.info("wrote %s", args.out)
    else:
        print()
        print(f"  WER              {result.wer * 100:6.2f}%")
        if result.legal_term_wer is not None:
            print(f"  Legal-term WER   {result.legal_term_wer * 100:6.2f}%  "
                  f"({result.legal_term_count} tokens)")
        print(f"  S/D/I            {result.substitutions}/"
              f"{result.deletions}/{result.insertions}")
        print("\n  Top errors (legal terms marked *):")
        for e in result.top_errors[:15]:
            mark = "*" if e["legal"] else " "
            print(f"   {mark} {e['reference']:<22} -> {e['hypothesis']:<22} "
                  f"x{e['count']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
