"""
Stage 3 - Reconcile ASR words with speaker turns, emit sentence utterances.

This is the stage that produces the unit everything downstream cites: a
speaker-attributed, time-stamped sentence with a stable integer uid.

The uid is load-bearing. In extractive mode the summariser's entire output
vocabulary is the set of uids present here, so a claim that is not in this
file cannot be expressed at all. That is a structural guarantee, not a prompt
instruction, and it is the difference between "we asked the model not to
hallucinate" and "the model has no channel through which to hallucinate".

Pure CPU. No model is loaded, so this stage costs nothing on the VRAM budget.

Run standalone:
    python src/stages/merge.py --transcript work/h/transcript.json \\
        --speakers work/h/speakers.json --out work/h/merged.json
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time
from typing import List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.schemas import (                                     # noqa: E402
    ASRResult, DiarizationResult, MergedTranscript, SpeakerTurn, Utterance, Word,
)
from src.util.io import read_json, write_json                 # noqa: E402
from src.util.logging import get_logger                       # noqa: E402

log = get_logger("merge")


# Sentence terminators. The negative lookbehind keeps common abbreviations and
# single initials from splitting a sentence mid-citation ("Brown v. Bd. of
# Ed." must stay one unit, or the citation is destroyed by the segmentation).
_ABBREVS = (
    "mr", "mrs", "ms", "dr", "hon", "jr", "sr", "st", "no", "vs", "v",
    "u.s", "u.s.c", "c.f.r", "f.supp", "s.ct", "l.ed", "art", "sec", "secs",
    "cl", "amend", "para", "pp", "ed", "bd", "co", "corp", "inc", "ltd",
    "assn", "dept", "div", "cir", "dist", "et", "al", "i.e", "e.g", "etc",
)


def _is_terminator(token: str) -> bool:
    """Does this token end a sentence?"""
    stripped = token.strip()
    if not stripped or stripped[-1] not in ".!?":
        return False
    core = stripped.rstrip(".!?\"')]").lower()
    if core in _ABBREVS:
        return False
    # A single capital letter followed by a period is an initial: "John Q. Public"
    if len(core) == 1 and core.isalpha():
        return False
    return True


def _flatten_words(asr: ASRResult) -> List[Word]:
    words: List[Word] = []
    for seg in asr.segments:
        if seg.words:
            words.extend(seg.words)
        else:
            # No word timestamps: approximate by distributing the segment's
            # duration evenly across its tokens. Degrades speaker attribution
            # precision but keeps the pipeline running.
            toks = seg.text.split()
            if not toks:
                continue
            step = seg.duration / len(toks) if seg.duration > 0 else 0.0
            for i, tok in enumerate(toks):
                words.append(
                    Word(
                        start=seg.start + i * step,
                        end=seg.start + (i + 1) * step,
                        word=(" " + tok),
                        probability=None,
                    )
                )
    return words


def _speaker_for(word: Word, turns: List[SpeakerTurn]) -> str:
    """
    Attribute a word to the speaker turn it overlaps most.

    Falls back to the nearest turn by midpoint distance, which handles words
    landing in a diarization gap (breaths, short crosstalk) rather than
    dropping them.
    """
    if not turns:
        return "SPEAKER_00"

    best_label, best_overlap = None, 0.0
    for t in turns:
        overlap = min(word.end, t.end) - max(word.start, t.start)
        if overlap > best_overlap:
            best_overlap, best_label = overlap, t.speaker
    if best_label is not None:
        return best_label

    mid = (word.start + word.end) / 2.0
    nearest = min(
        turns,
        key=lambda t: 0.0 if t.start <= mid <= t.end
        else min(abs(mid - t.start), abs(mid - t.end)),
    )
    return nearest.speaker


def merge(
    asr: ASRResult, diar: Optional[DiarizationResult], max_words: int = 80
) -> MergedTranscript:
    t0 = time.perf_counter()
    words = _flatten_words(asr)
    if not words:
        raise ValueError("ASR produced no words - check the audio file")

    turns: List[SpeakerTurn] = []
    if diar is not None:
        turns = diar.exclusive_turns or diar.turns
        log.info(
            "using %s diarization (%d turns, %d speakers)",
            "exclusive" if diar.exclusive_turns else "standard",
            len(turns), diar.num_speakers_detected,
        )
    else:
        log.warning("no diarization supplied - all speech attributed to SPEAKER_00")

    labels = [_speaker_for(w, turns) for w in words]

    utterances: List[Utterance] = []
    buf_words: List[Word] = []
    buf_speaker: Optional[str] = None

    def flush() -> None:
        nonlocal buf_words, buf_speaker
        if not buf_words:
            return
        text = "".join(w.word for w in buf_words).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            probs = [w.probability for w in buf_words if w.probability is not None]
            utterances.append(
                Utterance(
                    uid=len(utterances),
                    start=buf_words[0].start,
                    end=buf_words[-1].end,
                    speaker=buf_speaker or "SPEAKER_00",
                    text=text,
                    confidence=(sum(probs) / len(probs)) if probs else None,
                )
            )
        buf_words = []

    for word, label in zip(words, labels):
        # A speaker change always closes the current utterance: a sentence
        # cannot span two speakers, and letting it do so corrupts attribution.
        if buf_speaker is not None and label != buf_speaker:
            flush()
        buf_speaker = label
        buf_words.append(word)

        if _is_terminator(word.word) or len(buf_words) >= max_words:
            flush()

    flush()

    speaker_labels = sorted({u.speaker for u in utterances})
    elapsed = time.perf_counter() - t0
    log.info(
        "%d utterances across %d speakers (%.2fs)",
        len(utterances), len(speaker_labels), elapsed,
    )

    return MergedTranscript(
        audio_path=asr.audio_path,
        audio_duration=asr.audio_duration,
        language=asr.language,
        utterances=utterances,
        speaker_labels=speaker_labels,
        elapsed_seconds=elapsed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge ASR + diarization")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--speakers", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int, default=80)
    args = ap.parse_args()

    asr = read_json(args.transcript, ASRResult)

    diar = None
    if args.speakers and pathlib.Path(args.speakers).exists():
        diar = read_json(args.speakers, DiarizationResult)

    try:
        merged = merge(asr, diar, max_words=args.max_words)
    except Exception as exc:
        log.error("merge failed: %s", exc, exc_info=True)
        return 1

    write_json(args.out, merged)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
