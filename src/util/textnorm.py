"""
Text normalisation for scoring, plus legal-token identification.

Two jobs:

1. Normalise reference and hypothesis so WER measures transcription error and
   not formatting disagreement ("U.S.C." vs "USC", "Section 5" vs "sec. 5").

2. Identify which reference tokens are *legally load-bearing* - Latin terms,
   citation components, statutory references, party and judge names. Overall
   WER hides the errors that actually matter: a system can score 6% WER while
   mangling every case citation, and for legal work that system is useless.
   Section 8.4 of the survey flags exactly this propagation risk; this module
   is what lets us measure it instead of asserting it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set, Tuple


# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------

LATIN_LEGAL_TERMS: Set[str] = {
    "certiorari", "certiorarum", "amicus", "amici", "curiae", "stare",
    "decisis", "voir", "dire", "habeas", "corpus", "mandamus", "subpoena",
    "duces", "tecum", "prima", "facie", "res", "judicata", "obiter", "dictum",
    "dicta", "ratio", "decidendi", "mens", "rea", "actus", "reus", "ultra",
    "vires", "inter", "alia", "in", "camera", "banc", "sua", "sponte",
    "pro", "se", "bono", "hac", "vice", "de", "novo", "nisi", "quo",
    "warranto", "quantum", "meruit", "estoppel", "laches", "certiorari",
    "forum", "conveniens", "locus", "standi", "ex", "parte", "nunc", "tunc",
}

PROCEDURAL_TERMS: Set[str] = {
    "petitioner", "petitioners", "respondent", "respondents", "appellant",
    "appellants", "appellee", "appellees", "plaintiff", "plaintiffs",
    "defendant", "defendants", "movant", "affiant", "deponent", "intervenor",
    "amicus", "counsel", "solicitor", "attorney", "magistrate", "justice",
    "justices", "judge", "chancellor", "arbitrator", "juror", "jurors",
    "indictment", "arraignment", "deposition", "interrogatory",
    "interrogatories", "injunction", "remand", "remanded", "affirmed",
    "reversed", "vacated", "overruled", "sustained", "dissent", "dissenting",
    "concurring", "concurrence", "holding", "dictum", "statute", "statutory",
    "ordinance", "regulation", "codified", "preemption", "preempted",
    "severability", "justiciable", "justiciability", "mootness", "standing",
    "jurisdiction", "jurisdictional", "venue", "discovery", "privilege",
    "privileged", "hearsay", "objection", "overrule", "stipulation",
}

# Reporter abbreviations that appear inside citations.
REPORTERS: Set[str] = {
    "usc", "us", "cfr", "fsupp", "f2d", "f3d", "sct", "led", "ny", "cal",
    "scc", "air", "scr", "wlr", "qb", "ac", "ewca", "ewhc", "ukhl", "uksc",
}


# --------------------------------------------------------------------------
# Number expansion
# --------------------------------------------------------------------------

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    """Spell an integer. ASR emits words; references often use digits."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        rest = n % 10
        return _TENS[n // 10] + (" " + _UNITS[rest] if rest else "")
    if n < 1000:
        rest = n % 100
        return _UNITS[n // 100] + " hundred" + (
            " " + _int_to_words(rest) if rest else ""
        )
    for div, name in ((1_000_000_000, "billion"), (1_000_000, "million"),
                      (1_000, "thousand")):
        if n >= div:
            rest = n % div
            return _int_to_words(n // div) + f" {name}" + (
                " " + _int_to_words(rest) if rest else ""
            )
    return str(n)


_ORDINAL_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th)$")

_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    "'m": " am", "let's": "let us", "it's": "it is", "that's": "that is",
    "there's": "there is", "what's": "what is", "he's": "he is",
    "she's": "she is", "who's": "who is", "here's": "here is",
}

_ABBREV = {
    "u.s.c.": "usc", "u.s.": "us", "c.f.r.": "cfr", "f.supp.": "fsupp",
    "f.2d": "f2d", "f.3d": "f3d", "s.ct.": "sct", "l.ed.": "led",
    "sec.": "section", "secs.": "sections", "§": "section",
    "no.": "number", "nos.": "numbers", "v.": "versus", "vs.": "versus",
    "mr.": "mister", "mrs.": "missus", "dr.": "doctor", "hon.": "honorable",
    "jr.": "junior", "sr.": "senior", "et al.": "and others",
    "i.e.": "that is", "e.g.": "for example", "etc.": "etcetera",
    "art.": "article", "cl.": "clause", "amend.": "amendment",
    "para.": "paragraph", "pp.": "pages", "p.": "page",
}


def normalize(text: str) -> str:
    """
    Canonical form for scoring.

    Order matters: multi-character abbreviations are substituted before
    punctuation is stripped, otherwise "U.S.C." becomes "u s c".
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Multi-word / dotted abbreviations first.
    for src, dst in sorted(_ABBREV.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(src, f" {dst} ")

    for src, dst in _CONTRACTIONS.items():
        text = text.replace(src, dst)

    # Whisper emits filler; references usually do not. Removing it from both
    # sides is standard practice and avoids penalising a correct transcript.
    text = re.sub(r"\b(uh|um|erm|mm|hmm|uh-huh|mm-hmm)\b", " ", text)

    # Strip everything that is not a letter, digit, space, or intra-word hyphen.
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)

    tokens = []
    for tok in text.split():
        m = _ORDINAL_SUFFIX.match(tok)
        if m:
            tokens.append(_int_to_words(int(m.group(1))) + "th")
            continue
        if tok.isdigit():
            tokens.append(_int_to_words(int(tok)))
            continue
        tokens.append(tok)

    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


# --------------------------------------------------------------------------
# Legal token identification
# --------------------------------------------------------------------------

# "384 U.S. 436", "42 U.S.C. § 1983", "Rule 12(b)(6)"
CITATION_PATTERN = re.compile(
    r"\b\d+\s+(?:U\.?\s?S\.?(?:C\.?)?|F\.?\s?(?:2d|3d|Supp\.?)|S\.?\s?Ct\.?|"
    r"L\.?\s?Ed\.?|C\.?F\.?R\.?|SCC|AIR|SCR)\s*(?:§\s*)?\d+",
    re.IGNORECASE,
)

RULE_PATTERN = re.compile(r"\bRule\s+\d+(?:\([a-z0-9]\))*", re.IGNORECASE)

SECTION_PATTERN = re.compile(
    r"(?:§+|\bSection\s+|\bSec\.\s*)\s*\d+[A-Za-z0-9\-.()]*"
)

CASE_NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)*)\s+v\.?\s+"
    r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)*)"
)


def extract_legal_spans(raw_text: str) -> List[Tuple[str, str]]:
    """
    Find legally load-bearing spans in *unnormalised* text.

    Returns (span_text, kind) pairs. Runs before normalisation because
    capitalisation and punctuation are the signal for names and citations.
    """
    spans: List[Tuple[str, str]] = []
    for pat, kind in (
        (CITATION_PATTERN, "citation"),
        (RULE_PATTERN, "rule"),
        (SECTION_PATTERN, "section"),
        (CASE_NAME_PATTERN, "case_name"),
    ):
        for m in pat.finditer(raw_text):
            spans.append((m.group(0), kind))
    return spans


def legal_token_indices(raw_reference: str) -> Set[int]:
    """
    Indices (into the normalised token stream) of tokens that are legally
    load-bearing. Used to compute a restricted WER.

    Coverage:
      - Latin and procedural vocabulary (lexicon match)
      - tokens inside a detected citation / rule / section / case-name span
      - capitalised tokens in the raw text that are not sentence-initial,
        which is a cheap and surprisingly effective proper-name proxy
    """
    normed_tokens = tokenize(raw_reference)
    indices: Set[int] = set()

    for i, tok in enumerate(normed_tokens):
        if tok in LATIN_LEGAL_TERMS or tok in PROCEDURAL_TERMS:
            indices.add(i)
        if tok in REPORTERS:
            indices.add(i)

    # Tokens contributed by legal spans.
    span_tokens: Set[str] = set()
    for span_text, _kind in extract_legal_spans(raw_reference):
        span_tokens.update(tokenize(span_text))
    for i, tok in enumerate(normed_tokens):
        if tok in span_tokens:
            indices.add(i)

    # Proper-name proxy: mid-sentence capitalisation in the raw text.
    raw_words = raw_reference.split()
    proper: Set[str] = set()
    sentence_start = True
    for w in raw_words:
        core = w.strip(".,;:!?\"'()[]")
        if core and core[0].isupper() and not sentence_start and not core.isupper():
            proper.add(normalize(core))
        sentence_start = w.endswith((".", "!", "?"))
    proper.discard("")
    for i, tok in enumerate(normed_tokens):
        if tok in proper:
            indices.add(i)

    return indices
