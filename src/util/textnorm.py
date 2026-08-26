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

# Latin that is unambiguous on its own. Anything whose surface form is also a
# common English word is NOT here - it belongs in LATIN_PHRASES, matched as a
# phrase.
#
# This distinction is not pedantry. An earlier version of this set contained
# the bare particles "in", "de", "res", "pro", "se", "ex", "quo", "sua",
# "hac", "novo", "nisi", "forum", "ratio", "dicta", "banc", "inter", "alia"
# and "stare", harvested by splitting phrases like "in camera" and "de novo"
# into their words. "in" alone is a top-20 English word. On an hour of real
# argument that single entry flagged hundreds of ordinary tokens as legally
# load-bearing, and the restricted error rate it produced was measuring
# English, not law.
LATIN_LEGAL_TERMS: Set[str] = {
    "certiorari", "certiorarum", "amicus", "amici", "curiae", "decisis",
    "habeas", "mandamus", "subpoena", "tecum", "duces", "judicata",
    "decidendi", "obiter", "dictum", "sponte", "warranto", "meruit",
    "estoppel", "laches", "conveniens", "vires", "voir",
}

# Latin whose individual words are ambiguous but whose sequence is not. Matched
# as consecutive tokens, so "in camera" is legal vocabulary and a bare "in" is
# not.
LATIN_PHRASES: Set[Tuple[str, ...]] = {
    ("de", "novo"), ("in", "camera"), ("ex", "parte"), ("ex", "rel"),
    ("res", "judicata"), ("prima", "facie"), ("sua", "sponte"),
    ("pro", "se"), ("pro", "bono"), ("pro", "hac", "vice"),
    ("inter", "alia"), ("quo", "warranto"), ("quantum", "meruit"),
    ("mens", "rea"), ("actus", "reus"), ("ultra", "vires"),
    ("obiter", "dictum"), ("ratio", "decidendi"), ("stare", "decisis"),
    ("habeas", "corpus"), ("duces", "tecum"), ("nunc", "pro", "tunc"),
    ("locus", "standi"), ("forum", "non", "conveniens"), ("voir", "dire"),
    ("en", "banc"), ("amicus", "curiae"), ("amici", "curiae"),
    ("nisi", "prius"),
}

# Closed-class words that are never legally load-bearing, whatever route
# flagged them. A case name contains "of" and "the"; that does not make every
# "of" in the proceeding a citation component, and it does not make one inside
# the citation worth measuring either - nobody's case turns on a dropped
# article.
NEVER_LEGAL: Set[str] = {
    # Articles, pronouns, prepositions, conjunctions, auxiliaries. None of
    # these is ever "the token a lawyer would cite", which is what this
    # measure exists to isolate. Note "shall", "may" and "not" are absent on
    # purpose - they carry legal force.
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "now", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "well", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "yes", "you",
    "your", "yours", "yourself",
    # Titles. They surround names without being names, and "Mr." is the most
    # frequent capitalised token in an oral argument by a wide margin.
    "mister", "missus", "miss", "doctor", "honorable", "junior", "senior",
    "professor",
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
    "amendment", "amendments", "constitution", "constitutional", "clause",
    "circuit", "precedent", "precedents", "doctrine", "provision",
    "provisions", "opinion", "dissents", "concur", "certiorari",
    "plaintiff's", "licensing", "ordinances", "unconstitutional",
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
    r"(?:§+|\bSection\s+|\bSec\.\s*)\s*\d+[A-Za-z0-9\-.()]*",
    re.IGNORECASE,
)

# Words that end in a period without ending a sentence. Without this the
# proper-name proxy loses every name that follows a title - "Mr. Clement",
# "Justice Alito, Jr." - because the token after "Mr." looks sentence-initial
# and capitalisation there carries no information.
NON_TERMINAL_ABBREV: Set[str] = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "hon.", "jr.", "sr.", "st.",
    "no.", "nos.", "v.", "vs.", "sec.", "art.", "cl.", "para.", "pp.", "p.",
    "e.g.", "i.e.", "etc.", "al.", "u.s.", "u.s.c.", "c.f.r.", "f.supp.",
    "s.ct.", "l.ed.", "inc.", "co.", "corp.", "ltd.", "assn.", "dept.",
}

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


def _token_origins(raw_reference: str) -> Tuple[List[str], List[int]]:
    """
    Normalised tokens plus, for each, the index of the raw word it came from.

    Needed because the flagging below has to be POSITIONAL. Matching by word
    *type* was the second half of the bug: a case name is detected once, its
    words go into a set, and from then on every occurrence of "City", "State",
    "New" or "York" anywhere in the proceeding counts as a citation component.
    On this corpus that inflated the legally-load-bearing set to a quarter of
    all tokens, which is another way of saying it stopped selecting anything.

    Returns ([], []) when per-word normalisation does not reproduce the
    whole-string normalisation exactly - a handful of abbreviations such as
    "et al." span two words - so the caller can fall back rather than
    misalign indices against the token stream the scorer actually uses.
    """
    raw_words = raw_reference.split()
    toks: List[str] = []
    origin: List[int] = []
    for wi, word in enumerate(raw_words):
        for t in tokenize(word):
            toks.append(t)
            origin.append(wi)
    if toks != tokenize(raw_reference):
        return [], []
    return toks, origin


def legal_token_indices(raw_reference: str) -> Set[int]:
    """
    Indices (into the normalised token stream) of tokens that are legally
    load-bearing. Used to compute a restricted WER.

    Coverage:
      - unambiguous Latin and procedural vocabulary (lexicon match)
      - multi-word Latin phrases, matched as sequences
      - tokens *positionally inside* a detected citation / rule / section /
        case-name span
      - tokens whose raw word is capitalised mid-sentence, a cheap and
        surprisingly effective proper-name proxy

    Closed-class function words are excluded from every route. The measure
    exists to isolate the tokens a practitioner would cite; an article that
    happens to sit inside a case name is not one of them.
    """
    normed_tokens = tokenize(raw_reference)
    indices: Set[int] = set()

    # ---- 1. Lexicon (type match is correct here: it is a vocabulary).
    for i, tok in enumerate(normed_tokens):
        if tok in LATIN_LEGAL_TERMS or tok in PROCEDURAL_TERMS:
            indices.add(i)
        if tok in REPORTERS:
            indices.add(i)

    # ---- 2. Multi-word Latin, matched as consecutive tokens.
    max_len = max((len(p) for p in LATIN_PHRASES), default=0)
    for i in range(len(normed_tokens)):
        for n in range(2, max_len + 1):
            # The bound matters. A slice past the end silently shortens, so
            # normed_tokens[76:79] on a 78-token stream returns two tokens and
            # compares equal to a two-word phrase - then range(76, 79) writes
            # an index one past the array. It only fires when a Latin phrase
            # falls at the very end of a transcript, which is exactly where
            # "stare decisis" tends to land.
            if i + n > len(normed_tokens):
                break
            if tuple(normed_tokens[i:i + n]) in LATIN_PHRASES:
                indices.update(range(i, i + n))

    toks, origin = _token_origins(raw_reference)
    raw_words = raw_reference.split()

    if toks:
        # ---- 3. Citation / rule / section / case-name spans, by position.
        # Character offsets are mapped to raw-word indices, and only the
        # tokens produced by those specific words are flagged.
        word_start: List[int] = []
        cursor = 0
        for w in raw_words:
            cursor = raw_reference.index(w, cursor)
            word_start.append(cursor)
            cursor += len(w)

        span_word_indices: Set[int] = set()
        for pat, _kind in (
            (CITATION_PATTERN, "citation"), (RULE_PATTERN, "rule"),
            (SECTION_PATTERN, "section"), (CASE_NAME_PATTERN, "case_name"),
        ):
            for m in pat.finditer(raw_reference):
                lo, hi = m.start(), m.end()
                for wi, ws in enumerate(word_start):
                    if ws < hi and ws + len(raw_words[wi]) > lo:
                        span_word_indices.add(wi)

        # ---- 4. Proper-name proxy, also by position.
        #
        # Capitalisation mid-sentence is a decent name signal and a noisy one.
        # An oral argument is a sequence of interruptions, and a turn that ends
        # without terminal punctuation - which is most of them - makes the next
        # turn's first word look mid-sentence. That flagged "Well", "Am",
        # "Here" and "Second" as proper nouns on this corpus.
        #
        # Filtered by a document-internal test: a real name is essentially
        # never written in lower case anywhere in the same transcript, whereas
        # every one of those false positives is. It costs nothing and needs no
        # external word list.
        lowercase_types = {
            normalize(w.strip(".,;:!?\"'()[]"))
            for w in raw_words
            if w[:1].islower()
        }
        proper_word_indices: Set[int] = set()
        sentence_start = True
        for wi, w in enumerate(raw_words):
            core = w.strip(".,;:!?\"'()[]")
            if (core and len(core) >= 3 and core[0].isupper()
                    and not sentence_start and not core.isupper()
                    and normalize(core) not in lowercase_types):
                proper_word_indices.add(wi)
            # "--" is how a verbatim court transcript marks an interruption or
            # a false start, and an oral argument is full of them. The word
            # after one is turn-initial, so its capitalisation says nothing.
            sentence_start = (
                w.endswith("--")
                or (w.endswith((".", "!", "?"))
                    and w.lower().strip(",;:\"')]") not in NON_TERMINAL_ABBREV)
            )

        marked_words = span_word_indices | proper_word_indices
        for i, wi in enumerate(origin):
            if wi in marked_words:
                indices.add(i)
    else:
        # Fallback: per-word normalisation diverged, so positions cannot be
        # trusted. Type matching is over-inclusive, but with function words
        # excluded below it stays far tighter than the original.
        span_tokens: Set[str] = set()
        for span_text, _kind in extract_legal_spans(raw_reference):
            span_tokens.update(tokenize(span_text))
        for i, tok in enumerate(normed_tokens):
            if tok in span_tokens:
                indices.add(i)

    # ---- 5. Function words are never legally load-bearing.
    return {i for i in indices if normed_tokens[i] not in NEVER_LEGAL}
