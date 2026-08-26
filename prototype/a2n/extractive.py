"""
Extractive, rule-based note generation.

This module implements the deterministic note-generation mode described in the
paper. It uses NO neural model and NO external API. Because every output
sentence is copied verbatim from the transcript, this mode cannot fabricate
content -- the key faithfulness property discussed in the paper's evaluation
section.

All functions here are pure Python and depend only on the standard library,
so they can be unit-tested without Whisper, torch, or transformers installed.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Vocabulary / pattern resources
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "at", "for", "from", "with", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "them", "his", "her", "our", "their", "my", "your",
    "me", "us", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "not", "so", "than", "too", "very", "just",
    "about", "into", "over", "after", "before", "up", "down", "out", "off", "again",
    "further", "once", "here", "there", "when", "where", "why", "how", "what",
    "which", "who", "whom",
}

# Cues that typically indicate a task / obligation / follow-up.
ACTION_PATTERNS = [
    r"\bneed to\b", r"\bneeds to\b", r"\bshould\b", r"\bmust\b", r"\bwill\b",
    r"\btodo\b", r"\bto do\b", r"\baction item\b", r"\bfollow up\b",
    r"\bfollow-up\b", r"\bdeadline\b", r"\bassign(?:ed)?\b", r"\bsubmit\b",
    r"\bcomplete\b", r"\bfile\b", r"\bdraft\b", r"\bnotify\b",
]

# Cues that typically indicate a decision / ruling / agreed term.
DECISION_PATTERNS = [
    r"\bdecided\b", r"\bagreed\b", r"\bfinalized\b", r"\bconfirmed\b",
    r"\bapproved\b", r"\bselected\b", r"\bresolved\b", r"\bgranted\b",
    r"\bdenied\b", r"\bordered\b", r"\bheld\b", r"\bruled\b",
]


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and tidy spacing before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    for bad, good in ((" ,", ","), (" .", "."), (" ?", "?"), (" !", "!")):
        text = text.replace(bad, good)
    return text


def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def build_frequency_table(sentences: List[str]) -> Counter:
    return Counter(tokenize(" ".join(sentences)))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def summarize_sentences(sentences: List[str], max_sentences: int = 5) -> List[str]:
    """Frequency-based extractive summary. Returns sentences in original order."""
    if len(sentences) <= max_sentences:
        return sentences

    freq = build_frequency_table(sentences)
    scored = []
    for idx, sentence in enumerate(sentences):
        words = tokenize(sentence)
        if not words:
            continue
        score = sum(freq[w] for w in words) / len(words)
        scored.append((score, idx, sentence))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    top = sorted(top, key=lambda x: x[1])  # restore original order
    return [item[2] for item in top]


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    counts = Counter(tokenize(text))
    return [word for word, _ in counts.most_common(top_k)]


def extract_matching_sentences(sentences: List[str], patterns: List[str]) -> List[str]:
    results = []
    for sentence in sentences:
        if any(re.search(p, sentence, flags=re.IGNORECASE) for p in patterns):
            results.append(sentence)
    return unique_preserve_order(results)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_timestamped_transcript(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} - {end}] {clean_text(seg['text'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notes assembly
# ---------------------------------------------------------------------------

def generate_markdown_notes(
    title: str,
    transcript: str,
    segments: List[Dict[str, Any]],
    whisper_model: str,
    audio_file: str,
) -> str:
    """Assemble structured Markdown notes from a transcript (extractive mode)."""
    sentences = split_sentences(transcript)

    summary = summarize_sentences(sentences, max_sentences=5)
    key_points = summarize_sentences(sentences, max_sentences=7)
    action_items = extract_matching_sentences(sentences, ACTION_PATTERNS) or [
        "No explicit action items detected."
    ]
    decisions = extract_matching_sentences(sentences, DECISION_PATTERNS) or [
        "No explicit decisions detected."
    ]
    keywords = extract_keywords(transcript, top_k=10)

    summary_text = " ".join(summary) if summary else "No summary could be generated."

    md = f"# {title}\n\n"
    md += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md += f"**Audio file:** {audio_file}\n"
    md += f"**Transcription model (local):** Whisper-{whisper_model}\n"
    md += "**Note mode:** extractive / rule-based (no LLM, no API)\n\n"

    md += "## 1. Summary\n" + summary_text + "\n\n"

    md += "## 2. Key Points\n"
    md += "".join(f"- {p}\n" for p in key_points) + "\n"

    md += "## 3. Decisions\n"
    md += "".join(f"- {d}\n" for d in decisions) + "\n"

    md += "## 4. Action Items\n"
    md += "".join(f"- {a}\n" for a in action_items) + "\n"

    md += "## 5. Keywords\n"
    md += (", ".join(keywords) if keywords else "No keywords detected.") + "\n\n"

    md += "## 6. Time-stamped Transcript\n"
    if segments:
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            md += f"- **[{start} - {end}]** {clean_text(seg['text'])}\n"
    else:
        md += transcript + "\n"

    return md
