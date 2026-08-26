"""
Tests for the extractive (rule-based) note logic.

These run WITHOUT whisper/torch/transformers, so they verify the core
pipeline on any machine. Run with:  pytest
"""

from a2n import extractive


SAMPLE_TRANSCRIPT = (
    "The client met with counsel on Monday to review the contract. "
    "We decided to proceed with the merger after due diligence. "
    "The team needs to submit the disclosure documents before the deadline. "
    "The court ruled that the prior agreement was valid. "
    "We must follow up with the regulator next week. "
    "The parties agreed on the final settlement amount."
)

SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 4.5, "text": "The client met with counsel on Monday."},
    {"start": 4.5, "end": 9.0, "text": "We decided to proceed with the merger."},
    {"start": 9.0, "end": 14.0, "text": "The team needs to submit the documents."},
]


def test_clean_text_normalizes_spacing():
    assert extractive.clean_text("hello ,  world .") == "hello, world."


def test_split_sentences_counts():
    sents = extractive.split_sentences(SAMPLE_TRANSCRIPT)
    assert len(sents) == 6


def test_format_timestamp():
    assert extractive.format_timestamp(3661) == "01:01:01"
    assert extractive.format_timestamp(0) == "00:00:00"


def test_action_items_detected():
    sents = extractive.split_sentences(SAMPLE_TRANSCRIPT)
    actions = extractive.extract_matching_sentences(sents, extractive.ACTION_PATTERNS)
    assert any("submit" in a.lower() for a in actions)
    assert any("follow up" in a.lower() for a in actions)


def test_decisions_detected():
    sents = extractive.split_sentences(SAMPLE_TRANSCRIPT)
    decisions = extractive.extract_matching_sentences(sents, extractive.DECISION_PATTERNS)
    assert any("decided" in d.lower() for d in decisions)
    assert any("ruled" in d.lower() for d in decisions)
    assert any("agreed" in d.lower() for d in decisions)


def test_keywords_are_content_words():
    kws = extractive.extract_keywords(SAMPLE_TRANSCRIPT, top_k=10)
    assert "the" not in kws  # stopword removed
    assert all(len(k) > 2 for k in kws)


def test_markdown_notes_has_all_sections():
    md = extractive.generate_markdown_notes(
        title="Test Notes",
        transcript=SAMPLE_TRANSCRIPT,
        segments=SAMPLE_SEGMENTS,
        whisper_model="base",
        audio_file="sample.wav",
    )
    for header in [
        "# Test Notes",
        "## 1. Summary",
        "## 2. Key Points",
        "## 3. Decisions",
        "## 4. Action Items",
        "## 5. Keywords",
        "## 6. Time-stamped Transcript",
    ]:
        assert header in md
    assert "[00:00:00 - 00:00:04]" in md  # timestamp formatting in transcript
