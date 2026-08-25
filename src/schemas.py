"""
Stage contracts for the Lausanne pipeline.

Every pipeline stage runs in its own OS process and communicates ONLY through
JSON files on disk. These Pydantic models are the wire format. Nothing is held
in memory across a stage boundary, which is what makes the pipeline survivable
on a 16 GB / 8 GB-VRAM machine (see docs/MEMORY.md).

A second, less obvious benefit: because each contract is an explicit schema,
every stage is independently testable and independently restartable, and the
evaluation harness can consume intermediate artifacts directly.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Stage 1: ASR  ->  work/transcript.json
# --------------------------------------------------------------------------


class Word(BaseModel):
    """A single word with its timing. Needed for speaker attribution."""

    start: float
    end: float
    word: str
    probability: Optional[float] = None


class ASRSegment(BaseModel):
    """One Whisper decode segment."""

    id: int
    start: float
    end: float
    text: str
    words: List[Word] = Field(default_factory=list)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ASRResult(BaseModel):
    """Output of stages/asr.py."""

    audio_path: str
    audio_duration: float
    language: str
    language_probability: float
    model: str
    compute_type: str
    device: str
    segments: List[ASRSegment]
    # Wall-clock seconds the stage took, for the perf table in the paper.
    elapsed_seconds: float
    # Real-time factor: elapsed / audio_duration. Lower is better.
    rtf: float

    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


# --------------------------------------------------------------------------
# Stage 2: Diarization  ->  work/speakers.json
# --------------------------------------------------------------------------


class SpeakerTurn(BaseModel):
    """A contiguous span attributed to one speaker label (SPEAKER_00, ...)."""

    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    """Output of stages/diarize.py."""

    audio_path: str
    model: str
    device: str
    num_speakers_detected: int
    turns: List[SpeakerTurn]
    # pyannote community-1 exposes an "exclusive" variant with non-overlapping
    # spans, which reconciles far more cleanly with word timestamps.
    exclusive_turns: List[SpeakerTurn] = Field(default_factory=list)
    elapsed_seconds: float


# --------------------------------------------------------------------------
# Stage 3: Merge  ->  work/merged.json
# --------------------------------------------------------------------------


class Utterance(BaseModel):
    """
    The unit the rest of the pipeline reasons about: a speaker-attributed,
    time-stamped sentence.

    `uid` is stable and is the ONLY thing the extractive summarizer is allowed
    to emit. That is the mechanism by which fabrication is made structurally
    impossible rather than merely discouraged.
    """

    uid: int
    start: float
    end: float
    speaker: str
    text: str
    # Mean word-level confidence, used to flag low-confidence legal terms.
    confidence: Optional[float] = None

    def timestamp(self) -> str:
        def hhmmss(t: float) -> str:
            t = max(0, int(t))
            return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

        return f"[{hhmmss(self.start)} - {hhmmss(self.end)}]"


class MergedTranscript(BaseModel):
    """Output of stages/merge.py."""

    audio_path: str
    audio_duration: float
    language: str
    utterances: List[Utterance]
    speaker_labels: List[str]
    elapsed_seconds: float

    def by_uid(self) -> Dict[int, Utterance]:
        return {u.uid: u for u in self.utterances}


# --------------------------------------------------------------------------
# Stage 4: Note selection  ->  work/notes.json
# --------------------------------------------------------------------------


class NoteMode(str, Enum):
    """
    The three arms of the system. These are the experimental conditions the
    evaluation compares, and they trade faithfulness against fluency in
    exactly the way the survey's Section 6 predicts.
    """

    # Zero neural models. Frequency scoring + regex cues. Fully deterministic.
    EXTRACTIVE_RULE = "extractive-rule"
    # An LLM chooses which utterances matter, but may ONLY return uids.
    # A validator re-materialises the text from the transcript, so every output
    # sentence is verbatim source. Fluency of an LLM, faithfulness of extraction.
    EXTRACTIVE_LLM = "extractive-llm"
    # Free generation. Fluent prose, and inherits full hallucination risk.
    ABSTRACTIVE = "abstractive"


class Citation(BaseModel):
    """A pointer back into the record. Every claim in the notes carries one."""

    uid: int
    start: float
    end: float
    speaker: str


class NoteItem(BaseModel):
    """
    One line of the notes.

    `text` is what gets rendered. `verbatim` records whether `text` is a
    byte-exact span of the transcript. The renderer marks non-verbatim items
    visually so a reader always knows what is quoted and what is paraphrased.
    """

    text: str
    citations: List[Citation] = Field(default_factory=list)
    verbatim: bool = True
    # Only set for rule-based mode: which regex cue fired.
    cue: Optional[str] = None


class NoteSet(BaseModel):
    """Output of stages/extract.py or stages/summarize.py."""

    mode: NoteMode
    model: Optional[str] = None
    tier: Optional[str] = None
    summary: List[NoteItem] = Field(default_factory=list)
    key_points: List[NoteItem] = Field(default_factory=list)
    decisions: List[NoteItem] = Field(default_factory=list)
    action_items: List[NoteItem] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    # Populated by the faithfulness validator. Non-empty means the model tried
    # to emit something that is not in the transcript.
    validation_warnings: List[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    # Number of LLM calls made (map + reduce), for the cost table.
    llm_calls: int = 0

    def all_items(self) -> List[NoteItem]:
        return (
            self.summary + self.key_points + self.decisions + self.action_items
        )


# --------------------------------------------------------------------------
# LLM structured-output schemas (passed to Ollama's `format` parameter)
# --------------------------------------------------------------------------


class SelectionResponse(BaseModel):
    """
    What the model is allowed to return in EXTRACTIVE_LLM mode.

    Note there is no free-text field anywhere. The model's entire output
    surface is integer ids drawn from a list we gave it. Anything it invents
    fails schema validation or fails the uid-membership check.
    """

    summary_uids: List[int] = Field(default_factory=list)
    key_point_uids: List[int] = Field(default_factory=list)
    decision_uids: List[int] = Field(default_factory=list)
    action_item_uids: List[int] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class AbstractiveSection(BaseModel):
    claim: str
    supporting_uids: List[int] = Field(default_factory=list)


class AbstractiveResponse(BaseModel):
    """
    Free generation, but we still force per-claim citations so the judge has
    something to check each claim against.
    """

    summary: List[AbstractiveSection] = Field(default_factory=list)
    key_points: List[AbstractiveSection] = Field(default_factory=list)
    decisions: List[AbstractiveSection] = Field(default_factory=list)
    action_items: List[AbstractiveSection] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class ChecklistAnswer(BaseModel):
    """One atomic yes/no item in the CheckEval-style judge."""

    item_id: str
    answer: bool
    evidence_uid: Optional[int] = None


class JudgeResponse(BaseModel):
    """
    Judge output. Deliberately boolean-per-item rather than a 1-5 score:
    pointwise Likert scoring is poorly calibrated and prompt-sensitive
    (survey Section 6.2), whereas atomic binary items aggregate cleanly.
    """

    answers: List[ChecklistAnswer] = Field(default_factory=list)
    # Canary: if the judge reports this, evaluated text tried to hijack it.
    injection_detected: bool = False


# --------------------------------------------------------------------------
# Evaluation artifacts
# --------------------------------------------------------------------------


class WERResult(BaseModel):
    reference_words: int
    hypothesis_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    # The number that actually matters for legal work: error rate restricted
    # to tokens that are legal terms, proper names, or citation components.
    legal_term_wer: Optional[float] = None
    legal_term_count: int = 0
    # Worst offenders, for qualitative discussion in the paper.
    top_errors: List[Dict[str, Any]] = Field(default_factory=list)


class DERResult(BaseModel):
    total_speech_seconds: float
    missed_detection: float
    false_alarm: float
    speaker_confusion: float
    der: float
    reference_speakers: int
    hypothesis_speakers: int


class FaithfulnessResult(BaseModel):
    mode: str
    total_claims: int
    # Claims that are a byte-exact span of the transcript. For extractive
    # modes this must be 100% or there is a bug.
    verbatim_claims: int
    verbatim_rate: float
    # Judge-scored support rate for non-verbatim claims.
    judged_claims: int
    supported_claims: int
    support_rate: Optional[float] = None
    injection_attempts_detected: int = 0
    checklist_detail: List[Dict[str, Any]] = Field(default_factory=list)


class ExtractionPRResult(BaseModel):
    """Precision/recall of action-item and decision detection vs gold."""

    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
