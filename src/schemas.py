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

    # Bounded, for the same reason the citation lists are.
    #
    # The prompt asks for "up to 5 key points". Nothing enforced it, and under
    # one particular condition the model stopped discriminating and began
    # emitting uids in sequence - 474, 476, 481, 482, 483, 495, ... - until it
    # ran out of generation budget mid-array and the stage died. The prompt is
    # a request; maxItems is a constraint.
    #
    # These are per-response caps, not caps on the finished note set: the map
    # stage pools across chunks, so a long proceeding still yields more than
    # this in total. They exist to make a runaway response unrepresentable.
    summary_uids: List[int] = Field(default_factory=list, max_length=12)
    key_point_uids: List[int] = Field(default_factory=list, max_length=20)
    decision_uids: List[int] = Field(default_factory=list, max_length=20)
    action_item_uids: List[int] = Field(default_factory=list, max_length=20)
    keywords: List[str] = Field(default_factory=list, max_length=25)


# A citation that points at 138 transcript lines is not a citation. It is a
# gesture at the whole document, it cannot be checked in one action, and it
# defeats both the per-claim provenance requirement and the one-action
# verification the interface is built around.
#
# It also breaks the run outright. Asked to justify a claim, a model will
# happily enumerate every line it read; on a 139-line chunk that list alone
# consumed the entire generation budget and the response was truncated
# mid-array, failing schema validation and killing the stage.
#
# Bounding it in the schema fixes both at once. Pydantic's max_length becomes
# `maxItems` in the JSON schema handed to the inference engine, so the
# constraint is enforced by the decoding grammar rather than requested in a
# prompt - the same mechanism that makes fabrication impossible in the
# extractive modes, applied to citation volume.
MAX_UIDS_PER_CLAIM = 6


class AbstractiveSection(BaseModel):
    claim: str
    supporting_uids: List[int] = Field(
        default_factory=list, max_length=MAX_UIDS_PER_CLAIM
    )


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


class ChecklistItemId(str, Enum):
    """
    The five checklist items, as a closed vocabulary.

    A free-form `item_id: str` let the judge answer with ids that matched
    nothing. Votes were then tallied against an empty dict, every item scored
    zero agreement, and the calibration layer - correctly, given what it was
    handed - recorded every claim as undecided. The run reported 100%
    abstention and 0% support on output that had never actually been judged.

    An enum becomes a JSON-schema `enum`, which the decoding grammar enforces,
    so an id outside this set is not something the model can emit. Same
    mechanism as the uid-only selection schema: make the failure
    inexpressible rather than detect it afterwards.
    """

    C1_SUPPORTED = "C1_supported"
    C2_NO_NEW_ENTITIES = "C2_no_new_entities"
    C3_ATTRIBUTION = "C3_attribution"
    C4_NO_ADDED_CONCLUSION = "C4_no_added_conclusion"
    C5_NO_NEGATION_FLIP = "C5_no_negation_flip"


CHECKLIST_IDS: List[str] = [m.value for m in ChecklistItemId]


class ChecklistAnswer(BaseModel):
    """One atomic yes/no item in the CheckEval-style judge."""

    item_id: ChecklistItemId
    answer: bool
    evidence_uid: Optional[int] = None


class JudgeResponse(BaseModel):
    """
    Judge output. Deliberately boolean-per-item rather than a 1-5 score:
    pointwise Likert scoring is poorly calibrated and prompt-sensitive
    (survey Section 6.2), whereas atomic binary items aggregate cleanly.

    `answers` is required to hold exactly one entry per checklist item. An
    empty list was previously the cheapest schema-valid response a model could
    produce, and it produced it every time.
    """

    # No default. That is the whole point.
    #
    # With `default_factory=list` this field was absent from the schema's
    # `required` array, and two things followed. The decoding grammar let the
    # model omit the key altogether - which it then did on every single call,
    # because omitting it is the cheapest path the grammar allows. And
    # Pydantic does not validate defaults, so `min_length` never ran: the
    # object validated cleanly with zero answers. A whole corpus run reported
    # 100% abstention on claims no judge had ever looked at.
    #
    # Required + minItems/maxItems makes the empty response unrepresentable at
    # decode time rather than detected afterwards.
    answers: List[ChecklistAnswer] = Field(
        min_length=len(CHECKLIST_IDS),
        max_length=len(CHECKLIST_IDS),
    )
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

    # Substitution-only rates, reported beside the standard ones.
    #
    # A verbatim court transcript records false starts, repetitions and
    # interruptions; an ASR system cleans them up. Scored against each other,
    # deletions swamp everything else - on a 62-minute argument they were 628
    # of 797 total errors - and they are a transcription-convention
    # disagreement, not a recognition failure. Substitutions are where the
    # model actually wrote a different word, which is the error class that
    # matters when the word is a party's name.
    substitution_rate: Optional[float] = None
    legal_substitution_rate: Optional[float] = None
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

    # ---- calibration ----------------------------------------------------
    # Work on reward-model calibration shows that model-based scorers are
    # badly calibrated and should not be consumed as point estimates. When the
    # self-consistency panel splits, recording an abstention is more honest
    # than reporting whichever side won by one vote, and it keeps an unstable
    # verdict out of the support rate rather than silently inside it.
    undecided_claims: int = 0
    abstention_rate: Optional[float] = None
    # Mean fraction of the panel agreeing, across every checklist item judged.
    # 1.0 means the panel never disagreed; 0.5 means it was a coin toss.
    mean_item_agreement: Optional[float] = None
    # Support rate over decided claims only. Reported alongside support_rate,
    # which counts undecided claims as unsupported.
    support_rate_decided: Optional[float] = None


class SpeakerBiasResult(BaseModel):
    """
    Whether whose name is attached to an utterance changes whether it is selected.

    Work on identity-label bias in model decision making shows that a label
    attached to material shifts the decision independently of the material
    itself. In a legal transcript the labels are the speakers, so the question
    is whether the selector treats a sentence differently when it is
    attributed to the bench rather than to counsel. Notes that systematically
    favour one party's statements are a fairness problem specific to this
    domain, and one that no accuracy metric in this harness would surface:
    WER, DER, verbatim rate and judged support are all satisfied by a note set
    that quotes one side of a case perfectly and ignores the other.

    The probe re-runs selection over transcripts whose CONTENT is byte-identical
    and whose SPEAKER LABELS have been altered, so any change in what gets
    selected is attributable to the labels alone.

    Conditions
        control      Labels unchanged. This is the noise floor - sampling
                     variation alone moves the selection, and no shift below
                     this floor is evidence of anything.
        anonymised   Every label collapsed to one neutral token, so speaker
                     identity carries no information at all.
        permuted     The same set of labels, cyclically reassigned to
                     different speakers. Content and label vocabulary both
                     unchanged; only the pairing moves.

    Reading the numbers - both point the same way, and both must clear the
    control run's floor before they mean anything:
        jaccard          selection stability. LOWER = the labels changed what
                         was selected.
        max_share_shift  movement in how the selection is distributed across
                         the people who actually spoke. HIGHER = the selector
                         is following labels rather than content.
    """

    condition: str                    # control | anonymised | permuted | named
    baseline_selected: int
    variant_selected: int
    overlap: int
    jaccard: float
    # Utterances selected under one labelling and not the other.
    only_baseline: List[int] = Field(default_factory=list)
    only_variant: List[int] = Field(default_factory=list)
    # Share of the selection falling to each ORIGINAL speaker - the person who
    # actually said it - under each labelling. Keyed by original speaker in
    # both, so the two are directly comparable: under permutation a
    # content-driven selector keeps choosing the same people and these stay
    # put, while a label-driven one follows the favoured label onto whoever
    # now wears it.
    baseline_share: Dict[str, float] = Field(default_factory=dict)
    variant_share: Dict[str, float] = Field(default_factory=dict)
    # Largest absolute change in any speaker's share of the selection.
    max_share_shift: float = 0.0
    shifted_speaker: Optional[str] = None
    # The relabelling actually applied, so a reviewer can reproduce the run.
    label_map: Dict[str, str] = Field(default_factory=dict)
    # Whether this condition was read by the model in the SAME windows as the
    # baseline. The chunker measures rendered line length against a token
    # budget, so a label of a different width re-splits the transcript and the
    # condition then differs from baseline in two ways at once. A row where
    # this is False is confounded and cannot be called evidence, however large
    # its shift.
    chunking_matched: bool = True
    # Jaccard of the control condition, carried on every row so no row can be
    # read without its floor.
    noise_floor_jaccard: Optional[float] = None
    # True when the shift exceeds what the control run showed. False, or None
    # when no control was run, means the row is not evidence of label bias.
    exceeds_noise_floor: Optional[bool] = None


class ExtractionPRResult(BaseModel):
    """Precision/recall of action-item and decision detection vs gold."""

    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
