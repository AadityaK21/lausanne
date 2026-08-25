"""
Configuration and the two-tier model abstraction.

The survey (Section 8.2) specifies Llama-3.3-70B-Instruct for the generative
mode. A 70B model at 4-bit is ~40 GB of weights; the development laptop has
8 GB of VRAM and 16 GB of system RAM. Rather than quietly substituting a
smaller model and leaving the paper's claim unsupported, the pipeline makes
the tier an explicit, recorded property of every run.

    local   -> 8B-class, 4-bit, runs on the RTX 4060. This is the tier the
               privacy argument is actually about: it runs on a machine a
               solo practitioner might own.
    hosted  -> 70B-class, 4-bit, on a rented single GPU (L40S/A100 48-80 GB).
               Used ONLY on public court audio, so no privileged material
               ever touches rented hardware and the confidentiality claim in
               Section 8.3 survives intact.

Every artifact records which tier produced it. The evaluation reports them as
separate rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in requirements.txt
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Hardware budget
# --------------------------------------------------------------------------


@dataclass
class MemoryBudget:
    """
    Declared VRAM ceilings per stage, in MB.

    These are asserted at runtime by util/gpu.py. If a stage would exceed its
    ceiling the run fails loudly rather than thrashing into shared-memory
    fallback, which on Windows silently degrades to ~50x slower instead of
    raising OOM.
    """

    total_vram_mb: int = 8188          # RTX 4060 Laptop, minus nothing yet
    reserved_display_mb: int = 900     # Windows WDDM desktop composition
    asr_ceiling_mb: int = 3600
    diarize_ceiling_mb: int = 2200
    llm_ceiling_mb: int = 6200

    @property
    def usable_vram_mb(self) -> int:
        return self.total_vram_mb - self.reserved_display_mb

    def check(self) -> list[str]:
        """Return a list of budget violations, empty if the plan is coherent."""
        problems = []
        for name, ceiling in (
            ("asr", self.asr_ceiling_mb),
            ("diarize", self.diarize_ceiling_mb),
            ("llm", self.llm_ceiling_mb),
        ):
            if ceiling > self.usable_vram_mb:
                problems.append(
                    f"{name} ceiling {ceiling} MB exceeds usable VRAM "
                    f"{self.usable_vram_mb} MB"
                )
        return problems


# --------------------------------------------------------------------------
# Stage configs
# --------------------------------------------------------------------------


@dataclass
class ASRConfig:
    # large-v3 is the accuracy ceiling; "turbo" (large-v3-turbo) is ~4x faster
    # with a small WER cost and is the sane default on 8 GB.
    model: str = "large-v3-turbo"
    device: str = "cuda"
    # int8_float16 keeps the encoder in fp16 and quantises the rest. On the
    # 4060 this is ~2.5 GB resident vs ~4.5 GB for float16.
    compute_type: str = "int8_float16"
    beam_size: int = 5
    # Zero temperature for reproducibility, per Section 8.1.
    temperature: float = 0.0
    batch_size: int = 8
    vad_filter: bool = True
    word_timestamps: bool = True
    language: Optional[str] = "en"
    condition_on_previous_text: bool = False  # prevents runaway repetition
    # Legal proper nouns and Latin terms Whisper reliably mangles. Supplied as
    # an initial prompt to bias decoding.
    initial_prompt: Optional[str] = (
        "This is a transcript of a court proceeding. Terms that may appear "
        "include: certiorari, amicus curiae, stare decisis, voir dire, "
        "habeas corpus, en banc, in camera, sua sponte, respondent, "
        "petitioner, appellant, appellee, U.S.C., C.F.R., plaintiff."
    )


@dataclass
class DiarizeConfig:
    # community-1 supersedes 3.1: lower DER across AMI/AliMeeting/AISHELL and,
    # more usefully here, it exposes an exclusive (non-overlapping) variant
    # that reconciles cleanly with word timestamps.
    model: str = "pyannote/speaker-diarization-community-1"
    fallback_model: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    use_exclusive: bool = True


# A tier that has not been pointed at a real endpoint. Deliberately a
# non-local, unresolvable address in the reserved .invalid TLD rather than a
# blank or a localhost default. A blank would read as "in-process" to the
# confidentiality guard, and a localhost default is worse still: selecting the
# 70B tier would then quietly run the 8B local model and label the output as
# 70B, putting a false row in a benchmark table.
UNCONFIGURED_HOST = "https://configure-me.invalid"


@dataclass
class LLMTier:
    name: str
    model: str
    context_tokens: int
    # Tokens of transcript per map call, leaving room for instructions + output.
    chunk_tokens: int
    chunk_overlap_tokens: int
    host: str = "http://localhost:11434"
    keep_alive: str = "0"          # unload immediately; we cannot afford residency
    temperature: float = 0.0
    num_predict: int = 1536
    # llama.cpp KV cache quantisation. q8_0 halves KV memory at negligible cost.
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    notes: str = ""

    @property
    def is_configured(self) -> bool:
        return self.host != UNCONFIGURED_HOST


DEFAULT_TIERS: Dict[str, LLMTier] = {
    "local": LLMTier(
        name="local",
        # ~5.2 GB at Q4_K_M, 40K context. Leaves headroom under the 6.2 GB
        # ceiling once KV cache is quantised.
        model="qwen3:8b",
        context_tokens=8192,
        chunk_tokens=5000,
        chunk_overlap_tokens=500,
        notes=(
            "Runs on RTX 4060 8 GB. This is the configuration the privacy "
            "argument is about. Alternatives that also fit: llama3.1:8b, "
            "mistral:7b, gemma3:4b. Anything 12B+ at Q4 collides with the "
            "display reservation."
        ),
    ),
    "hosted": LLMTier(
        name="hosted",
        model="llama3.3:70b",
        # No usable default. This tier must be pointed at a real endpoint
        # before it will run; see the note on UNCONFIGURED_HOST above.
        host=UNCONFIGURED_HOST,
        context_tokens=16384,
        chunk_tokens=12000,
        chunk_overlap_tokens=800,
        num_predict=2048,
        notes=(
            "Q4_K_M ~43 GB. Requires a rented 48-80 GB GPU (L40S / A100). "
            "PUBLIC COURT AUDIO ONLY - never point this tier at privileged "
            "material, or the Section 8.3 confidentiality claim is void."
        ),
    ),
}


@dataclass
class JudgeConfig:
    """
    LLM-as-a-Judge settings.

    Deliberately defaults to a *different* model family than the generator.
    A model judging its own output inflates support rates; the survey flags
    this in Section 6.2 and it is cheap to avoid.
    """

    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    keep_alive: str = "0"
    # Present the evaluated text inside a hard delimiter and instruct the
    # judge that content inside it is data, never instruction.
    delimiter: str = "<<<EVALUATED_CONTENT>>>"
    # A phrase the judge is told to report if the evaluated text tries to
    # instruct it. Lets us measure injection susceptibility instead of
    # assuming it away.
    injection_canary: str = "INJECTION_ATTEMPT_OBSERVED"
    # Run each checklist item N times and take the majority, to damp the
    # prompt-sensitivity that pointwise judging is known for.
    self_consistency_runs: int = 3


@dataclass
class ExtractConfig:
    """Rule-based extractive mode. No neural model touches this path."""

    summary_sentences: int = 8
    key_points: int = 10
    keywords: int = 15
    min_sentence_words: int = 6
    # Cue sets from Section 8.2, expanded with courtroom-specific language.
    action_cues: tuple = (
        r"\bneed(?:s)? to\b", r"\bmust\b", r"\bshall\b", r"\bdeadline\b",
        r"\bfollow(?:[- ])?up\b", r"\bassign(?:ed|s)?\b", r"\bsubmit\b",
        r"\bfile(?:d|s)? (?:a|an|the)\b", r"\bserve(?:d)? (?:a|an|the)\b",
        r"\bby (?:no later than|the end of)\b", r"\bordered to\b",
        r"\bdirected to\b", r"\bwithin \d+ days\b", r"\bbrief(?:ing)? "
        r"(?:schedule|due)\b", r"\bproduce\b", r"\bdisclose\b",
    )
    decision_cues: tuple = (
        r"\bdecided\b", r"\bagreed\b", r"\bfinali[sz]ed\b", r"\bconfirmed\b",
        r"\bapproved\b", r"\bgranted\b", r"\bdenied\b", r"\bsustained\b",
        r"\boverruled\b", r"\bwe hold\b", r"\bthe court (?:finds|holds|rules)\b",
        r"\bso ordered\b", r"\bmotion is\b", r"\bjudgment (?:is|for)\b",
        r"\baffirmed\b", r"\breversed\b", r"\bremanded\b", r"\bstipulat",
    )


@dataclass
class Config:
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    memory: MemoryBudget = field(default_factory=MemoryBudget)
    tiers: Dict[str, LLMTier] = field(
        default_factory=lambda: {k: v for k, v in DEFAULT_TIERS.items()}
    )
    active_tier: str = "local"
    work_dir: str = "work"
    output_dir: str = "output"

    def tier(self, name: Optional[str] = None) -> LLMTier:
        key = name or self.active_tier
        if key not in self.tiers:
            raise KeyError(
                f"Unknown tier {key!r}. Known tiers: {sorted(self.tiers)}"
            )
        return self.tiers[key]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _merge(base: Any, override: Dict[str, Any]) -> None:
    """Shallow-merge a dict of overrides onto a dataclass instance."""
    for key, value in override.items():
        if not hasattr(base, key):
            continue
        current = getattr(base, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(base, key, value)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load config.yaml if present, else return defaults.

    Environment overrides (useful for the hosted tier, where you do not want
    to edit files on a rented box):
        LAUSANNE_TIER, OLLAMA_HOST, LAUSANNE_ASR_MODEL
    """
    cfg = Config()

    candidate = Path(path) if path else REPO_ROOT / "config.yaml"
    if candidate.exists() and yaml is not None:
        with open(candidate, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        tiers_raw = raw.pop("tiers", None)
        if tiers_raw:
            for tname, tvals in tiers_raw.items():
                if tname in cfg.tiers:
                    _merge(cfg.tiers[tname], tvals)
                else:
                    cfg.tiers[tname] = LLMTier(name=tname, **tvals)
        _merge(cfg, raw)

    if os.environ.get("LAUSANNE_TIER"):
        cfg.active_tier = os.environ["LAUSANNE_TIER"]
    if os.environ.get("LAUSANNE_ASR_MODEL"):
        cfg.asr.model = os.environ["LAUSANNE_ASR_MODEL"]
    if os.environ.get("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"]
        if not host.startswith("http"):
            host = f"http://{host}"
        for t in cfg.tiers.values():
            t.host = host
        cfg.judge.host = host

    return cfg
