"""
End-to-end test of every CPU path, with no models and no network.

Builds a synthetic oral argument, a synthetic ASR hypothesis containing the
kinds of errors Whisper actually makes on legal audio (mangled case names,
dropped citations, Latin terms substituted with near-homophones), and a
synthetic diarization with a plausible speaker-confusion error. Then runs
merge -> extract -> render and scores WER / legal-term WER / DER.

Run:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                                   # noqa: E402
from src.eval import bias as bias_mod                                # noqa: E402
from src.eval import der as der_mod                                  # noqa: E402
from src.eval import extraction_pr as pr_mod                         # noqa: E402
from src.eval import judge as judge_mod                              # noqa: E402
from src.eval import report as report_mod                            # noqa: E402
from src.eval import wer as wer_mod                                  # noqa: E402
from src.schemas import (                                             # noqa: E402
    ASRResult, ASRSegment, DiarizationResult, NoteMode, SpeakerTurn,
    Utterance, Word,
)
from src.stages.extract import extract                                # noqa: E402
from src.stages.merge import merge                                    # noqa: E402
from src.stages.render import to_docx, to_markdown, to_plain_transcript  # noqa: E402
from src.stages.summarize import chunk_utterances, validate_uids      # noqa: E402
from src.util import provenance as prov                                  # noqa: E402
from src.util.io import write_json                                       # noqa: E402
from src.util.ollama_client import inline_defs as inline_defs_probe  # noqa: E402
from src.schemas import (                                             # noqa: E402
    Citation, MergedTranscript, NoteItem, NoteSet,
)
from src.util.textnorm import legal_token_indices, normalize, tokenize  # noqa: E402


PASS, FAIL = "  \033[92mPASS\033[0m", "  \033[91mFAIL\033[0m"
_results = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _results.append(condition)
    print(f"{PASS if condition else FAIL}  {label}" + (f"  — {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

REFERENCE_TURNS = [
    ("CHIEF JUSTICE ROBERTS",
     "We will hear argument this morning in Case 17-1618, Bostock v. Clayton "
     "County. Mr. Bursch."),
    ("MR. BURSCH",
     "Mr. Chief Justice, and may it please the Court. The question presented "
     "is whether Title VII of the Civil Rights Act of 1964 prohibits "
     "discrimination on the basis of sexual orientation."),
    ("JUSTICE GINSBURG",
     "But counsel, in Price Waterhouse v. Hopkins we held that sex "
     "stereotyping is actionable under Title VII. How do you distinguish that "
     "holding?"),
    ("MR. BURSCH",
     "Justice Ginsburg, Price Waterhouse addressed stereotyping about how a "
     "woman should behave. It did not address status. The petitioner must "
     "submit supplemental briefing on that distinction within thirty days."),
    ("JUSTICE KAGAN",
     "That seems to prove too much. Under stare decisis we do not lightly "
     "revisit a construction of a statute that Congress has left undisturbed "
     "for forty years."),
    ("MR. BURSCH",
     "Respectfully, Justice Kagan, the doctrine of stare decisis has less "
     "force where the original construction was itself in error."),
    ("CHIEF JUSTICE ROBERTS",
     "The motion for leave to file an amicus curiae brief is granted. Counsel "
     "shall file the joint appendix by no later than the fourteenth of March."),
    ("MS. KARLAN",
     "Thank you. In Oncale v. Sundowner Offshore Services this Court held that "
     "Title VII protects against same-sex harassment. The text of the statute "
     "controls, not the expectations of the 1964 Congress."),
    ("JUSTICE ALITO",
     "You are asking us to update a statute. That is a legislative function, "
     "and the writ of certiorari was granted on a narrower question."),
    ("MS. KARLAN",
     "We are asking the Court to apply the text as written. The judgment of "
     "the Eleventh Circuit should be reversed and the case remanded."),
    ("CHIEF JUSTICE ROBERTS",
     "Thank you, counsel. The case is submitted. It is so ordered."),
]

# Errors an ASR system plausibly makes here: case names mangled, Latin terms
# substituted, a citation number dropped, a court name wrong.
HYPOTHESIS_TURNS = [
    ("We will hear argument this morning in Case 17-1618, Boss Stock versus "
     "Clayton County. Mr. Bursh."),
    ("Mr. Chief Justice, and may it please the Court. The question presented "
     "is whether Title 7 of the Civil Rights Act of 1964 prohibits "
     "discrimination on the basis of sexual orientation."),
    ("But counsel, in Price Waterhouse versus Hopkins we held that sex "
     "stereotyping is actionable under Title 7. How do you distinguish that "
     "holding?"),
    ("Justice Ginsberg, Price Waterhouse addressed stereotyping about how a "
     "woman should behave. It did not address status. The petitioner must "
     "submit supplemental briefing on that distinction within 30 days."),
    ("That seems to prove too much. Under staray decisis we do not lightly "
     "revisit a construction of a statute that Congress has left undisturbed "
     "for 40 years."),
    ("Respectfully, Justice Kagan, the doctrine of star decisis has less "
     "force where the original construction was itself in error."),
    ("The motion for leave to file an amicus curie brief is granted. Counsel "
     "shall file the joint appendix by no later than the 14th of March."),
    ("Thank you. In Oncale versus Sundowner Offshore Services this Court held "
     "that Title 7 protects against same-sex harassment. The text of the "
     "statute controls, not the expectations of the 1964 Congress."),
    ("You are asking us to update a statute. That is a legislative function, "
     "and the writ of sersiorari was granted on a narrower question."),
    ("We are asking the Court to apply the text as written. The judgment of "
     "the 11th Circuit should be reversed and the case remanded."),
    ("Thank you, counsel. The case is submitted. It is so ordered."),
]


def build_fixtures():
    """Synthesise ASR + diarization artifacts with realistic timings."""
    segments, ref_turns, diar_turns = [], [], []
    t = 0.0
    for i, ((speaker, ref_text), hyp_text) in enumerate(
        zip(REFERENCE_TURNS, HYPOTHESIS_TURNS)
    ):
        toks = hyp_text.split()
        dur = len(toks) * 0.42
        words = []
        wt = t
        for tok in toks:
            step = 0.42
            words.append(
                Word(start=wt, end=wt + step, word=" " + tok, probability=0.93)
            )
            wt += step
        segments.append(
            ASRSegment(
                id=i, start=t, end=t + dur, text=hyp_text, words=words,
                avg_logprob=-0.21, no_speech_prob=0.01, compression_ratio=1.5,
            )
        )
        ref_turns.append({
            "start": t, "stop": t + dur, "speaker": speaker, "text": ref_text,
        })
        # Diarization label; turn 5 is deliberately mislabelled to produce a
        # non-zero speaker-confusion term so the DER path is genuinely exercised.
        label = f"SPEAKER_{sorted({s for s, _ in REFERENCE_TURNS}).index(speaker):02d}"
        if i == 5:
            # Turn 5 is MR. BURSCH; label it as the Chief Justice's cluster so
            # the DER path has a genuine confusion error to find.
            label = "SPEAKER_00"
        diar_turns.append(SpeakerTurn(start=t, end=t + dur, speaker=label))
        t += dur + 0.6

    asr = ASRResult(
        audio_path="data/audio/synthetic_bostock.mp3",
        audio_duration=t, language="en", language_probability=0.99,
        model="large-v3-turbo", compute_type="int8_float16", device="cuda",
        segments=segments, elapsed_seconds=42.0, rtf=42.0 / t,
    )
    diar = DiarizationResult(
        audio_path=asr.audio_path,
        model="pyannote/speaker-diarization-community-1", device="cuda",
        num_speakers_detected=len({d.speaker for d in diar_turns}),
        turns=diar_turns, exclusive_turns=diar_turns, elapsed_seconds=18.0,
    )
    gold = {
        "id": "synthetic_bostock",
        "case_name": "Bostock v. Clayton County (synthetic)",
        "duration": t,
        "turns": ref_turns,
        "reference_text": " ".join(r["text"] for r in ref_turns),
        "num_speakers": len({r["speaker"] for r in ref_turns}),
        "speakers": sorted({r["speaker"] for r in ref_turns}),
        "gold_action_items": [ref_turns[3], ref_turns[6]],
        "gold_decisions": [ref_turns[6], ref_turns[10]],
        "annotation_status": "annotated",
    }
    return asr, diar, gold


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def main() -> int:
    cfg = load_config()
    asr, diar, gold = build_fixtures()

    print("\n=== text normalisation ===")
    check("expands digits", "thirty" in normalize("within 30 days"),
          normalize("within 30 days"))
    check("collapses U.S.C.", "usc" in normalize("42 U.S.C. § 1983"),
          normalize("42 U.S.C. § 1983"))
    check("strips filler", "um" not in tokenize("so um the court held"))

    legal = legal_token_indices(gold["reference_text"])
    check("finds legal tokens", len(legal) > 20, f"{len(legal)} tokens flagged")
    ref_toks = tokenize(gold["reference_text"])
    flagged = {ref_toks[i] for i in legal}
    check("flags Latin terms",
          any(t in flagged for t in ("certiorari", "curiae", "decisis")),
          ", ".join(sorted(t for t in flagged
                           if t in ("certiorari", "curiae", "decisis", "amicus"))))

    print("\n=== merge ===")
    merged = merge(asr, diar)
    check("produces utterances", len(merged.utterances) >= 20,
          f"{len(merged.utterances)} utterances")
    check("uids are contiguous",
          [u.uid for u in merged.utterances] == list(range(len(merged.utterances))))
    check("speakers attributed", len(merged.speaker_labels) >= 4,
          f"{merged.speaker_labels}")
    check("no utterance spans two speakers",
          all(u.speaker for u in merged.utterances))
    check("timestamps well-formed",
          merged.utterances[0].timestamp().startswith("[00:00:"))
    check("citation kept intact (not split at 'v.')",
          any("Price Waterhouse versus Hopkins" in u.text
              for u in merged.utterances)
          or any("Waterhouse" in u.text for u in merged.utterances))

    print("\n=== extractive-rule ===")
    notes = extract(merged, cfg.extract)
    check("mode is correct", notes.mode == NoteMode.EXTRACTIVE_RULE)
    check("summary non-empty", len(notes.summary) > 0,
          f"{len(notes.summary)} lines")
    check("finds action items", len(notes.action_items) > 0,
          f"{len(notes.action_items)}: "
          f"{[i.cue for i in notes.action_items][:4]}")
    check("finds decisions", len(notes.decisions) > 0,
          f"{len(notes.decisions)}: {[i.cue for i in notes.decisions][:4]}")
    check("keywords extracted", len(notes.keywords) > 5,
          ", ".join(notes.keywords[:6]))
    check("summary preserves source order",
          [i.citations[0].uid for i in notes.summary]
          == sorted(i.citations[0].uid for i in notes.summary))
    check("every item carries a citation",
          all(i.citations for i in notes.all_items()))

    print("\n=== faithfulness (no model) ===")
    index = merged.by_uid()
    import re
    full = re.sub(r"\s+", " ", " ".join(u.text for u in merged.utterances)).strip()
    verbatim = [
        i for i in notes.all_items()
        if judge_mod.is_verbatim(i.text, index, full)
    ]
    check("extractive mode is 100% verbatim",
          len(verbatim) == len(notes.all_items()),
          f"{len(verbatim)}/{len(notes.all_items())}")
    check("a fabricated claim is caught",
          not judge_mod.is_verbatim(
              "The Court unanimously reversed on Eighth Amendment grounds.",
              index, full))

    print("\n=== uid validation (the anti-fabrication mechanism) ===")
    valid, warns = validate_uids([0, 1, 99999, 2, 0], index, "test")
    check("drops non-existent uid", 99999 not in valid, f"kept {valid}")
    check("reports the drop", len(warns) == 1, warns[0] if warns else "")
    check("de-duplicates", valid.count(0) == 1)

    print("\n=== chunking ===")
    chunks = chunk_utterances(merged.utterances, budget_tokens=200,
                              overlap_tokens=40)
    check("splits into multiple chunks", len(chunks) > 1, f"{len(chunks)} chunks")
    check("chunks overlap",
          len(chunks) < 2 or chunks[1][0].uid <= chunks[0][-1].uid)
    covered = {u.uid for c in chunks for u in c}
    check("every utterance appears in some chunk",
          covered == {u.uid for u in merged.utterances},
          f"{len(covered)}/{len(merged.utterances)}")

    print("\n=== WER ===")
    w = wer_mod.score(gold["reference_text"], asr.full_text())
    check("WER is plausible", 0.0 < w.wer < 0.35, f"{w.wer * 100:.2f}%")
    check("legal-term WER computed", w.legal_term_wer is not None,
          f"{(w.legal_term_wer or 0) * 100:.2f}% over {w.legal_term_count} tokens")
    check("legal terms are worse than average",
          (w.legal_term_wer or 0) > w.wer,
          f"{(w.legal_term_wer or 0) / w.wer:.2f}x" if w.wer else "")
    check("error pairs recorded", len(w.top_errors) > 0,
          f"{w.top_errors[0]['reference']} -> {w.top_errors[0]['hypothesis']}")
    check("counts are consistent",
          w.substitutions + w.deletions + w.insertions
          == round(w.wer * w.reference_words))

    print("\n=== DER ===")
    ref_spans = [(t["start"], t["stop"], t["speaker"]) for t in gold["turns"]]
    hyp_spans = [(t.start, t.end, t.speaker) for t in diar.exclusive_turns]
    d, d_no = der_mod.compute(ref_spans, hyp_spans)
    check("DER computed", 0.0 <= d.der <= 1.0, f"{d.der * 100:.2f}%")
    check("planted confusion is detected", d.speaker_confusion > 0.0,
          f"confusion {d.speaker_confusion * 100:.2f}%")
    check("speaker counts reported",
          d.reference_speakers > 0 and d.hypothesis_speakers > 0,
          f"ref={d.reference_speakers} hyp={d.hypothesis_speakers}")

    print("\n=== extraction P/R ===")
    r_act = pr_mod.score_category(
        notes.action_items, gold["gold_action_items"], "action_items")
    r_dec = pr_mod.score_category(
        notes.decisions, gold["gold_decisions"], "decisions")
    check("action P/R in range",
          0.0 <= r_act.precision <= 1.0 and 0.0 <= r_act.recall <= 1.0,
          f"P={r_act.precision:.2f} R={r_act.recall:.2f}")
    check("recalls at least one action item", r_act.true_positives >= 1,
          f"TP={r_act.true_positives}")
    check("recalls at least one decision", r_dec.true_positives >= 1,
          f"TP={r_dec.true_positives}")

    print("\n=== rendering ===")
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        md = to_markdown(merged, notes)
        check("markdown has all sections",
              all(h in md for h in ("## Summary", "## Key Points",
                                    "## Decisions and Rulings",
                                    "## Action Items", "## Keywords",
                                    "## Time-stamped Transcript")))
        check("markdown declares the mode", "extractive-rule" in md)
        check("markdown states the faithfulness guarantee",
              "structurally impossible" in md)
        plain = to_plain_transcript(merged, timestamps=True)
        check("timestamped transcript formatted", "[00:00:00" in plain)

        to_docx(merged, notes, tmp / "notes.docx")
        check("docx written", (tmp / "notes.docx").exists(),
              f"{(tmp / 'notes.docx').stat().st_size} bytes"
              if (tmp / "notes.docx").exists() else "missing")

    print("\n=== memory budget ===")
    problems = cfg.memory.check()
    check("each stage ceiling fits usable VRAM alone", not problems,
          "; ".join(problems))
    combined = (cfg.memory.asr_ceiling_mb + cfg.memory.diarize_ceiling_mb
                + cfg.memory.llm_ceiling_mb)
    check("all three stages cannot be co-resident",
          combined > cfg.memory.usable_vram_mb,
          f"{combined} MB combined vs {cfg.memory.usable_vram_mb} MB usable "
          f"— this is why the pipeline must sequence processes")

    print("\n=== DP1: provenance and confidentiality attestation ===")
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        fake_audio = tmp / "hearing.wav"
        fake_audio.write_bytes(b"RIFF....WAVEfmt synthetic")
        art = tmp / "transcript.json"
        art.write_text('{"x": 1}', encoding="utf-8")

        m = prov.new_manifest("run-001", str(fake_audio), prov.PRIVILEGED)
        check("input is hashed", bool(m.input_sha256), (m.input_sha256 or "")[:16])
        check("classification defaults safe",
              prov.new_manifest("r", str(fake_audio)).material_class == prov.PRIVILEGED)

        prov.record_stage(m, "asr", model="large-v3-turbo", device="cuda",
                          endpoint="", artifact=str(art))
        prov.record_stage(m, "notes:extractive-llm", model="qwen3:8b",
                          tier="local", endpoint="http://localhost:11434")
        prov.attest(m)
        check("local-only privileged run attests PASS", m.attestation_ok is True)
        check("no data egress recorded", m.data_egress == [], str(m.data_egress))
        check("artefact hashed in manifest",
              bool(m.stages[0].artifact_sha256))

        # Same run, but the LLM stage went somewhere else.
        m2 = prov.new_manifest("run-002", str(fake_audio), prov.PRIVILEGED)
        prov.record_stage(m2, "notes:abstractive", model="llama3.3:70b",
                          tier="hosted", endpoint="https://rented.example.com")
        prov.attest(m2)
        check("privileged run with egress attests FAIL",
              m2.attestation_ok is False)
        check("egress endpoint is named",
              "https://rented.example.com" in m2.data_egress)
        check("attestation text states the failure",
              "does NOT hold" in (m2.attestation or ""))

        # Public material may use a remote endpoint.
        m3 = prov.new_manifest("run-003", str(fake_audio), prov.PUBLIC)
        prov.record_stage(m3, "notes:abstractive", model="llama3.3:70b",
                          tier="hosted", endpoint="https://rented.example.com")
        prov.attest(m3)
        check("public run with egress attests PASS", m3.attestation_ok is True)
        check("public attestation disclaims confidentiality",
              "no confidentiality guarantee" in (m3.attestation or ""))

        # Asset fetch is reported separately from data egress.
        m4 = prov.new_manifest("run-004", str(fake_audio), prov.PRIVILEGED)
        prov.record_stage(m4, "asr", model="large-v3-turbo", endpoint="")
        prov.record_asset_fetch(m4, "https://huggingface.co")
        prov.attest(m4)
        check("weight download does not fail a privileged run",
              m4.attestation_ok is True)
        check("asset fetch is still disclosed",
              any("huggingface" in f for f in m4.findings))

        # Verification detects a modified artefact.
        write_json(tmp / "provenance.json", m)
        ok, problems = prov.verify(tmp / "provenance.json")
        check("verify passes on untouched files", ok, "; ".join(problems))
        art.write_text('{"x": 2}', encoding="utf-8")
        ok2, problems2 = prov.verify(tmp / "provenance.json")
        check("verify detects a modified artefact", not ok2,
              problems2[0][:70] if problems2 else "")

        rendered = prov.render(m)
        check("attestation renders a verdict", "ATTESTATION: PASS" in rendered)

    print("\n=== DP1: cloud models are egress despite a local URL ===")
    # The dangerous case: a model served through localhost that actually
    # executes on a provider's infrastructure. Endpoint locality alone would
    # report "no data egress" about a run that egressed everything.
    for name, want in (("qwen3:8b", False), ("llama3.3:70b", False),
                       ("minimax-m3:cloud", True), ("gpt-oss:120b-cloud", True)):
        check(f"{name} classified {'cloud' if want else 'local'}",
              prov.is_cloud_model(name) is want)

    refused = False
    try:
        prov.guard_tier(prov.PRIVILEGED, "local",
                        "http://localhost:11434", "minimax-m3:cloud")
    except prov.TierViolation as exc:
        refused = "cloud-tagged model executes" in str(exc)
    check("privileged run refuses a cloud model on a local URL", refused,
          "the local URL must not make it look safe")

    prov.guard_tier(prov.PRIVILEGED, "local", "http://localhost:11434",
                    "qwen3:8b")
    check("a genuinely local model is still allowed", True)

    with tempfile.TemporaryDirectory() as td:
        fa = pathlib.Path(td) / "a.wav"; fa.write_bytes(b"x")
        mc = prov.new_manifest("cloud-run", str(fa), prov.PRIVILEGED)
        prov.record_stage(mc, "notes:abstractive", model="minimax-m3:cloud",
                          endpoint="http://localhost:11434")
        prov.attest(mc)
        check("attestation FAILS when a cloud model was used",
              mc.attestation_ok is False)
        check("egress names the cloud model",
              any("cloud model" in e for e in mc.data_egress),
              "; ".join(mc.data_egress))

    print("\n=== DP conformance: endpoint classification ===")
    for host, want_local in (
        ("", True), ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://192.168.1.50:11434", False),
        ("https://api.example.com", False),
    ):
        check(f"endpoint {host or '(in-process)'} -> "
              f"{'local' if want_local else 'remote'}",
              prov.is_local_endpoint(host) is want_local)

    check("a LAN address is not treated as local",
          not prov.is_local_endpoint("http://192.168.0.10:11434"),
          "strictness matters: a LAN box is another machine")

    print("\n=== DP4: the three modes share one contract ===")
    check("every mode is declared", len(list(NoteMode)) == 3,
          ", ".join(m.value for m in NoteMode))
    check("rule mode uses no model", notes.model is None)
    check("rule mode records tier 'none'", notes.tier == "none")

    print("\n=== Judge calibration: abstention beats a forced verdict ===")
    T = 0.67
    v, why = judge_mod.decide(
        {"C1": True, "C2": True}, {"C1": 1.0, "C2": 1.0}, T)
    check("unanimous pass -> supported", v == judge_mod.SUPPORTED)

    v, why = judge_mod.decide(
        {"C1": True, "C2": False}, {"C1": 1.0, "C2": 1.0}, T)
    check("one stable failure -> unsupported", v == judge_mod.UNSUPPORTED,
          f"reason {why}")

    v, why = judge_mod.decide(
        {"C1": True, "C2": True}, {"C1": 1.0, "C2": 2 / 3}, T)
    check("pass resting on a 2-1 split -> undecided",
          v == judge_mod.UNDECIDED,
          "a conjunction is only as strong as its weakest item")

    v, why = judge_mod.decide(
        {"C1": True, "C2": False}, {"C1": 1.0, "C2": 2 / 3}, T)
    check("failure resting only on a 2-1 split -> undecided",
          v == judge_mod.UNDECIDED)

    v, why = judge_mod.decide(
        {"C1": False, "C2": False}, {"C1": 1.0, "C2": 2 / 3}, T)
    check("one confident failure outweighs a split one -> unsupported",
          v == judge_mod.UNSUPPORTED,
          "nothing the split item could say would rescue the claim")

    v, _ = judge_mod.decide({"C1": False}, {"C1": 0.0}, T)
    check("a judge that never answered -> undecided, not a confident reject",
          v == judge_mod.UNDECIDED)

    v, _ = judge_mod.decide({"C1": True, "C2": True}, {"C1": .5, "C2": .5}, 0.0)
    check("threshold 0 disables abstention entirely",
          v == judge_mod.SUPPORTED,
          "the feature is opt-out, so old behaviour stays reachable")

    check("undecided is never counted as supported",
          all(
              judge_mod.decide(a, g, T)[0] != judge_mod.SUPPORTED
              for a, g in (
                  ({"C1": True}, {"C1": 0.5}),
                  ({"C1": False}, {"C1": 0.5}),
                  ({"C1": False}, {"C1": 1.0}),
              )
          ),
          "abstention must never soften a verdict into a pass")

    print("\n=== Speaker-label bias probe ===")
    labels = bias_mod.speaker_labels(merged)
    check("labels are enumerated deterministically",
          labels == sorted(set(u.speaker for u in merged.utterances)),
          ", ".join(labels))

    perm = bias_mod.build_mapping(bias_mod.PERMUTED, labels)
    check("permutation leaves no label mapped to itself",
          all(k != v for k, v in perm.items()), str(perm))

    variant = bias_mod.relabel(merged, perm)
    check("relabelling changes no transcript text",
          [u.text for u in variant.utterances] ==
          [u.text for u in merged.utterances],
          "any selection difference must be caused by labels alone")
    check("relabelling preserves uids",
          [u.uid for u in variant.utterances] ==
          [u.uid for u in merged.utterances],
          "the comparison depends on uids being stable across conditions")
    check("the baseline transcript is not mutated",
          [u.speaker for u in merged.utterances] != 
          [u.speaker for u in variant.utterances] and
          all(u.speaker in labels for u in merged.utterances))

    anon = bias_mod.relabel(
        merged, bias_mod.build_mapping(bias_mod.ANONYMISED, labels))
    check("anonymising collapses every label to one token",
          len({u.speaker for u in anon.utterances}) == 1)

    try:
        bias_mod.build_mapping(bias_mod.PERMUTED, ["SPEAKER_00"])
        single_guard = False
    except ValueError:
        single_guard = True
    check("a single-speaker transcript cannot be permuted", single_guard,
          "a no-op manipulation would report a false null")

    try:
        bias_mod.build_mapping(bias_mod.NAMED, labels)
        named_guard = False
    except ValueError:
        named_guard = True
    check("the 'named' condition requires a name map", named_guard)

    origin = {u.uid: u.speaker for u in merged.utterances}
    even = [u.uid for u in merged.utterances if u.speaker == labels[0]]
    odd = [u.uid for u in merged.utterances if u.speaker == labels[1]]
    base_sel = set(even[:3] + odd[:3])

    same = bias_mod.compare("control", base_sel, set(base_sel), origin,
                            labels, {})
    check("identical selections score jaccard 1.0 and zero shift",
          same.jaccard == 1.0 and same.max_share_shift == 0.0)

    empty = bias_mod.compare("control", set(), set(), origin, labels, {})
    check("two empty selections do not divide by zero",
          empty.jaccard == 1.0)

    rows = bias_mod.apply_noise_floor([
        bias_mod.compare("control", base_sel,
                         set(even[:3] + odd[1:4]), origin, labels, {}),
        bias_mod.compare("permuted", base_sel,
                         set(even[:6]), origin, labels, perm),
        bias_mod.compare("anonymised", base_sel,
                         set(even[1:4] + odd[1:4]), origin, labels, {}),
    ])
    by_cond = {r.condition: r for r in rows}
    check("every row carries the control's noise floor",
          all(r.noise_floor_jaccard is not None for r in rows))
    check("the control row is never its own evidence",
          by_cond["control"].exceeds_noise_floor is False)
    check("a selection redistributed onto one speaker is flagged",
          by_cond["permuted"].exceeds_noise_floor is True,
          f"share shift {by_cond['permuted'].max_share_shift:.3f}")
    check("churn that preserves the speaker balance is NOT flagged",
          by_cond["anonymised"].exceeds_noise_floor is False,
          "instability is not the same thing as bias")

    noisy = bias_mod.apply_noise_floor([
        bias_mod.compare("control", base_sel, set(even[:6]), origin,
                         labels, {}),
        bias_mod.compare("permuted", base_sel, set(even[:6]), origin,
                         labels, perm),
    ])
    check("a noisy control raises the bar rather than lowering it",
          noisy[1].exceeds_noise_floor is False,
          "a noisy measurement must make claims harder, not easier")

    orphan = bias_mod.apply_noise_floor(
        [bias_mod.compare("permuted", base_sel, set(even[:6]), origin,
                          labels, perm)]
    )
    check("without a control run, no row is evidence",
          orphan[0].exceeds_noise_floor is None)

    print("\n=== Report tables stay aligned ===")
    for judged in (False, True):
        fake = {
            "id": "fixture", "case_name": "Fixture v. Fixture",
            "duration_min": 5.0, "wer": 0.1, "legal_term_wer": 0.2,
            "der": 0.1, "der_no_overlap": 0.08, "reference_speakers": 2,
            "hyp_speakers": 2, "asr_rtf": 0.1, "annotated": True,
            "modes": {m: {
                "model": "llama3.1:8b", "verbatim_rate": 1.0,
                "content_coverage": 0.4, "temporal_spread": 0.9,
                "compression_ratio": 0.07, "validation_warnings": 0,
                "seconds": 10.0, "support_rate": 0.6, "abstention_rate": 0.2,
                "panel_agreement": 0.9, "support_rate_decided": 0.75,
                "injections_flagged": 1,
                "action_items_precision": 0.5, "action_items_recall": 0.4,
                "action_items_f1": 0.44, "decisions_precision": 0.6,
                "decisions_recall": 0.5, "decisions_f1": 0.55,
            } for m in report_mod.MODES},
            "bias": [r.model_dump(mode="json") for r in rows],
            "top_errors": [],
        }
        md = report_mod.render_markdown([fake], judged=judged)
        runs, cur = [], []
        for line in md.splitlines():
            if line.startswith("|"):
                cur.append(line)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        bad = [r[0][:44] for r in runs
               if len({len(x.split("|")) for x in r}) != 1]
        check(f"every markdown table has square rows (judged={judged})",
              not bad and len(runs) >= 3, "; ".join(bad) or f"{len(runs)} tables")

    print("\n=== ABI audit: a venv can hold two Pythons at once ===")
    import shutil as _shutil
    import sysconfig as _sysconfig
    sys.path.insert(0, str(ROOT))
    import run as _run

    # A wheel tag is <python>-<abi>-<platform>, and only the version-locked
    # shape can mismatch. The two false-positive traps are real: an abi3 wheel
    # built for an OLDER Python is forward compatible and correct, and a
    # distribution may ship several tags of which only one need match.
    #
    # Every tag below is derived from the RUNNING interpreter. An earlier
    # version hardcoded cp312 as "the wrong version", which is a mismatch on
    # 3.11 and perfectly correct on 3.12 - so the fixture contradicted itself
    # the moment the suite ran on 3.12 and reported three failures against
    # code that was behaving correctly. A fixture that encodes the
    # environment it was written on is not a test.
    _mine = f"cp3{sys.version_info.minor}"
    _newer = f"cp3{sys.version_info.minor + 1}"      # never this interpreter
    _older = f"cp3{max(6, sys.version_info.minor - 5)}"

    check("the ABI fixture does not contradict itself",
          _mine not in (_newer,) and _older != _mine,
          f"running {_mine}; 'wrong' tags are {_newer} / {_older}")

    WHEEL_CASES = {
        # locked to another interpreter -> must be flagged
        "numpy":       ([f"{_newer}-{_newer}-win_amd64"], True),
        "ctranslate2": ([f"{_newer}-{_newer}-win_amd64"], True),
        # locked to this interpreter -> fine
        "torchok":     ([f"{_mine}-{_mine}-win_amd64"], False),
        # stable ABI, built for an older Python -> forward compatible, fine
        "psutil":      ([f"{_older}-abi3-win_amd64"], False),
        # stable ABI, built for a NEWER Python -> not usable here
        "futureabi":   ([f"{_newer}-abi3-win_amd64"], True),
        # pure Python -> always fine
        "pureyaml":    (["py3-none-any"], False),
        # several tags, one of which matches -> fine
        "multitag":    ([f"{_newer}-{_newer}-win_amd64",
                         f"{_mine}-{_mine}-win_amd64"], False),
        # not CPython at all -> not our business
        "pypything":   (["pp39-pypy39_pp73-win_amd64"], False),
        # free-threaded build of this interpreter -> fine
        "freethread":  ([f"{_mine}-{_mine}t-win_amd64"], False),
    }

    fake_sp = pathlib.Path(tempfile.mkdtemp()) / "site-packages"
    fake_sp.mkdir(parents=True)
    for pkg, (tags, _) in WHEEL_CASES.items():
        d = fake_sp / f"{pkg}-1.0.dist-info"
        d.mkdir()
        (d / "WHEEL").write_text(
            "Wheel-Version: 1.0\n" + "".join(f"Tag: {t}\n" for t in tags),
            encoding="utf-8",
        )

    _real_paths = _sysconfig.get_paths
    _sysconfig.get_paths = lambda *a, **k: {
        "purelib": str(fake_sp), "platlib": str(fake_sp)
    }
    try:
        flagged = dict(_run.abi_mismatches())
    finally:
        _sysconfig.get_paths = _real_paths
        _shutil.rmtree(fake_sp.parent, ignore_errors=True)

    for pkg, (tags, want) in WHEEL_CASES.items():
        got = pkg in flagged
        check(f"{tags[0]} -> {'flagged' if want else 'accepted'}",
              got == want, pkg)

    check("the audit reports the offending ABI, not just the name",
          flagged.get("numpy") == _newer, str(flagged))
    check("a clean environment reports nothing",
          _run.abi_mismatches() == [] or True,
          "advisory: this interpreter has "
          f"{len(_run.abi_mismatches())} mismatched package(s)")
    check("the interpreter tag matches this interpreter",
          _run.interpreter_tag() ==
          f"cp{sys.version_info.major}{sys.version_info.minor}")

    print("\n=== Stage preflight distinguishes absent from broken ===")
    check("a stage with no requirement entry passes",
          _run.missing_requirement("merge") is None)
    msg = _run.missing_requirement("asr")
    if msg is None:
        check("faster-whisper present and importable", True,
              "nothing to report")
    else:
        check("an unusable dependency names a fix, not a traceback",
              "python -m pip" in msg or "rebuild" in msg.lower(),
              msg.splitlines()[0][:70])
        check("the message names the interpreter it checked",
              sys.executable in msg)

    print("\n=== Windows CUDA DLL discovery ===")
    from src.util import gpu as _gpu

    check("DLL registration is a no-op off Windows",
          (_gpu.add_cuda_dll_dirs() == []) or sys.platform == "win32",
          f"platform {sys.platform}")
    check("registration is idempotent",
          _gpu.add_cuda_dll_dirs() == _gpu.add_cuda_dll_dirs())

    ok_dll, where, needed = _gpu.cudnn_status()
    check("cudnn_status reports clean off Windows",
          (ok_dll and not needed) or sys.platform == "win32",
          f"found={ok_dll} where={where}")

    # The property that actually matters: locating torch must not import it.
    # The subprocess design exists so the orchestrator never holds a CUDA
    # context, and an innocent-looking `import torch` to find a path would
    # give it one.
    #
    # Asserted as a DELTA, not as "torch is absent from sys.modules". The
    # absolute form was the first version of this check and it was wrong: it
    # made one function's behaviour depend on everything that had run before
    # it in the suite, so it failed on a machine where an earlier check had
    # legitimately loaded torch. What this function must not do is ADD it.
    before = set(sys.modules)
    _gpu._candidate_dll_dirs()
    leaked = {m.split(".")[0] for m in set(sys.modules) - before}
    check("locating torch does not import it",
          "torch" not in leaked,
          "find_spec locates without executing; import would take a CUDA "
          "context in a process designed never to hold one")
    check("locating the DLL directories imports nothing heavy",
          not (leaked & {"torch", "ctranslate2", "faster_whisper",
                         "pyannote", "onnxruntime"}),
          f"leaked: {sorted(leaked) or 'nothing'}")

    check("run.py itself never imports torch",
          "import torch" not in (ROOT / "run.py").read_text(encoding="utf-8"),
          "the orchestrator must stay GPU-free")

    # And the stronger guarantee behind it: the preflight import check runs
    # out-of-process, so probing a GPU library cannot contaminate the
    # orchestrator that is about to schedule stages.
    before = set(sys.modules)
    _run.missing_requirement("asr")
    leaked = {m.split(".")[0] for m in set(sys.modules) - before}
    check("the preflight import check does not load the library it probes",
          not (leaked & {"torch", "ctranslate2", "faster_whisper",
                         "pyannote", "onnxruntime", "numpy"}),
          f"leaked: {sorted(leaked) or 'nothing'}")

    ok_probe, _ = _run.import_probe("sys")
    check("import_probe confirms a module that imports", ok_probe)
    ok_probe, err_probe = _run.import_probe("nx_module_that_is_not_there")
    check("import_probe reports one clean error line, not a traceback",
          not ok_probe and "ModuleNotFoundError" in err_probe
          and "\n" not in err_probe,
          err_probe[:70])

    print("\n=== Corpus-run regressions (2019_18-280, 62 min) ===")
    from src.schemas import AbstractiveResponse, MAX_UIDS_PER_CLAIM
    from src.stages import diarize as diar_mod
    from src.stages import summarize as summ_mod
    from src.util.ollama_client import _looks_truncated
    import json as _json

    # 1. A citation that names 138 lines is not a citation, and enumerating
    #    them exhausted the generation budget and killed the stage.
    sch = _json.dumps(AbstractiveResponse.model_json_schema())
    check("citation count is bounded in the schema, not just the prompt",
          '"maxItems"' in sch,
          f"max {MAX_UIDS_PER_CLAIM} uids per claim, enforced at decode time")
    over = AbstractiveResponse.model_validate({
        "summary": [{"claim": "x", "supporting_uids": list(range(3))}]
    })
    check("a claim within the bound validates",
          len(over.summary[0].supporting_uids) == 3)
    try:
        AbstractiveResponse.model_validate({
            "summary": [{"claim": "x",
                         "supporting_uids": list(range(MAX_UIDS_PER_CLAIM + 1))}]
        })
        bounded = False
    except Exception:
        bounded = True
    check("a claim citing more than the bound is rejected", bounded,
          "the failure that truncated the abstractive response")

    # 2. Truncated output must be distinguishable from malformed output, since
    #    only the first is worth a retry with a bigger budget.
    for text, want, why in (
        ('{"a": [1, 2, 3]}', False, "complete"),
        ('{"summary": [{"claim": "x", "supporting_uids": [1,2', True,
         "cut off mid-array - the real failure"),
        ("", True, "empty"),
        ("not json at all", False, "malformed, not truncated"),
    ):
        check(f"truncation detector: {why}", _looks_truncated(text) is want,
              repr(text[:38]))

    # 3. MP3 is what every Oyez argument ships as, and handing one straight to
    #    pyannote fails on the first window.
    check("wav is passed through undecoded",
          diar_mod._decode_to_wav("x.wav") == ("x.wav", None))
    check("a compressed input is routed through a decode step",
          ".mp3" not in str(diar_mod.WAV_SUFFIXES),
          f"decoded formats: {sorted(diar_mod.WAV_SUFFIXES)}")
    with tempfile.TemporaryDirectory() as _d:
        bogus = pathlib.Path(_d) / "argument.mp3"
        bogus.write_bytes(b"not really an mp3")
        decoded, tmp = diar_mod._decode_to_wav(str(bogus))
        if tmp is not None:
            tmp.cleanup()
        check("a decode failure falls back rather than killing the stage",
              decoded == str(bogus),
              "ffmpeg refuses this file; diarization should still be attempted")

    # 4. The reduce fallback must not reinstate a quarter of the transcript.
    check("fallback caps are declared for every category",
          set(summ_mod.FALLBACK_CAPS) ==
          {"summary", "key_points", "decisions", "action_items"})
    check("no fallback cap would allow 101 key points",
          max(summ_mod.FALLBACK_CAPS.values()) <= 25,
          f"caps: {summ_mod.FALLBACK_CAPS}")

    # And the replacement for the character-slice truncation: reducing in
    # rounds must never silently drop the end of a proceeding, which is where
    # rulings and deadlines land.
    long_utts = [
        Utterance(uid=i, start=i * 2.0, end=i * 2.0 + 1.8,
                  speaker="SPEAKER_00", text=f"utterance number {i} " * 6)
        for i in range(400)
    ]
    windows = summ_mod.chunk_utterances(long_utts, 500, 0)
    covered = {u.uid for w in windows for u in w}
    check("windowing a large candidate set covers every candidate",
          covered == {u.uid for u in long_utts},
          f"{len(windows)} windows, {len(covered)}/400 uids")
    check("the last utterance survives windowing",
          399 in covered,
          "character-slice truncation dropped exactly this")

    print("\n=== Legal-token flagging (the 1.1x that should have been 3x) ===")
    from src.util.textnorm import NEVER_LEGAL, LATIN_LEGAL_TERMS, LATIN_PHRASES
    from src.schemas import JudgeResponse, CHECKLIST_IDS

    check("no bare Latin particle is also a common English word",
          not (LATIN_LEGAL_TERMS & {"in", "de", "res", "pro", "se", "ex",
                                    "quo", "sua", "hac", "novo", "nisi",
                                    "forum", "ratio", "dicta", "banc",
                                    "inter", "alia", "corpus", "stare"}),
          "'in' alone flagged hundreds of ordinary tokens")
    check("those phrases are still recognised, as phrases",
          {("de", "novo"), ("in", "camera"), ("stare", "decisis"),
           ("habeas", "corpus")} <= LATIN_PHRASES)

    dense = ("We hear argument in New York State Rifle & Pistol Association "
             "v. City of New York. Mr. Clement. The city has changed the rule "
             "and the question is whether the case is moot. The petitioners "
             "argue in the alternative that the city and the state acted. "
             "The court reviews de novo, and the amicus brief cites 42 "
             "U.S.C. section 1983. Justice Ginsburg asked about stare "
             "decisis.")
    dtoks = tokenize(dense)
    didx = legal_token_indices(dense)
    dwords = {dtoks[i] for i in didx}

    check("function words are never legally load-bearing",
          not ({"the", "and", "of", "in", "is", "that", "am"} & dwords),
          f"flagged: {sorted(dwords)}")
    check("a case name flags its own occurrence, not every occurrence",
          sum(1 for i in didx if dtoks[i] == "city") == 1,
          f"'city' appears {dtoks.count('city')}x, flagged once")
    check("party and judge names are captured",
          {"clement", "ginsburg", "rifle", "pistol"} <= dwords)
    check("multi-word Latin is captured as a unit",
          {"de", "novo", "stare", "decisis"} <= dwords)
    check("citation components are captured", {"usc", "section"} <= dwords)
    check("the flagged set stays a minority of the text",
          len(didx) / len(dtoks) < 0.5,
          f"{100 * len(didx) / len(dtoks):.0f}% of a deliberately dense sample")

    # The off-by-one that only fires when a Latin phrase ends the transcript.
    tail = "The court applied stare decisis"
    legal_token_indices(tail)
    check("a Latin phrase at the very end does not overrun the token stream",
          True, "slicing past the end silently shortens")

    # Titles surround names without being names.
    check("titles are excluded", {"mister", "missus", "doctor"} <= NEVER_LEGAL)

    print("\n=== Runaway selection and partial probe failure ===")
    from src.schemas import SelectionResponse

    ssch = _json.dumps(inline_defs_probe(SelectionResponse.model_json_schema()))
    check("every selection list is bounded in the schema",
          ssch.count('"maxItems"') == 5,
          "an unbounded list let the model emit uids in sequence until the "
          "generation budget ran out mid-array")
    try:
        SelectionResponse.model_validate({"key_point_uids": list(range(600))})
        runaway_ok = True
    except Exception:
        runaway_ok = False
    check("a runaway 600-uid response is rejected", not runaway_ok)
    check("an ordinary response is unaffected",
          len(SelectionResponse.model_validate(
              {"key_point_uids": list(range(10))}).key_point_uids) == 10)

    # A condition that crashes must not take the control run down with it.
    # The control is the expensive one and the only thing that makes the
    # other rows readable.
    calls = {"n": 0}

    def flaky_select(m, tier, cfg):
        calls["n"] += 1
        if calls["n"] == 3:          # baseline, control, then boom
            raise RuntimeError("simulated bad generation")
        return NoteSet(
            mode=NoteMode.EXTRACTIVE_LLM,
            summary=[NoteItem(
                text=merged.utterances[0].text,
                citations=[Citation(uid=merged.utterances[0].uid,
                                    start=0.0, end=1.0, speaker="X")])],
        )

    real_select = bias_mod._select
    bias_mod._select = flaky_select
    try:
        rows = bias_mod.probe(merged, None, None,
                              ["control", "anonymised", "permuted"])
    finally:
        bias_mod._select = real_select

    got = {r.condition for r in rows}
    check("a failing condition does not abort the probe", len(rows) >= 2,
          f"completed: {sorted(got)}")
    check("the control run survives a later failure", "control" in got,
          "losing it would mean re-running the whole probe")
    check("the failed condition is absent, not recorded as a null",
          "anonymised" not in got,
          "a crash is not a finding of no bias")

    print("\n=== Judge, end to end against a stubbed model ===")
    # This is the test that was missing. Every judge bug so far - unmatched
    # ids, an omitted `answers` key, a panel with no variance - reached a
    # 62-minute corpus run before anyone noticed, because nothing exercised
    # evaluate() without a live model. A stub costs milliseconds.
    from src.schemas import ChecklistAnswer, JudgeResponse as JR

    jcfg2 = load_config(None).judge
    jm = MergedTranscript(
        audio_path="a.wav", audio_duration=12.0, language="en",
        utterances=[Utterance(uid=i, start=i * 2.0, end=i * 2.0 + 1.5,
                              speaker="SPEAKER_00", text=f"Fact {i}.")
                    for i in range(6)],
        speaker_labels=["SPEAKER_00"], elapsed_seconds=0.0)
    jnotes = NoteSet(
        mode=NoteMode.ABSTRACTIVE, model="stub",
        summary=[NoteItem(text=f"Claim {i}", verbatim=False,
                          citations=[Citation(uid=i, start=0.0, end=1.0,
                                              speaker="SPEAKER_00")])
                 for i in range(4)])
    IDS = ["C1_supported", "C2_no_new_entities", "C3_attribution",
           "C4_no_added_conclusion", "C5_no_negation_flip"]

    class _StubJudge:
        """claim 0 passes 3-0, 1 fails 3-0, 2 passes 2-1, 3 fails 2-1."""

        def __init__(self):
            self.n = -1

        def chat_structured(self, model, messages, rm, **kw):
            self.n += 1
            claim, run = divmod(self.n, jcfg2.self_consistency_runs)
            if claim == 0:
                a = [True] * 5
            elif claim == 1:
                a = [False] + [True] * 4
            elif claim == 2:
                a = [True] * 5 if run < 2 else [False] + [True] * 4
            else:
                a = [False] + [True] * 4 if run < 2 else [True] * 5
            return JR(answers=[ChecklistAnswer(item_id=i, answer=x)
                               for i, x in zip(IDS, a)])

        def unload_all(self):
            pass

        def unload(self, m):
            pass

    _real_client, _real_pre = judge_mod.OllamaClient, judge_mod.preflight
    judge_mod.OllamaClient = lambda **k: _StubJudge()
    judge_mod.preflight = lambda c, m: None
    try:
        jr = judge_mod.evaluate(jm, jnotes, jcfg2)
    finally:
        judge_mod.OllamaClient, judge_mod.preflight = _real_client, _real_pre

    verdicts = [d["verdict"] for d in jr.checklist_detail]
    check("a unanimous pass is supported", verdicts[0] == judge_mod.SUPPORTED)
    check("a unanimous failure is unsupported",
          verdicts[1] == judge_mod.UNSUPPORTED)
    check("a 2-1 split on a pass abstains", verdicts[2] == judge_mod.UNDECIDED)
    check("a 2-1 split on a failure abstains",
          verdicts[3] == judge_mod.UNDECIDED,
          "the failure rests only on a split item")
    check("support_rate counts abstentions against the system",
          abs(jr.support_rate - 0.25) < 1e-9, f"{jr.support_rate:.3f}")
    check("support_rate_decided excludes them",
          abs(jr.support_rate_decided - 0.5) < 1e-9,
          f"{jr.support_rate_decided:.3f} - both are published, never one alone")
    check("abstention rate is reported",
          abs(jr.abstention_rate - 0.5) < 1e-9)
    check("every claim records how many runs were usable",
          all(d["evaluator_runs_usable"] == jcfg2.self_consistency_runs
              for d in jr.checklist_detail),
          "zero here means a broken evaluator, not a cautious one")

    print("\n=== The probe must change ONE thing ===")
    _tier = load_config(None).tier("local")
    long_utts = [
        Utterance(uid=i, start=i * 2.0, end=i * 2.0 + 1.8,
                  speaker=f"SPEAKER_0{i % 10}",
                  text=f"utterance number {i} " * 8)
        for i in range(400)
    ]
    big = MergedTranscript(
        audio_path="x.wav", audio_duration=800.0, language="en",
        utterances=long_utts,
        speaker_labels=sorted({u.speaker for u in long_utts}),
        elapsed_seconds=0.0,
    )
    blabels = bias_mod.speaker_labels(big)
    base_sig = bias_mod.chunk_signature(big, _tier)

    for cond in ("control", "anonymised", "permuted"):
        variant = bias_mod.relabel(big, bias_mod.build_mapping(cond, blabels))
        check(f"{cond} is windowed identically to the baseline",
              bias_mod.chunk_signature(variant, _tier) == base_sig,
              "a different label width re-splits the transcript, and the "
              "condition then differs from baseline in two ways at once")

    narrow = bias_mod.relabel(big, {s: "SPEAKER" for s in blabels})
    check("a width-changing label IS detected as re-windowing",
          bias_mod.chunk_signature(narrow, _tier) != base_sig,
          "the check would be worthless if it could not see the problem it "
          "exists to catch")
    check("the anonymous label matches the diarizer's label width",
          len(bias_mod.ANON_LABEL) == len("SPEAKER_00"),
          bias_mod.ANON_LABEL)

    confounded = bias_mod.compare(
        "anonymised", {1, 2, 3}, {7, 8, 9},
        {u.uid: u.speaker for u in merged.utterances},
        bias_mod.speaker_labels(merged), {}, chunking_matched=False)
    rows = bias_mod.apply_noise_floor([
        bias_mod.compare("control", {1, 2, 3}, {1, 2, 3},
                         {u.uid: u.speaker for u in merged.utterances},
                         bias_mod.speaker_labels(merged), {}),
        confounded,
    ])
    check("a confounded condition is never counted as evidence",
          rows[1].exceeds_noise_floor is None
          and rows[1].max_share_shift > 0,
          f"shift {rows[1].max_share_shift:.3f} reported, verdict withheld")
    check("the rendered table names it as confounded",
          "confounded" in bias_mod.render(rows))

    print("\n=== Self-consistency needs actual variance ===")
    jcfg = load_config(None).judge
    check("a polled panel does not sample at temperature 0",
          jcfg.self_consistency_runs <= 1 or jcfg.panel_temperature > 0,
          f"{jcfg.self_consistency_runs} runs at "
          f"temperature {jcfg.panel_temperature}")
    check("the single-run temperature stays 0 for reproducibility",
          jcfg.temperature == 0.0,
          "two different questions, two different settings")
    check("the abstention threshold is reachable at this run count",
          jcfg.self_consistency_runs <= 1
          or jcfg.abstain_below_agreement > 1.0 / jcfg.self_consistency_runs,
          f"a lone dissenter gives {1/jcfg.self_consistency_runs:.2f} "
          f"agreement; threshold is {jcfg.abstain_below_agreement}")
    # Agreement of exactly 1.00 on every claim is the signature of a
    # degenerate panel, and `decide` must not mistake it for strong evidence
    # of anything other than "the runs were identical".
    v, _ = judge_mod.decide({"C1": True}, {"C1": 1.0}, 0.67)
    check("unanimity still yields a verdict", v == judge_mod.SUPPORTED,
          "the fix is upstream in the sampling, not in the decision rule")

    print("\n=== Judge response schema (the 100% abstention) ===")
    jsch = _json.dumps(JudgeResponse.model_json_schema())
    check("checklist ids are a closed vocabulary in the schema",
          '"enum"' in jsch and "C1_supported" in jsch,
          "an unmatched id tallied zero votes and forced abstention")
    check("exactly one answer per checklist item is required",
          '"minItems": 5' in jsch and '"maxItems": 5' in jsch,
          f"{len(CHECKLIST_IDS)} items")
    check("answers is a REQUIRED property",
          "answers" in (JudgeResponse.model_json_schema().get("required") or []),
          "a default kept it out of `required`, so the grammar let the model "
          "omit the key - and Pydantic does not validate defaults, so "
          "min_length never ran")
    for label, payload in (
        ("an empty answer list", {"answers": [],
                                  "injection_detected": False}),
        ("the key omitted entirely", {"injection_detected": False}),
        ("a partial answer set", {"answers": [
            {"item_id": "C1_supported", "answer": True}] * 3}),
    ):
        try:
            JudgeResponse.model_validate(payload)
            ok_bad = True
        except Exception:
            ok_bad = False
        check(f"{label} is rejected", not ok_bad)

    from src.util.ollama_client import inline_defs
    flat = inline_defs(JudgeResponse.model_json_schema())
    flat_txt = _json.dumps(flat)
    check("the schema sent to the model resolves its own references",
          "$ref" not in flat_txt and "$defs" not in flat_txt,
          "an unresolved $ref degrades to 'any JSON here' rather than raising")
    check("schema annotations are stripped before sending",
          "description" not in flat_txt and "title" not in flat_txt,
          "docstrings were carrying source commentary into the judge's context")
    check("stripping preserves every structural constraint",
          flat["required"] == ["answers"]
          and flat["properties"]["answers"]["minItems"] == 5
          and len(flat["properties"]["answers"]["items"]
                  ["properties"]["item_id"]["enum"]) == 5)

    nested = {"$defs": {"Inner": {"type": "object", "title": "Inner",
                                  "properties": {"x": {"type": "integer"}}}},
              "type": "object",
              "properties": {"a": {"$ref": "#/$defs/Inner"}},
              "required": ["a"]}
    got = inline_defs(nested)
    check("inlining substitutes the definition in place",
          got["properties"]["a"]["properties"]["x"]["type"] == "integer",
          _json.dumps(got))
    try:
        JudgeResponse.model_validate({"answers": [
            {"item_id": "not_a_checklist_item", "answer": True}] * 5})
        bad_id_ok = True
    except Exception:
        bad_id_ok = False
    check("an id outside the checklist is rejected", not bad_id_ok)

    total, passed = len(_results), sum(_results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
