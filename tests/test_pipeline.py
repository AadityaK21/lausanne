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
from src.eval import der as der_mod                                  # noqa: E402
from src.eval import extraction_pr as pr_mod                         # noqa: E402
from src.eval import judge as judge_mod                              # noqa: E402
from src.eval import wer as wer_mod                                  # noqa: E402
from src.schemas import (                                             # noqa: E402
    ASRResult, ASRSegment, DiarizationResult, NoteMode, SpeakerTurn, Word,
)
from src.stages.extract import extract                                # noqa: E402
from src.stages.merge import merge                                    # noqa: E402
from src.stages.render import to_docx, to_markdown, to_plain_transcript  # noqa: E402
from src.stages.summarize import chunk_utterances, validate_uids      # noqa: E402
from src.util import provenance as prov                                  # noqa: E402
from src.util.io import write_json                                       # noqa: E402
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

    total, passed = len(_results), sum(_results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
