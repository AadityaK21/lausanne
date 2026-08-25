# Notes for revising Section 8

What this implementation changes about the claims in
*NLP and LLMs for Legal AI: A Survey of Methods, Benchmarks, Reliability, and
a Note-Generation Prototype* (June 2026).

Three of these are corrections. One is a new contribution that did not exist
in the June draft and is, I think, the most defensible thing in Section 8.

---

## 1. The Llama-3.3-70B claim needs qualification (§8.2, §8.4)

**As written.** §8.2 says the transcript "can instead be passed to Meta's
Llama-3.3-70B-Instruct — an open-weight instruction-tuned model run locally on
the GPU workstation." §8.4 acknowledges it "exceeds the memory of a single
48 GB-class workstation GPU in half precision and must therefore be served with
weight quantization (e.g. 4-bit)."

**The problem.** At Q4_K_M, Llama-3.3-70B is ~43 GB of weights. §8.4's framing
implies that 4-bit quantisation resolves the memory issue for a workstation
GPU, which it does — for a 48 GB card. It does not make the model runnable on
consumer hardware, and the privacy argument in §8.3 is at its strongest
precisely when the hardware is consumer hardware. A confidentiality guarantee
that requires a data-centre GPU is a weaker guarantee than one that runs on a
laptop a solo practitioner already owns.

**What the implementation does.** The tier is now an explicit, recorded
property of every run (`src/config.py`):

- `local` — an 8B-class model at 4-bit, ~5.2 GB, runs on an RTX 4060 (8 GB).
  This is the configuration the privacy argument is about.
- `hosted` — the 70B configuration on a rented 48–80 GB GPU, used **only on
  public court audio**, so no privileged material ever reaches rented
  hardware and §8.3 survives intact.

Every artifact records which tier produced it; the evaluation reports them as
separate rows.

**Suggested revision.** Replace the §8.2 sentence with something like:

> The transcript can instead be passed to a local instruction-tuned model. The
> system supports two tiers: an 8B-class model at 4-bit quantisation, which
> runs on a consumer laptop GPU (8 GB VRAM) and is the configuration the
> confidentiality argument of §8.3 concerns; and a 70B-class model at 4-bit,
> which requires a 48 GB-class GPU and is evaluated here only on public court
> audio. The tier is recorded with every output.

And in §8.4, state the consumer-hardware figure directly — it is a stronger
result than the workstation one, not a weaker one.

---

## 2. "Fabrication is impossible" is true of the rule-based mode only (§8.2)

**As written.** §8.2 describes the extractive mode as "a summarizer that, by
construction, cannot fabricate content because every output sentence appears
verbatim in the source." That is correct.

**The gap.** The claim rests on the mode using *no neural model at all*. §8.3
then argues the extractive mode "is arguably the better fit for substantive
work" — but the reason it is the better fit (faithfulness) and the reason it is
worse (frequency-based selection is a poor proxy for legal salience) are never
separated. The paper leaves the reader with a binary: safe and dumb, or fluent
and unreliable.

**That binary is false, and this is the contribution.** Faithfulness comes from
*where the output text originates*, not from *what chose it*. So the
implementation adds a third mode:

> **Model-selected extraction.** A local LLM reads the numbered transcript and
> returns only integer utterance ids. Its response schema — enforced at decode
> time by the inference engine's grammar constraint — contains no free-text
> field for claims. The note text is then re-materialised from the transcript
> by id lookup. Selection is made by a language model; wording cannot be.

The faithfulness guarantee is identical to the rule-based mode (every line is a
byte-exact transcript span, verified without inference in
`src/eval/judge.py::is_verbatim`), while selection quality is that of an 8B
model rather than word-frequency scoring.

It also yields a free hallucination proxy. The model sometimes returns ids that
do not exist in the transcript. Those are dropped and counted
(`validation_warnings`). The rate at which a model invents identifiers, when
identifiers are the only thing it can emit, is a direct and very cheap measure
of its disposition to fabricate — reported in Table 2.

**Suggested revision.** Restructure §8.2 around three modes rather than two,
and rewrite Table 1 accordingly:

| Aspect | Extractive (rule) | Extractive (model-selected) | Abstractive |
|---|---|---|---|
| Transcription | Local Whisper | Local Whisper | Local Whisper |
| Selection | Frequency + regex | Local LLM (ids only) | Local LLM |
| Output text originates from | Transcript | Transcript | Model |
| Data leaves host | No | No | No |
| Fabrication risk | None (structural) | None (structural) | Generative |
| Interpretability | High | High | Lower |

---

## 3. §8.4's "not evaluated" is now addressable (§8.4)

**As written.** "It has *not* been evaluated on legal audio, and no
quantitative accuracy claims are made here. A proper evaluation would measure
transcription word-error-rate on legal speech and assess the faithfulness and
coverage of the generated notes (and the precision/recall of the rule-based
action-item and decision extraction) against expert-annotated gold summaries;
this is left to future work."

**What the harness now does.** Oyez publishes US Supreme Court oral argument
audio aligned to the official transcripts, with per-turn speaker attribution
and timings, under CC-BY-NC. That single source supplies gold references for
two of the three items §8.4 lists, at zero annotation cost:

| §8.4 item | Status | Module |
|---|---|---|
| WER on legal speech | Automated from Oyez | `src/eval/wer.py` |
| Faithfulness of notes | Verbatim rate: verified without a model. Support rate for abstractive: checklist judge | `src/eval/judge.py` |
| Coverage of notes | Content coverage + temporal spread | `src/eval/report.py` |
| P/R of action items and decisions | **Still needs human annotation** | `src/eval/extraction_pr.py` |
| *(new)* Diarization error rate | Automated from Oyez speaker turns | `src/eval/der.py` |

The harness refuses to print zeros for the unannotated category. An annotation
template generator is provided; until it is filled in, Table 3 says so.

**A caveat worth stating in the revision.** SCOTUS oral argument is not
representative of all legal audio: the recording quality is unusually good, the
speakers are unusually articulate, and there is no cross-examination. WER
measured here is a *floor*. Depositions, trial-court proceedings, and
client-intake interviews will be worse. Say this rather than letting a good
number imply general robustness.

---

## 4. New finding to add to §8.4 (and it belongs in §6.1 too)

The evaluation separates overall WER from WER restricted to *legally
load-bearing tokens* — case names, citation components, statutory references,
Latin terms, and mid-sentence proper nouns (`src/util/textnorm.py`).

§8.4 already notes that "ASR errors on legal terminology, party names, and
citations propagate downstream regardless of the note mode." That is stated as
a caveat. It is measurable, and measuring it changes its weight in the
argument: an aggregate WER in the single digits looks reassuring while the
error rate on exactly the tokens a lawyer would cite is several times higher.
Table 1 reports both and their ratio.

This connects directly to §6.1. Dahl et al. measure hallucination in
*generation*. For a speech-first pipeline there is an earlier failure mode: the
model can be perfectly faithful to a transcript that already has the case name
wrong. Extractive summarisation guarantees faithfulness to the transcript, not
to the proceeding. That distinction is not currently drawn anywhere in the
survey and it is worth a paragraph — it is the one substantive limitation the
"fabrication is structurally impossible" framing does not cover.

---

## 5. Smaller corrections

**§8.2, Word export.** The draft says Markdown emphasis is converted "to
bold/underline". The implementation drops underline: in legal documents,
underlining conventionally marks a case name, and overloading it for emphasis
is actively misleading in the one document type where that convention matters.

**§8.1, model selection.** The draft says model size is "selectable (from tiny
through large/turbo)". Worth naming the operating point actually used and why —
`large-v3-turbo` at `int8_float16` is ~2.5 GB resident versus ~4.5 GB for
`large-v3` at fp16, which is what makes the three-model pipeline fit in 8 GB
at all.

**§8.1, live capture.** Implemented as file input only in this version. Either
drop the live-capture sentence or mark it as not evaluated.

**§6.2, judge security.** The survey notes that "prompt-injection studies show
that evaluated text can hijack a judge". The harness implements the defences
(delimiter fencing, explicit instruction hierarchy, a detection canary) *and*
measures whether they hold, by planting attacks in the evaluated content and
counting compliance (`--injection-test`). If you report those numbers, §6.2
gains an empirical sentence instead of only a citation.

---

## Suggested experimental protocol

1. `python src/eval/fetch_oyez.py --term 2019 --limit 10` — ~10 hours of audio.
2. `python run.py <each file> --mode all` on the local tier.
3. Annotate 3–4 cases for action items and decisions (the template generator
   makes this ~1 hour per case).
4. `python src/eval/report.py --gold data/gold --work work --judge`.
5. Re-run the abstractive arm on the hosted tier for the 70B row.
6. `--injection-test` for the §6.2 numbers.

Reportable results, none of which the June draft has:

- WER and legal-term WER on real legal speech, with the ratio between them.
- DER on adversarial multi-speaker audio with constant judicial interruption.
- Verbatim rate by mode — 100% for both extractive modes, verified without a
  model; near zero for abstractive.
- Content coverage and temporal spread, showing whether model-selected
  extraction closes the informativeness gap to abstractive.
- Invalid-uid rate as a fabrication proxy.
- Judge injection-compliance rate.
- Wall clock and peak VRAM per stage on a consumer laptop.
