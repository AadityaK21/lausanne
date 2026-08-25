# Design Principles for Local, Confidentiality-Preserving GenAI Creativity Support Tools in Legal Documentation

*Draft for §8 of the Legal AI survey. Formulated using the design principle
template of Chandra et al. (2015), following the presentation of Herath et al.
(2026).*

---

## 1. Positioning

Herath et al. (2026) derive seven design principles for text-to-image GenAI
creativity support tools (CSTs) in visual design, developed through an action
design research study at Ubisoft. Their artefact, Generative.ConceptCrafter, is
a *bridging artefact*: it interposes process guidance and a simplified interface
between an untamed foundation model (Stable Diffusion) and a production-oriented
creative workflow.

The artefact described here — Lausanne — occupies the same structural position
in a different domain. It bridges untamed open-weight speech and language models
(Whisper, and an instruction-tuned LLM served locally) and the production
workflow of legal documentation: transcribing and summarising hearings,
depositions, client intake, and internal case discussion.

The domains share three of their four framing challenges and diverge sharply on
the rest. Setting out that divergence is what makes a separate set of principles
necessary rather than a restatement of theirs.

### 1.1 Shared challenges

**AI illiteracy and skill gap.** Legal professionals are no more likely than
concept artists to hold the technical knowledge that off-the-shelf GenAI demands.
Local deployment makes this *worse* rather than better: where Herath et al.'s
users faced a complex web interface, a locally hosted stack additionally exposes
CUDA toolkit versions, weight quantisation formats, gated model repositories, and
inference-engine parameters. The skill gap shifts from *using* the tool to
*maintaining* it.

**Professional resistance.** Present in both domains, but differently
constituted. Herath et al. report resistance grounded in deskilling and
displacement anxiety. In law, resistance is additionally grounded in
**liability**: a practitioner who files a brief containing a fabricated citation
faces professional sanction, not merely a poor work product. *Mata v. Avianca*
is the reference case, and it makes legal resistance to GenAI a rational risk
assessment rather than an aversion to be designed around.

**Errors and hallucination.** Shared in name, structurally different in kind.
See §1.3.

### 1.2 A challenge with no analogue: confidentiality

Concept art at Ubisoft is commercially sensitive. Audio of a privileged legal
proceeding is *categorically* restricted: transmitting it to a third-party
service is frequently impermissible regardless of that service's security
posture, because the obligation attaches to disclosure itself and not to the
risk of breach.

This converts a deployment decision into a design constraint. Where Herath et al.
could treat model hosting as an implementation detail, here the topology of
inference is a first-class property of the artefact, and one the user must be
able to demonstrate rather than merely be assured of.

### 1.3 A second error surface

Visual design GenAI has one error surface: the model generates something
incorrect. A speech-first legal pipeline has two.

An automatic speech recogniser can mishear a case name, a citation, a statutory
reference, or a party. Every downstream stage then operates faithfully on a
corrupted record. A summariser that is *provably* faithful to its transcript can
still misattribute a holding, because faithfulness to the transcript is not
faithfulness to the proceeding.

The hallucination literature the survey reviews in §6 — Dahl et al. on
fabrication rates, Magesh et al. on retrieval-augmented commercial tools —
measures the *generative* surface only. The upstream surface is invisible to it,
and no summarisation method can repair it. Principles DP5 and DP6 below exist
because of this.

---

## 2. The principles

Following the template of Chandra et al. (2015):

> Provide the system with **[material property]** in order for users to
> **[action potential]**, given that **[boundary condition]**.

---

### DP1 — Deployment locality and provenance

Provide the system with an inference topology in which every model executes on
hardware under the user's control, and record the execution tier and model
identity on every artefact the system emits **[material property]**, so that
users can *demonstrate* — not merely assert — that privileged material never
left the host, and can segregate work by sensitivity **[action potential]**,
given deployment in domains where disclosure to a third party is prohibited
independently of that party's security posture **[boundary condition]**.

*Rationale.* The confidentiality claim is only as strong as its auditability. A
system that runs locally but does not record where each output was produced
cannot support the claim under scrutiny. Recording the tier also makes a
two-tier deployment safe: a higher-capacity model on rented hardware remains
available for non-privileged material — public court records, training corpora
— without contaminating the guarantee that applies to privileged work.

*Instantiation.* `src/config.py` defines `local` and `hosted` tiers; every
`NoteSet` carries `tier` and `model`. The hosted tier is documented as
public-material-only.

---

### DP2 — Process guidance with environment verifiability

Provide the system with a staged workflow that sequences the generation process,
**and** a diagnostic routine that verifies every external dependency, hardware
budget, and model availability before execution **[material property]**, so that
users can operate the tool without inference-stack expertise and can distinguish
a misconfigured environment from a poor result **[action potential]**, given
locally deployed GenAI CSTs whose correct operation depends on components the
user did not install and cannot inspect **[boundary condition]**.

*Rationale.* This extends Herath et al.'s DP1 (process guidance) with a
condition that arises specifically from local deployment. Their users' failures
were legible: a bad image is visibly bad. A locally hosted stack fails
*illegibly*. On Windows, exceeding available video memory does not raise an
error — the driver silently spills to shared system memory and the pipeline
completes correctly but roughly fifty times slower. A user without diagnostics
attributes this to the model "thinking", and has no path to the actual cause.
Guidance on *how to use* the tool is insufficient when the tool can be
silently broken.

*Instantiation.* `run.py doctor`; the four-stage guidance workflow; per-stage
memory-budget assertions in `src/util/gpu.py`.

---

### DP3-a — Structural constraint of the output vocabulary

Provide the system with a response schema whose expressible vocabulary is
restricted to identifiers of source material, enforced at decode time by the
inference engine's grammar constraint, together with a re-materialisation step
that reconstructs output text from the source by identifier lookup **[material
property]**, so that users obtain model-quality *selection* while remaining
unable to receive model-authored *content* **[action potential]**, given
high-stakes documentation contexts in which an unattributable claim carries
professional or legal consequence **[boundary condition]**.

*Rationale.* This is the principal extension of Herath et al.'s DP2-b. Their
negative prompting operates by *exclusion*: the user enumerates what should not
appear, and the model is steered away from it. Steering is probabilistic and
degrades under distribution shift.

The principle here operates by *construction*. The model's entire output surface
is a set of integers drawn from a list it was given; the schema contains no
free-text field for claims at all. Fabricated content is not improbable — it is
inexpressible. The distinction between a soft constraint on what a model is
*likely* to emit and a structural constraint on what it is *able* to emit is
a nice-to-have in visual design and load-bearing in law.

The principle also yields a measurement at no additional cost. Models
occasionally return identifiers absent from the source. These are rejected and
counted, and the rate constitutes a direct behavioural proxy for a model's
disposition to fabricate under conditions where fabrication is structurally
blocked. See DP5.

*Instantiation.* `SelectionResponse` in `src/schemas.py`; `validate_uids` and
`items_from_uids` in `src/stages/summarize.py`.

---

### DP3-b — Mandatory provenance

Provide the system with a schema in which every emitted claim carries one or
more citations resolving to a located position in the source record — speaker,
timestamp, and identifier **[material property]**, so that users can verify any
individual claim against the record without re-reading the whole of it **[action
potential]**, given documentation practices in which the derived artefact may be
relied upon but the source record remains authoritative **[boundary condition]**.

*Rationale.* Verifiability at the level of the document is not verifiability. A
practitioner asked to confirm a single asserted obligation must be able to reach
the twelve seconds of audio that support it. Making citation a schema
requirement rather than a prompt instruction means an uncited claim is a
validation failure with a recorded warning, not a stylistic lapse that passes
unnoticed.

*Instantiation.* `Citation` and `NoteItem` in `src/schemas.py`; uncited claims
recorded in `validation_warnings`.

---

### DP3-c — Modal disclosure

Provide the system with a rendering that visually and typographically
distinguishes verbatim source material from model-authored paraphrase, and that
states the generation mode and its faithfulness properties on the artefact
itself **[material property]**, so that any reader — including one who did not
operate the tool — can identify which portions of a document are the record and
which are an account of it **[action potential]**, given derived documents that
circulate beyond their author to colleagues, clients, or opposing parties
**[boundary condition]**.

*Rationale.* This principle has no counterpart in Herath et al., and the reason
is instructive: a generated image does not travel with an implicit claim about
its own provenance. A legal note does. A document that renders quotation and
paraphrase identically asks its reader to extend trust without indicating where
that trust is being spent — and the reader of a circulated memo is frequently
not the person who chose the generation mode.

The boundary condition is deliberately about *circulation*, not about
generation. The design obligation arises from the document's onward audience.

*Instantiation.* `verbatim` flag on `NoteItem`; mode banner and differential
rendering in `src/stages/render.py`, in both Markdown and Word output.

---

### DP3-d — One-action verification

Provide the system with an interface in which every emitted claim is a control
that resolves its own citation — scrolling the source record to the cited
passage, marking it, and positioning the recording at the moment it was spoken
**[material property]**, so that users verify a claim by acting on it rather
than by searching for it **[action potential]**, given derived documents whose
value depends on the practitioner actually checking them against a source that
is long, linear, and tedious to search **[boundary condition]**.

*Rationale.* DP3-b makes every claim carry a citation. That is necessary and
not sufficient, because a citation is only as useful as the cost of following
it. A timestamped reference in a Word document requires the reader to open a
media player, scrub to a position, and re-locate the passage in a transcript —
perhaps thirty seconds of friction per claim, across dozens of claims. At that
price verification is theoretically available and practically skipped, which
returns the reader to trusting the model.

Collapsing the same operation into a single click does not add information; it
changes whether the information gets used. This is the difference between a
system that *permits* verification and one whose normal mode of reading *is*
verification.

The boundary condition names the property that makes this worth the
engineering: it applies where the source is costly to search. In a domain where
the source is a single page, it would not be a principle at all.

*Instantiation.* `app/static/index.html` — each note renders as a button whose
handler highlights the cited utterances, scrolls the record to them, and seeks
the audio element; `app/server.py` serves audio with HTTP byte-range support so
the seek is instant rather than requiring the whole file.

---

### DP4 — Graduated autonomy across generation modes

Provide the system with multiple generation modes spanning the range from fully
deterministic extraction to free generation, selectable per task over identical
input and identically structured output **[material property]**, so that users
can match the degree of model autonomy to the risk profile of the specific
document rather than to the tool **[action potential]**, given professional
domains in which the acceptable faithfulness–fluency trade-off varies by
document type within a single user's workflow **[boundary condition]**.

*Rationale.* Herath et al. encounter this trade-off and resolve it by
*compromise*: they acknowledge (§7.1) that simplifying the interface and hard
coding parameters "may curtail radical creativity", and accept the cost. That is
a reasonable resolution when the trade-off is uniform across a user's work.

In legal documentation it is not. A fluent narrative summary of an internal case
discussion and a citable record of a deposition impose different requirements on
the same practitioner in the same week. Fixing the trade-off at design time
forces a tool-level answer to a document-level question. Exposing it as a mode
selection converts an unavoidable compromise into a user judgement — which is
also the concrete form of the augmentation-over-automation commitment that
Raisch and Krakowski (2021) and Berente et al. (2021) argue for.

*Instantiation.* `NoteMode` — `extractive-rule`, `extractive-llm`,
`abstractive` — sharing the `MergedTranscript` input and `NoteSet` output; the
interface exposes them as a selection at run time and as tabs afterwards, so a
practitioner can view the same proceeding under each mode and see what the
trade-off costs on their own material.

---

### DP5 — Verifiable and disaggregated performance metrics

Provide the system with performance metrics that are, wherever the domain
permits, computed *without* model inference against ground truth, reported
separately from model-judged metrics, and disaggregated by the classes of
content on which domain correctness depends **[material property]**, so that
users and their organisations can evaluate the artefact on the dimensions that
carry consequence rather than on aggregates that conceal them **[action
potential]**, given application domains that possess verifiable ground truth and
in which error costs are unevenly distributed across content types **[boundary
condition]**.

*Rationale.* This is the direct strengthening of Herath et al.'s DP4. Their
metrics — accuracy, consistency, visual quality, creativity, originality — could
not be operationalised as in-system scores; they report that the metrics
"served as a basis for qualitative feedback" instead. That limitation is
imposed by the domain: visual design has no ground truth against which
originality can be scored.

Legal documentation does, and the principle should be stated at the strength the
domain allows.

Two components matter. First, **model-free verification where available**:
whether a claim is a byte-exact span of the source is decidable by string
comparison. For the extractive modes this settles faithfulness with no inference
and no evaluator error, and reduces the role of a fallible LLM judge to the one
mode where it is unavoidable. Second, **disaggregation by criticality**: overall
word error rate on courtroom audio is reassuring and misleading. The error rate
restricted to case names, citations, statutory references, and Latin terms runs
substantially higher, and those are precisely the tokens a practitioner would
rely on. An aggregate that averages the tokens that matter together with the
tokens that do not is not a measurement of fitness for legal use.

*Instantiation.* `is_verbatim` in `src/eval/judge.py` (no inference);
`legal_term_wer` in `src/eval/wer.py`; separate reporting of verbatim rate and
judged support rate in `src/eval/report.py`.

---

### DP6 — Adversarially hardened evaluation

Where model-based evaluation is unavoidable, provide the evaluator with a
decomposition into atomic verifiable items rather than a graded score, an
instance drawn from a different model family than the generator, isolation of
evaluated content behind an explicit instruction hierarchy, and a procedure that
plants adversarial instructions in the evaluated content to measure compliance
**[material property]**, so that users can calibrate how much confidence the
evaluation itself warrants **[action potential]**, given evaluation of
open-ended generation in adversarial or high-stakes settings where the evaluated
text is an untrusted input **[boundary condition]**.

*Rationale.* Herath et al.'s DP4 assumes a cooperative evaluation setting. The
LLM-as-a-Judge literature reviewed in §6.2 of this survey establishes that the
setting is not cooperative: pointwise Likert scoring is poorly calibrated and
prompt-sensitive; checklist decomposition (CheckEval) improves interpretability
and reduces susceptibility to stylistic cues; and prompt-injection studies show
that evaluated text can hijack an evaluator outright.

The last point is a structural distinction between evaluation and ordinary
generation, and it has a consequence the survey states but does not act on: an
unreliable generator assessed by an exploitable judge produces a benchmark that
looks healthy while concealing failure. Implementing the defences is standard.
*Measuring whether they hold* — by planting attacks and counting compliance —
converts an assumption into a reported number, and is the difference between
citing the injection literature and answering it.

*Instantiation.* `CHECKLIST`, delimiter fencing, `injection_canary`, and
`--injection-test` in `src/eval/judge.py`; judge model defaulting to a different
family than the generator.

---

### DP6-b — Interface-level enforcement of the confidentiality boundary

Provide the system with a control surface that binds only to the local machine,
refuses to bind a routable interface, and applies the same confidentiality
guard as the command line rather than accepting its own parameters
**[material property]**, so that adding an interface does not add a route
around the artefact's guarantees **[action potential]**, given a tool whose
central claim is a property of *where* computation happens **[boundary
condition]**.

*Rationale.* Herath et al.'s DP5 argues for a streamlined interface that
handles model configuration in the backend. That is right, and in a
confidentiality-bearing system it is incomplete, because an interface is also
new surface area. A graphical front end that accepts a host parameter, or that
listens on all interfaces for convenience, silently converts a local-only tool
into a network service.

The principle is therefore stated as a constraint on interfaces rather than an
endorsement of them: the interface may simplify what the user must decide, and
may not widen what the system will do. The server refuses a non-loopback bind
with an explanation rather than a validation error, and routes every run
through the identical guard the command line uses.

*Instantiation.* `ALLOWED_BINDS` and the refusal message in `app/server.py`;
`prov.guard_tier` called in the job thread before any stage starts;
`tests/test_app.py` asserts both.

---

### DP7 — Resource-bounded stage isolation

Provide the system with an execution model in which each model-bearing stage
runs as an isolated operating-system process with a declared resource ceiling,
communicating only through validated artefacts persisted to disk **[material
property]**, so that users can run a multi-model pipeline on hardware they
already possess, and resume from the point of failure rather than from the
beginning **[action potential]**, given confidentiality requirements that
preclude cloud inference and therefore bound the deployment to commodity or
on-premises hardware **[boundary condition]**.

*Rationale.* DP1 is an aspiration until it is made to fit. The pipeline's three
model stages require roughly 12 GB of accelerator memory in combination against
approximately 7.3 GB available on a current consumer GPU after display
reservation. They cannot be co-resident, and releasing memory reliably is
harder than it appears: common deep-learning runtimes do not return accelerator
memory within a process, the accelerator context itself persists until process
exit, and local inference servers retain models between calls by default.

Process termination is the only eviction primitive that is guaranteed to
succeed. Enforcing it through a strict inter-stage contract yields two
properties beyond memory: recovery, since every intermediate is durable and a
failure in a late stage does not invalidate earlier computation; and
independent testability, since each stage is a function from one validated
artefact to another and can be exercised with synthetic input on hardware that
has no accelerator at all.

This principle is what makes confidentiality-preserving deployment economically
available to a practitioner who is not an enterprise. It is therefore not an
implementation note but a condition of DP1's reach.

*Instantiation.* `run.py` orchestration by subprocess; Pydantic stage contracts
in `src/schemas.py`; `MemoryBudget` in `src/config.py`; 44 offline checks in
`tests/test_pipeline.py`.

---

## 3. Rationalisation and challenge mapping

| DP | Extends / relation to Herath et al. (2026) | Challenge addressed |
|---|---|---|
| DP1 | No analogue — confidentiality is not a design constraint in their setting | Confidentiality; professional resistance |
| DP2 | Extends their DP1 with environment verifiability under local deployment | AI illiteracy and skill gap |
| DP3-a | Strengthens their DP2-b from soft exclusion to structural impossibility | Hallucination; resistance and liability |
| DP3-b | No analogue — generated images do not carry provenance claims | Hallucination; verification burden |
| DP3-d | No analogue — extends DP3-b from citation to one-action verification | Verification burden; resistance |
| DP3-c | No analogue — arises from onward circulation of the artefact | Resistance and liability |
| DP4 | Reframes the simplicity/creativity compromise they accept in §7.1 as a user-facing control | Resistance, deskilling; augmentation over automation |
| DP5 | Strengthens their DP4 to model-free verification and criticality disaggregation | Hallucination; upstream transcription error |
| DP6 | Extends their DP4 into adversarial evaluation settings | Hallucination; evaluator reliability |
| DP6-b | Constrains their DP5: an interface may simplify, not widen | Confidentiality |
| DP7 | No analogue — enabling condition for DP1 on commodity hardware | Confidentiality; skill gap |

### Relationships among the principles

DP1 and DP7 are jointly necessary: locality is unattainable in practice without
resource-bounded staging, and staging is unmotivated without the confidentiality
requirement. DP3-a, DP3-b, and DP3-c form a chain across the artefact's
lifecycle — constrain what can be produced, attach where it came from, disclose
what it is to whoever later reads it. DP2 and DP7 are mutually reinforcing:
declared stage ceilings are what the diagnostic in DP2 verifies against. DP5 and
DP6 are ordered rather than parallel: DP5 removes from DP6's scope every claim
that can be checked without inference, so that the fallible evaluator is applied
only where nothing else can be.

DP4 is the augmentation commitment made operational, and it depends on DP3-c:
offering modes of differing faithfulness is defensible only if the resulting
documents disclose which mode produced them.

---

## 4. Evidential status

Stated plainly, because Herath et al. set a standard these principles do not yet
meet.

Their principles rest on an action design research study with two build–
intervene–evaluate cycles, four FEDS evaluation episodes (Venable et al., 2016),
six internal and six external participants, a seven-evaluator reusability
assessment (Iivari et al., 2021), and institutional ethics approval.

The principles above are derived from artefact construction and technical
evaluation. They have not been subjected to user studies with legal
practitioners. In the terms of Nunamaker et al. (2015), proof of concept is
established; proof of value and proof of use are not.

The asymmetry runs in both directions, and it is worth being precise about it.
Their evidence is strong on acceptance and use and weak on output correctness,
which their domain does not permit them to measure. The evidence here is strong
on output correctness and absent on acceptance. DP2, DP3-c, and DP4 in
particular make claims about user comprehension and judgement that construction
cannot establish and that require practitioners to evaluate.

What would close it: a within-subjects study with practising legal
professionals across the three modes on real proceeding audio, measuring
verification behaviour, time to a usable work product, and stated willingness to
rely on the output; plus a reusability assessment of these principles with
developers of documentation tools in other high-stakes domains — clinical
records, regulatory filings, incident investigation — which share the
confidentiality and provenance conditions and would test the boundary conditions
as stated.

---

## References

Chandra, L., Seidel, S., & Gregor, S. (2015). Prescriptive knowledge in IS
research: Conceptualizing design principles in terms of materiality, action, and
boundary conditions. *48th HICSS*, 4039–4048.

Herath, S., et al. (2026). Design principles for text-to-image generative AI
creativity support tools for visual design: An action design research study at
Ubisoft. *European Journal of Information Systems*, 35(4), 679–704.

Iivari, J., Rotvit Perlt Hansen, M., & Haj-Bolouri, A. (2021). A proposal for
minimum reusability evaluation of design principles. *European Journal of
Information Systems*, 30(3), 286–303.

Nunamaker, J. F., Briggs, R. O., Derrick, D. C., & Schwabe, G. (2015). The last
research mile: Achieving both rigor and relevance in information systems
research. *Journal of Management Information Systems*, 32(3), 10–47.

Sein, M. K., Henfridsson, O., Purao, S., Rossi, M., & Lindgren, R. (2011).
Action design research. *MIS Quarterly*, 35(1), 37–56.

Venable, J., Pries-Heje, J., & Baskerville, R. (2016). FEDS: A framework for
evaluation in design science research. *European Journal of Information
Systems*, 25(1), 77–89.
