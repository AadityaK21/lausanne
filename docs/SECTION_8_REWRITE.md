# Section 8 — Rewritten

*Replaces §8.1–8.4 of the current draft. Restructured as a design science
contribution carrying design principles, following the presentation of Herath
et al. (2026) while remaining honest about a methodological difference set out
in §8.2.*

---

## 8. Lausanne: A Local, Confidentiality-Preserving CST for Legal Documentation

### 8.1 Motivation

Sections 6 and 7 established the field's central tension: capability without
reliability is insufficient in law, and the governance question has displaced
the performance question. This section grounds that argument in a constructed
artefact.

Legal documentation is an unusually clean site for the problem. The work —
transcribing and summarising hearings, depositions, client intake, oral argument
— is language-intensive, time-consuming, and largely undifferentiated from the
practitioner's perspective, which makes it an obvious automation target. It is
also material on which errors carry professional consequence, and which is
frequently privileged, so the two constraints that §6 and §7 treat separately
apply simultaneously to a single workflow.

Three challenges follow, mirroring those Herath et al. (2026) identify for
visual design, plus one their setting does not encounter.

**AI illiteracy and skill gap.** Legal professionals are no likelier than
concept artists to command the technical apparatus that off-the-shelf GenAI
demands. Local deployment makes this worse, not better: where a hosted service
exposes a complex interface, a locally hosted stack additionally exposes CUDA
toolkit versions, weight quantisation formats, gated model repositories, and
inference-engine parameters. The burden shifts from *using* the tool to
*maintaining* it.

**Professional resistance.** Present in both domains but differently
constituted. Herath et al. report resistance grounded in deskilling and
displacement. In law it is additionally grounded in liability: a practitioner
who files a brief containing a fabricated citation faces sanction, not merely a
weak work product. *Mata v. Avianca* is the reference case, and it makes legal
caution toward GenAI a rational risk assessment rather than an aversion to be
designed around.

**Errors and hallucination, across two surfaces.** Visual design GenAI has one
error surface: the model generates something wrong. A speech-first legal
pipeline has two. An automatic speech recogniser can mishear a case name, a
citation, or a party, and every downstream stage then operates faithfully on a
corrupted record. The hallucination literature reviewed in §6 — Dahl et al.,
Magesh et al. — measures the generative surface only. The upstream surface is
invisible to it and unrepairable by any summarisation method. §8.5 measures it.

**Confidentiality.** Audio of a privileged proceeding is categorically
restricted: disclosure to a third party is frequently impermissible regardless
of that party's security posture, because the obligation attaches to disclosure
itself rather than to the risk of breach. This converts a deployment decision
into a design constraint, and it is the challenge with no counterpart in the
visual design setting.

### 8.2 Method, and a necessary distinction

Herath et al. (2026) derive their principles through action design research
(Sein et al., 2011): an organisational intervention at Ubisoft, two
build–intervene–evaluate cycles shaped by concept artists in their own workflow,
and evaluation with internal and external practitioners.

**This work is not action design research, and should not be presented as
such.** ADR's defining commitment is reciprocal shaping between artefact and
organisational context, and its evidence comes from practitioners using the
artefact in their own setting. No legal practitioner has used Lausanne. Claiming
ADR without the intervention would misrepresent the evidence and would be
visible immediately to any reviewer familiar with the method.

What is claimed instead is design science research in the sense of Hevner et al.
(2004) and Gregor and Hevner (2013): construction of an artefact addressing a
stated class of problems, technical evaluation against ground truth, and
formalisation of the resulting knowledge as design principles using the template
of Chandra et al. (2015).

The distinction matters for what the principles can bear. Principles concerning
*artefact properties that can be verified by measurement* — output faithfulness,
transcription accuracy, resource behaviour, confidentiality topology — are
supported by the evaluation in §8.5. Principles concerning *user comprehension
and judgement* are argued from design rationale and remain to be tested. §8.7
separates them explicitly, and sets out the ADR study that would close the gap.

### 8.3 The artefact

Lausanne occupies the same structural position Herath et al. describe as a
*bridging artefact*: it interposes process guidance, a constrained interaction
design, and a resource-managed execution model between untamed open-weight
foundation models and a production-oriented professional workflow.

It comprises five stages, each executing as an isolated operating-system process
communicating solely through schema-validated artefacts persisted to disk:

1. **Transcription.** Whisper (`large-v3-turbo`, int8 quantised), locally
   executed, with word-level timings and a decoding prompt biased toward legal
   terminology.
2. **Diarization.** `pyannote/speaker-diarization-community-1`, locally
   executed, using its exclusive (non-overlapping) output.
3. **Reconciliation.** Word timings are attributed to speaker turns and
   segmented into sentence-level *utterances*, each carrying a stable integer
   identifier, a speaker, and a timestamp. This is the unit everything
   downstream cites.
4. **Note generation**, in one of three modes (§8.4).
5. **Rendering** to Markdown and Word, with a provenance attestation.

A local application (§8.4.4) provides the operating surface. It is served from
the loopback interface by a standard-library HTTP server and refuses to bind
anything routable.

### 8.4 Three design decisions

Most of the artefact is engineering. Three decisions carry the contribution.

#### 8.4.1 Structural constraint of the output vocabulary

The current draft presents two note modes and, in doing so, offers the reader a
false choice: rule-based extraction that cannot fabricate but selects poorly, or
free generation that reads well and may invent. The dichotomy dissolves once one
observes that faithfulness is a property of *where output text originates*, not
of *what selects it*.

The third mode follows. The model receives a numbered transcript and returns a
response conforming to a schema containing **only lists of integers** — no
free-text field for claims exists anywhere in it — enforced at decode time by the
inference engine's grammar constraint. Note text is then reconstructed from the
transcript by identifier lookup.

Selection is performed by a language model; wording cannot be. This differs in
kind, not degree, from the negative prompting that Herath et al.'s DP2-b employs:
theirs steers a model away from undesired output probabilistically, whereas here
the undesired output is not expressible. The distinction is a refinement in
visual design and load-bearing in law.

The design also yields a measurement at no cost. Models occasionally return
identifiers absent from the source; these are rejected and counted, and the rate
is a direct behavioural proxy for a model's disposition to fabricate under
conditions where fabrication is structurally blocked.

#### 8.4.2 Provenance and demonstrable confidentiality

Running locally is necessary for the confidentiality claim and not sufficient
for it, because the claim must survive being questioned — by a supervising
partner, an opposing party, or a client. "We ran it locally" is an assertion.

Each run therefore emits a manifest recording a cryptographic hash of the input,
the model and execution endpoint of every stage, hashes of all artefacts
produced, and a classification of every network endpoint contacted that
separates **data egress** (run content sent off-host) from **asset retrieval**
(model weights fetched). The separation is essential: downloading Whisper's
weights is outbound traffic disclosing nothing about the proceeding, whereas
sending a transcript to a hosted endpoint discloses everything. A record
reporting both as "network activity" trains its reader to ignore it.

The manifest concludes in a pass/fail attestation, and the classification is
enforced rather than documented: a run declared privileged refuses, before any
inference occurs, to execute against a non-loopback endpoint. This is what makes
the two-tier configuration safe — a higher-capacity model on rented hardware
remains available for public court recordings without contaminating the
guarantee that applies to privileged work.

#### 8.4.3 Modal disclosure

Every note line records whether it is a verbatim source span or model-authored
paraphrase, and the renderer distinguishes them typographically, stating the
generation mode and its faithfulness properties on the document itself.

The rationale is a disanalogy with visual design. A generated image does not
travel with an implicit claim about its own provenance; a legal note does, and
it circulates to colleagues, clients, and occasionally opposing parties. A
document rendering quotation and paraphrase identically asks readers to extend
trust without indicating where it is being spent — and the reader of a forwarded
memo is rarely the person who chose the mode.

#### 8.4.4 Verification as the normal mode of reading

§8.4.3 establishes that every claim carries a citation. That is necessary and
insufficient: a citation is worth what it costs to follow. A timestamped
reference in a document requires the reader to open a player, scrub to a
position, and re-find the passage — perhaps thirty seconds per claim. At that
price verification is available in principle and skipped in practice, which
returns the reader to trusting the model, which is the condition the whole
design exists to avoid.

The interface therefore renders every note line as a control. Selecting it
highlights the cited utterances in the record, scrolls to them, and positions
the recording at the moment they were spoken. The same operation costs one
action instead of thirty seconds, and the change is not informational but
behavioural: it determines whether checking happens at all.

Two properties of the interface are worth stating because they constrain rather
than extend the system. It is served by a standard-library HTTP server with no
web framework and no external assets, so it functions offline and does not
retrieve a stylesheet from a third party on a page whose premise is that
nothing leaves the machine. And it binds the loopback interface only, refusing
any routable address, while routing every run through the same confidentiality
guard the command line uses. Herath et al.'s DP5 argues for interface
simplification, correctly; in a confidentiality-bearing system that argument
needs its complement, since an interface is also new surface area, and one that
accepted a host parameter would convert a local tool into a network service.

### 8.5 Evaluation

Evaluated using FEDS (Venable et al., 2016). Two of four episodes are complete.

| Episode | Type | Question | Status |
|---|---|---|---|
| 1. Component simulation | Formative, artificial | Do the prompting, schema, and parameter configurations function? | Complete — PoC |
| 2. Technical evaluation on public legal audio | Summative, artificial | Is output accurate and faithful against ground truth? | Complete |
| 3. Practitioner task study | Summative, naturalistic | Does it produce value in real legal work? | **Not conducted** |
| 4. Principle reusability | Summative, artificial | Do the principles transfer within the class? | **Not conducted** |

Episode 2 uses US Supreme Court oral argument audio from Oyez (CC-BY-NC),
published aligned to official transcripts with per-turn speaker attribution and
timings. One corpus supplies ground truth for both transcription and
diarization, and the material is public, so the high-capacity tier can be
benchmarked without privileged content reaching rented hardware.

Reported measures:

- **Word error rate**, and separately **word error rate restricted to legally
  load-bearing tokens** — case names, citation components, statutory references,
  Latin terminology, and mid-sentence proper nouns.
- **Diarization error rate**, scoring overlapping speech rather than excluding
  it, since interruption from the bench is constant and excluding it would
  flatter the system on precisely the hardest material.
- **Verbatim rate** — the proportion of note lines that are byte-exact source
  spans, decided by string comparison with no model involved. For the extractive
  modes this settles faithfulness without inference and without evaluator error.
- **Judged support rate** for abstractive claims, using a checklist evaluator
  (Chung et al.'s CheckEval approach) with atomic binary items rather than a
  graded score, instantiated from a different model family than the generator,
  with evaluated content fenced behind an explicit instruction hierarchy.
- **Evaluator injection compliance**, measured by planting adversarial
  instructions in evaluated content and counting how often the evaluator obeys.

The last measure deserves comment. §6.2 observes that evaluated text can hijack
an evaluator, and that an unreliable generator assessed by an exploitable judge
yields a benchmark that looks healthy while concealing failure. Implementing the
standard defences is unremarkable; *measuring whether they hold* converts a
cited concern into a reported number.

**The finding most relevant to §6.** Aggregate word error rate on courtroom
audio is reassuring. Restricted to legally load-bearing tokens it is
substantially higher — and those are precisely the tokens a practitioner would
cite. A system can post a creditable aggregate while mangling case names.

This has an implication the current draft does not draw. Extraction guarantees
faithfulness to *the transcript*, not to *the proceeding*. A structurally
unfabricatable summary of a transcript that misheard a party name is still
wrong, and no downstream method recovers it. This failure mode sits upstream of
the entire hallucination literature reviewed in §6, and it belongs in that
discussion.

### 8.6 Design principles

Eleven principles are formalised using the template of Chandra et al. (2015) —
material property, action potential, boundary condition — and set out in full in
the accompanying design principles document. In summary:

| DP | Concern | Relation to Herath et al. (2026) |
|---|---|---|
| DP1 | Deployment locality and provenance | No analogue |
| DP2 | Process guidance with environment verifiability | Extends their DP1 |
| DP3-a | Structural constraint of output vocabulary | Strengthens their DP2-b |
| DP3-b | Mandatory provenance per claim | No analogue |
| DP3-d | One-action verification | No analogue |
| DP3-c | Modal disclosure to downstream readers | No analogue |
| DP4 | Graduated autonomy across modes | Reframes a compromise they accept |
| DP5 | Verifiable, criticality-disaggregated metrics | Strengthens their DP4 |
| DP6 | Adversarially hardened evaluation | Extends their DP4 |
| DP6-b | Interface may simplify, not widen | Constrains their DP5 |
| DP7 | Resource-bounded stage isolation | No analogue |

The four principles without analogue are where the domain contributes rather
than merely receives. Three arise from confidentiality and its consequences;
one, DP3-c, from the fact that legal documents circulate beyond their author.

### 8.7 Limitations, and what would close them

**Evidence for the principles is uneven, and the unevenness is systematic.**
DP1, DP3-a, DP3-b, DP5, DP6, DP6-b, and DP7 concern properties verifiable by
measurement, and §8.5 supports them. DP2, DP3-c, DP3-d, and DP4 make claims
about user comprehension and behaviour that construction cannot establish. They
are argued, not demonstrated.

DP3-d is the sharpest instance. The claim that one-action verification changes
whether practitioners verify is an empirical claim about behaviour, and it is
exactly the kind of claim an action design research study measures and a build
cannot. It is stated here as a design rationale and flagged as the primary
hypothesis for the study proposed below.

**The corpus is a floor.** Supreme Court oral argument has unusually good
recording quality and unusually articulate speakers, and involves no
cross-examination. Depositions, trial-court proceedings, and client intake will
be worse. Reported error rates are a lower bound and should be read as one.

**Extraction quality is unmeasured against expert judgement.** Whether the
selected utterances are the *right* ones requires annotation by qualified
practitioners. The harness generates an annotation template and declines to
report precision and recall until it is completed, rather than reporting zeros.

**Jurisdiction.** The configuration is English-language and oriented to US
appellate practice. The Indian thread of this survey — ILDC, NyayaAnumana —
would require native audio, native transcription evaluation, and cue patterns
rewritten for Indian court practice. The principles are stated to hold within
their boundary conditions, not universally.

**The route to a stronger claim.** An action design research study in a legal
organisation, following Sein et al. (2011): two build–intervene–evaluate cycles
with practising professionals, FEDS episode 3 as a within-subjects comparison
across the three modes on real proceeding audio — measuring verification
behaviour, time to usable work product, and stated willingness to rely on the
output — and episode 4 as a reusability assessment (Iivari et al., 2021) with
developers of documentation tools in adjacent high-stakes domains. Clinical
records, regulatory filings, and incident investigation share the
confidentiality and provenance conditions and would test the boundary conditions
as stated.

That study, not this section, is what would license the ADR claim.
