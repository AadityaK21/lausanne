"""
Build the project report.

Typeset in the same serif system as paper/manuscript.py, and
illustrated with the same generated figures, so the report and the
manuscript read as one body of work rather than two unrelated
deliverables. Single column, because a report is read start to finish
by one person and does not need a journal's column measure.
"""
import pathlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, KeepTogether, HRFlowable, PageBreak,
)

ROOT = pathlib.Path("/home/claude/lausanne")
OUT = ROOT / "docs" / "Lausanne_Project_Report.pdf"

INK      = colors.HexColor("#151A21")
TEXT     = colors.HexColor("#2C343D")
MUTED    = colors.HexColor("#626D79")
RULE     = colors.HexColor("#D5DBE0")
SOFT     = colors.HexColor("#EDF0F2")
ACCENT   = colors.HexColor("#A83A2C")
VERIFIED = colors.HexColor("#17655A")
WARNBG   = colors.HexColor("#FBF2E4")
ACCBG    = colors.HexColor("#FAEFEC")
VERBG    = colors.HexColor("#E8F2F0")

ss = getSampleStyleSheet()

def S(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

TITLE = S("T", fontName="Times-Bold", fontSize=24, leading=28, textColor=INK,
          spaceAfter=4)
SUB   = S("Sub", fontName="Times-Italic", fontSize=12.5, leading=17,
          textColor=MUTED, spaceAfter=16)
META  = S("Meta", fontName="Times-Roman", fontSize=8, leading=12, textColor=MUTED)
H1    = S("H1", fontName="Times-Bold", fontSize=15, leading=19, textColor=INK,
          spaceBefore=17, spaceAfter=7)
H2    = S("H2", fontName="Times-Bold", fontSize=10.8, leading=14,
          textColor=INK, spaceBefore=11, spaceAfter=4)
BODY  = S("B", fontName="Times-Roman", fontSize=10.0, leading=14.4, textColor=TEXT,
          alignment=TA_JUSTIFY, spaceAfter=7)
BULL  = S("Bu", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=4)
CODE  = S("C", fontName="Courier", fontSize=8.2, leading=11.6, textColor=INK,
          backColor=SOFT, borderPadding=6, leftIndent=2, spaceBefore=3,
          spaceAfter=8)
CAP   = S("Cap", fontName="Times-Italic", fontSize=8.2, leading=11,
          textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12)
TH    = S("TH", fontName="Times-Bold", fontSize=8.2, leading=10.8,
          textColor=colors.white)
TD    = S("TD", fontName="Times-Roman", fontSize=8.8, leading=12.0, textColor=TEXT)
TDB   = S("TDB", parent=TD, fontName="Helvetica-Bold", textColor=INK)


def callout(title, body, tone="accent"):
    bg, fg = (ACCBG, ACCENT) if tone == "accent" else (
        (VERBG, VERIFIED) if tone == "good" else (WARNBG, colors.HexColor("#8A5A1B")))
    t = Table([[Paragraph(f'<font color="{fg}"><b>{title}</b></font>',
                          S("ct", fontName="Times-Bold", fontSize=9.0,
                            leading=12, textColor=fg))],
               [Paragraph(body, S("cb", parent=BODY, fontSize=8.8, leading=13,
                                  spaceAfter=0))]],
              colWidths=[168*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("LEFTPADDING", (0,0), (-1,-1), 9), ("RIGHTPADDING", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (0,0), 7), ("BOTTOMPADDING", (0,-1), (-1,-1), 8),
        ("TOPPADDING", (0,1), (-1,-1), 2),
        ("LINEBEFORE", (0,0), (0,-1), 2.2, fg),
    ]))
    return t


def table(rows, widths, header=True):
    data = []
    for i, r in enumerate(rows):
        style = TH if (header and i == 0) else TD
        data.append([Paragraph(str(c), style) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LINEBELOW", (0,0), (-1,-2), 0.4, RULE),
        ("BOX", (0,0), (-1,-1), 0.6, RULE),
    ]
    if header:
        st += [("BACKGROUND", (0,0), (-1,0), INK)]
    t.setStyle(TableStyle(st))
    return t


def page_furniture(canv, doc):
    canv.saveState()
    canv.setStrokeColor(RULE); canv.setLineWidth(0.5)
    canv.line(21*mm, 16*mm, 189*mm, 16*mm)
    canv.setFont("Times-Italic", 7.6); canv.setFillColor(MUTED)
    canv.drawString(21*mm, 11.5*mm, "Lausanne — project report")
    canv.drawRightString(189*mm, 11.5*mm, f"{doc.page}")
    if doc.page > 1:
        canv.setFont("Times-Italic", 7.6)
        canv.drawString(21*mm, 285*mm, "A. Kumawat · IIT Delhi · AAIL, HEC Lausanne")
        canv.setStrokeColor(RULE)
        canv.line(21*mm, 283*mm, 189*mm, 283*mm)
    canv.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4,
                      leftMargin=21*mm, rightMargin=21*mm,
                      topMargin=20*mm, bottomMargin=22*mm,
                      title="Lausanne — Project Report",
                      author="Aaditya Kumawat")
frame = Frame(doc.leftMargin, doc.bottomMargin,
              doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                   onPage=page_furniture)])

F = []
A = F.append

# ---------------------------------------------------------------- cover
A(Paragraph("Lausanne", TITLE))
A(Paragraph("A local, confidentiality-preserving creativity support tool for "
            "legal documentation — and the evidence that it works", SUB))
A(HRFlowable(width="100%", thickness=1.4, color=INK, spaceAfter=8))
A(Paragraph(
    "Aaditya Kumawat &nbsp;·&nbsp; B.Tech, Production &amp; Industrial "
    "Engineering, IIT Delhi<br/>"
    "In connection with the Applied Artificial Intelligence Laboratory (AAIL), "
    "HEC Lausanne, University of Lausanne<br/>"
    "Supervisor: Prof. Yash Raj Shrestha", META))
A(Spacer(1, 14))

A(Paragraph("What this is", H1))
A(Paragraph(
    "Section 8 of the accompanying survey described a prototype that converts "
    "spoken audio into structured notes, and Section 8.4 conceded that it "
    "\"has not been evaluated on legal audio, and no quantitative accuracy "
    "claims are made here.\" This report covers the work that closes that gap: "
    "a rebuilt and instrumented system, an evaluation harness measuring it "
    "against public ground truth, an application through which practitioners "
    "would actually use it, and eleven design principles formalising what the "
    "construction taught.", BODY))
A(Paragraph(
    "Everything runs on one consumer laptop — an RTX 4060 with 8 GB of video "
    "memory and 16 GB of system memory. That constraint is not incidental. The "
    "confidentiality argument is strongest precisely when the hardware is "
    "hardware a solo practitioner already owns, and a guarantee that requires a "
    "data-centre GPU is a weaker guarantee than one that does not.", BODY))

A(callout(
    "The one-sentence version",
    "A language model reads a numbered transcript and returns only integer "
    "line identifiers — its response schema contains no free-text field for "
    "claims at all — so the notes are selected with a model's judgement while "
    "the wording remains byte-exact source. Fabrication is not improbable; it "
    "is inexpressible.", "good"))
A(Spacer(1, 6))

# ---------------------------------------------------------------- built
A(Paragraph("What was built", H1))
A(Paragraph(
    "Roughly 7,000 lines of Python across five subprocess-isolated pipeline "
    "stages, an evaluation harness, and a standard-library application server, "
    "with 219 automated checks that run without a GPU, without models, and "
    "without network access.", BODY))

A(table([
    ["Component", "What it does", "Evidence"],
    ["<b>Pipeline</b>",
     "Transcription, diarization, reconciliation, note generation, rendering "
     "to Word and Markdown. Each stage is a separate process communicating "
     "through validated JSON.",
     "187 checks"],
    ["<b>Three note modes</b>",
     "Rule-based extraction, model-selected extraction, and free generation — "
     "the same input and output contract, differing only in where the output "
     "text originates.",
     "verbatim rate"],
    ["<b>Evaluation harness</b>",
     "Word error rate and legal-term word error rate, diarization error rate, "
     "faithfulness with evaluator abstention, extraction precision and recall, "
     "evaluator injection susceptibility, and a speaker-label bias probe.",
     "Oyez gold data"],
    ["<b>Provenance</b>",
     "Every run emits a hashed manifest and a pass/fail confidentiality "
     "attestation, separating data egress from model-weight retrieval.",
     "verifiable"],
    ["<b>Application</b>",
     "A loopback-only local interface in which every note line resolves its "
     "own citation in one action.",
     "32 checks"],
], [30*mm, 106*mm, 26*mm]))
A(Spacer(1, 10))

# screenshot
shot = ROOT / "docs" / "app.png"
if shot.exists():
    A(KeepTogether([
        Image(str(shot), width=168*mm, height=168*mm*(1880/3120)),
        Paragraph(
            "The application. Selecting any note line highlights the sentence "
            "it came from in the record on the right and positions the "
            "recording at that moment. The badge at top right reports the "
            "run's confidentiality attestation.", CAP),
    ]))

# ---------------------------------------------------------------- contributions
A(Paragraph("The four things that are actually new", H1))

A(Paragraph("1. Faithfulness by construction, not by instruction", H2))
A(Paragraph(
    "The prototype offered a false choice: rule-based extraction that cannot "
    "fabricate but selects badly, or free generation that reads well and may "
    "invent. The dichotomy dissolves once one notices that faithfulness is a "
    "property of <i>where output text originates</i>, not of <i>what selects "
    "it</i>. The middle mode gives the selection to a language model while "
    "restricting its entire output surface to identifiers drawn from a list it "
    "was given, enforced at decode time by the inference engine's grammar "
    "constraint. The note text is then reconstructed from the transcript by "
    "lookup.", BODY))
A(Paragraph(
    "This differs in kind, not degree, from the negative prompting used in "
    "comparable work: steering a model away from unwanted output is "
    "probabilistic and degrades under distribution shift, whereas an output "
    "the schema cannot express is unavailable at any temperature. The design "
    "also yields a free measurement — models occasionally return identifiers "
    "that do not exist, and that rate is a direct behavioural proxy for "
    "fabrication.", BODY))

A(Paragraph("2. The error that sits upstream of the hallucination literature", H2))
A(Paragraph(
    "The evaluation separates overall word error rate from the rate restricted "
    "to <i>legally load-bearing tokens</i> — case names, citation components, "
    "statutory references, Latin terminology, mid-sentence proper nouns. The "
    "aggregate is reassuring; the restricted figure is several times higher, "
    "and those are exactly the tokens a practitioner would cite.", BODY))
modes = ROOT / "paper" / "figures" / "fig2_modes.png"
if modes.exists():
    A(KeepTogether([
        Image(str(modes), width=168*mm, height=168*mm*(1320/2160)),
        Paragraph(
            "The three note modes, which differ only in where output text "
            "originates. In the middle mode the model can emit nothing but "
            "identifiers, so the wording is byte-exact source.", CAP),
    ]))

A(callout(
    "A limitation the survey does not currently state",
    "Extraction guarantees faithfulness to the <i>transcript</i>, not to the "
    "<i>proceeding</i>. A structurally unfabricatable summary of a transcript "
    "that misheard a party name is still wrong, and no summarisation method "
    "recovers it. Published work on legal hallucination measures generation; "
    "for a speech-first system there is an earlier failure surface that is "
    "invisible to it.", "accent"))
A(Spacer(1, 4))

A(Paragraph("3. Confidentiality that is demonstrated rather than asserted", H2))
A(Paragraph(
    "Running locally is necessary for the confidentiality claim and not "
    "sufficient, because the claim must survive being questioned — by a "
    "supervising partner, an opposing party, or a client. Each run emits a "
    "manifest recording a hash of the input, the model and execution endpoint "
    "of every stage, hashes of all artefacts, and a classification of every "
    "network endpoint contacted that separates <b>data egress</b> from "
    "<b>asset retrieval</b>.", BODY))
A(Paragraph(
    "That separation is the part most easily got wrong. Downloading model "
    "weights is outbound traffic that discloses nothing about a proceeding; "
    "sending a transcript to a hosted endpoint discloses everything. A record "
    "reporting both as \"network activity\" trains its reader to ignore it. "
    "The classification is enforced rather than documented: a run declared "
    "privileged refuses, before any inference begins, to execute against a "
    "non-loopback endpoint.", BODY))

A(Paragraph("4. Asking whether the notes are even-handed, not only correct", H2))
A(Paragraph(
    "Every other measurement in the project asks whether the notes are "
    "accurate. None of them asks whether they are fair. A note set that quotes "
    "the bench flawlessly and omits counsel's strongest point records a "
    "perfect verbatim rate, passes every checklist item the evaluator applies, "
    "and registers no transcription or diarization error. For a lawyer it is "
    "still a bad set of notes, and nothing in the harness would have said so.",
    BODY))
A(Paragraph(
    "Recent work on identity-label bias finds that a label attached to "
    "material moves a model's decision independently of the material. In a "
    "transcript the labels are the speakers, and the question turns out to be "
    "answerable by construction: a transcript can be relabelled without "
    "altering a byte of what was said. Selection is re-run over transcripts "
    "whose text is identical and whose speaker labels have been anonymised, "
    "and separately permuted so the same labels sit on different people. Any "
    "difference in what is selected is caused by the labels.", BODY))
A(Paragraph(
    "The design commitment that matters here is the control condition. A run "
    "with the labels untouched is measured alongside the manipulated ones, "
    "and nothing counts as evidence unless it exceeds what resampling alone "
    "produced. On this generator the control returned the identical forty "
    "utterances, so the noise floor is exactly zero and any movement is "
    "attributable to the labels.", BODY))
A(table([
    ["Condition", "Jaccard vs baseline", "Share shift", "Evidence"],
    ["control — labels unchanged", "1.000", "0.000", "floor"],
    ["anonymised", "0.177", "0.404", "<b>yes</b>"],
    ["permuted", "0.407", "0.036", "no"],
], [52*mm, 40*mm, 34*mm, 36*mm]))
A(Spacer(1, 8))
A(Paragraph(
    "Both manipulations move the selection far past the floor, and they "
    "separate informatively. Permuting identities changes <i>which</i> "
    "sentences are chosen while leaving each speaker's share of the notes "
    "almost untouched — so the selector is not simply favouring whoever wears "
    "a particular label. Anonymising them shifts one speaker's share by forty "
    "percentage points and collapses the notes structurally: recorded rulings "
    "fall from twenty to six, obligations from seven to three. The selector "
    "uses speaker identity as structure, and removing it redistributes its "
    "attention wholesale.", BODY))
A(Paragraph(
    "Every condition above scores 100% verbatim with zero invalid "
    "identifiers. Nothing else in the harness would have shown this.", BODY))
A(callout(
    "Why this one may travel furthest",
    "The other three contributions are shaped by legal work. This one applies "
    "wherever a model selects from material attributed to identified people "
    "whose interests diverge — clinical handover between teams, incident "
    "reviews, performance calibration, grievance records. The manipulation is "
    "cheap and the measurement is decisive precisely because the content can "
    "be held byte-identical.", "accent"))
A(Spacer(1, 10))

# ---------------------------------------------------------------- evidence
A(Paragraph("What the evidence supports, and what it does not", H1))
A(Paragraph(
    "Stated plainly, because the comparable published work in this lab sets a "
    "standard this does not yet meet. That work derives its principles through "
    "action design research: an organisational intervention, two "
    "build–intervene–evaluate cycles shaped by practitioners in their own "
    "workflow, and evaluation with internal and external participants.", BODY))
A(Paragraph(
    "<b>This is not action design research and should not be presented as "
    "such.</b> No legal practitioner has used this system. What is claimed is "
    "design science research: construction, technical evaluation against "
    "ground truth, and formalisation as design principles.", BODY))

A(table([
    ["Evaluation episode", "Question", "Status"],
    ["1 · Component simulation",
     "Do the schema, prompting, and parameter configurations function?",
     "<b>Complete</b>"],
    ["2 · Technical evaluation",
     "Is output accurate and faithful against public ground truth?",
     "<b>Complete</b>"],
    ["3 · Practitioner study",
     "Does it produce value in real legal work?",
     "Not conducted"],
    ["4 · Principle reusability",
     "Do the principles transfer within the class?",
     "Not conducted"],
], [42*mm, 96*mm, 24*mm]))
A(Spacer(1, 8))

A(Paragraph(
    "The asymmetry runs in both directions and is worth being precise about. "
    "The comparable work is strong on acceptance and use and weak on output "
    "correctness, which its domain does not permit it to measure — visual "
    "design has no ground truth against which originality can be scored. This "
    "work is the reverse. Nine of the thirteen principles concern properties "
    "verifiable by measurement and are supported; four make claims about user "
    "comprehension and behaviour that construction cannot establish.", BODY))
A(callout(
    "The sharpest instance",
    "The claim that one-action verification changes whether practitioners "
    "actually verify is an empirical claim about behaviour. It is exactly what "
    "a practitioner study measures and a build cannot, and it is the primary "
    "hypothesis for the study proposed below.", "warn"))

A(PageBreak())

# ---------------------------------------------------------------- next
A(Paragraph("What remains", H1))

A(Paragraph("Immediate — days", H2))
for txt in [
    "Annotate three or four proceedings for obligations and rulings. The "
    "harness generates a template and deliberately refuses to report precision "
    "and recall until it is filled in, rather than printing zeros.",
    "Rent a GPU for a day and produce the high-capacity tier row on public "
    "court audio only.",
    "Extend the corpus beyond the single argument reported here. One case "
    "demonstrates that each property is measurable; it does not estimate any "
    "of their magnitudes across proceedings.",
]:
    A(Paragraph(txt, BULL, bulletText="—"))

A(Paragraph("The study that would change the claim — months", H2))
A(Paragraph(
    "An action design research study in a legal organisation: two "
    "build–intervene–evaluate cycles with practising professionals, a "
    "within-subjects comparison across the three modes on real proceeding "
    "audio measuring verification behaviour, time to a usable work product, "
    "and stated willingness to rely on the output; and a reusability "
    "assessment with developers of documentation tools in adjacent "
    "high-stakes domains. Clinical records, regulatory filings, and incident "
    "investigation share the confidentiality and provenance conditions and "
    "would test the boundary conditions as stated. Ethics approval would be "
    "required — the same committee that approved the comparable study.", BODY))

A(Paragraph("Known limitations", H2))
for txt in [
    "<b>The corpus is a floor.</b> Supreme Court oral argument has unusually "
    "good recording quality and unusually articulate speakers. Depositions and "
    "trial-court audio will be worse; reported error rates are a lower bound.",
    "<b>Jurisdiction.</b> The configuration is English-language and oriented to "
    "US appellate practice. The Indian thread of the survey would need native "
    "audio, native evaluation, and cue patterns rewritten for Indian court "
    "practice.",
    "<b>The attestation is a record, not a capture.</b> It describes the "
    "pipeline's own behaviour. It cannot speak for an operating-system "
    "telemetry agent or a backup client indexing the working directory, and it "
    "is documented as such.",
]:
    A(Paragraph(txt, BULL, bulletText="—"))

# ---------------------------------------------------------------- running
A(Paragraph("Running it", H1))
A(Paragraph("Nothing needs to be installed to see the application work:", BODY))
A(Paragraph("python app/server.py --demo", CODE))
A(Paragraph(
    "Demo mode substitutes fixtures for the two GPU stages only; "
    "reconciliation, extraction, and rendering are the production code paths. "
    "The full setup — CUDA-enabled PyTorch, the gated diarization models, and "
    "the local language models — is documented step by step, including the "
    "two failure modes that cost the most time: installing dependencies in the "
    "wrong order, which silently yields a CPU-only build, and the missing "
    "cuDNN library that the transcription engine looks for on the system path "
    "rather than where PyTorch puts it.", BODY))
A(Paragraph(
    "Both test suites run on any machine, with no GPU, no models, and no "
    "network — which is what makes the repository reproducible by a reviewer "
    "who does not own the hardware:", BODY))
# <br/>, not \n: a newline inside a Paragraph is collapsed to a space by
# reportlab, which ran the two commands together on one line.
A(Paragraph("python tests/test_pipeline.py &nbsp;&nbsp;&nbsp;# 187 checks<br/>"
            "python tests/test_app.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 32 checks",
            CODE))

A(Spacer(1, 10))
A(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=7))
A(Paragraph(
    "<i>This software is not legal advice and is not a substitute for the "
    "record. Outputs are an aid to review; responsibility for anything filed, "
    "advised, or decided remains with the human professional.</i>",
    S("d", parent=BODY, fontSize=8, leading=11.5, textColor=MUTED)))

doc.build(F)
print("wrote", OUT, OUT.stat().st_size, "bytes")
