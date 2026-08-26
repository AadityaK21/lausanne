"""
Typeset the manuscript in the layout of the reference article.

What is reproduced: the two-column body, the serif type at journal size, the
full-width title block, the ARTICLE INFO / ABSTRACT rule-bounded block,
numbered sections and subsections, figure and table captions in the Elsevier
idiom, and the reference list format.

What is deliberately NOT reproduced: the publisher's logo, journal name,
masthead, DOI, volume and page numbers, and copyright line. Reproducing those
would make a document that reads as an article already published in a journal
it has not been submitted to, which is a false claim about provenance rather
than a matter of formatting. What is produced instead is a submission-ready
manuscript - which is what a journal wants anyway.

Run:
    python paper/figures.py && python paper/manuscript.py
"""

from __future__ import annotations

import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = pathlib.Path(__file__).resolve().parent
FIGS = ROOT / "figures"
OUT = ROOT / "Lausanne_Manuscript.pdf"

# ---- geometry ------------------------------------------------------------
PW, PH = A4
LM = RM = 42
TM, BM = 46, 46
GUTTER = 17
COL_W = (PW - LM - RM - GUTTER) / 2.0
FULL_W = PW - LM - RM

INK = colors.HexColor("#111111")
LINK = colors.HexColor("#2A5DB0")
RULE = colors.HexColor("#000000")
GREY = colors.HexColor("#555555")

ss = getSampleStyleSheet()


def S(name, **kw):
    kw.setdefault("parent", ss["Normal"])
    return ParagraphStyle(name, **kw)


TITLE = S("T", fontName="Times-Roman", fontSize=17, leading=20.5,
          textColor=INK, spaceAfter=11)
AUTHORS = S("A", fontName="Times-Roman", fontSize=10.5, leading=13,
            textColor=INK, spaceAfter=3)
AFFIL = S("Af", fontName="Times-Italic", fontSize=7.6, leading=9.6,
          textColor=INK, spaceAfter=2)

INFO_H = S("IH", fontName="Times-Roman", fontSize=7.4, leading=9.4,
           textColor=INK, spaceAfter=5)
INFO_K = S("IK", fontName="Times-Italic", fontSize=7.4, leading=9.6,
           textColor=INK, spaceAfter=1)
INFO_B = S("IB", fontName="Times-Roman", fontSize=7.4, leading=9.6,
           textColor=INK, spaceAfter=1)
ABS = S("Ab", fontName="Times-Roman", fontSize=8.0, leading=10.2,
        textColor=INK, alignment=TA_JUSTIFY, spaceAfter=4)

H1 = S("H1", fontName="Times-Bold", fontSize=9.4, leading=12, textColor=INK,
       spaceBefore=11, spaceAfter=5)
H2 = S("H2", fontName="Times-Italic", fontSize=9.0, leading=11.5,
       textColor=INK, spaceBefore=8, spaceAfter=4)
BODY = S("B", fontName="Times-Roman", fontSize=8.9, leading=11.0,
         textColor=INK, alignment=TA_JUSTIFY, firstLineIndent=11,
         spaceAfter=0)
BODY0 = S("B0", parent=BODY, firstLineIndent=0)
CAP = S("C", fontName="Times-Roman", fontSize=7.5, leading=9.4, textColor=INK,
        alignment=TA_LEFT, spaceBefore=5, spaceAfter=9)
TBLH = S("TH", fontName="Times-Bold", fontSize=7.4, leading=9.2, textColor=INK)
TBL = S("TB", fontName="Times-Roman", fontSize=7.3, leading=9.2, textColor=INK)
REF = S("R", fontName="Times-Roman", fontSize=7.6, leading=9.4, textColor=INK,
        alignment=TA_JUSTIFY, leftIndent=10, firstLineIndent=-10,
        spaceAfter=2.5)
FOOT = S("F", fontName="Times-Roman", fontSize=7.0, leading=8.8,
         textColor=INK, alignment=TA_JUSTIFY)


def c(text: str) -> str:
    """Colour an in-text citation the way the reference article does."""
    return f'<font color="#2A5DB0">{text}</font>'


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

RUNNING_LEFT = "A. Kumawat"
RUNNING_RIGHT = "Manuscript — Lausanne: structural constraint in legal documentation"


def _furniture(canv, doc, first=False):
    canv.saveState()
    canv.setFont("Times-Roman", 7.4)
    canv.setFillColor(GREY)
    if not first:
        canv.setFont("Times-Italic", 7.4)
        canv.drawString(LM, PH - TM + 14, RUNNING_LEFT)
        canv.drawRightString(PW - RM, PH - TM + 14, RUNNING_RIGHT)
        canv.setStrokeColor(colors.HexColor("#BBBBBB"))
        canv.setLineWidth(0.4)
        canv.line(LM, PH - TM + 10, PW - RM, PH - TM + 10)
    canv.setFont("Times-Roman", 8)
    canv.setFillColor(INK)
    canv.drawCentredString(PW / 2.0, BM - 26, str(doc.page))
    canv.restoreState()


def on_first(canv, doc):
    _furniture(canv, doc, first=True)


def on_later(canv, doc):
    _furniture(canv, doc, first=False)


def build_doc():
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
        title="Lausanne: structural constraint of generative output in "
              "confidentiality-bearing legal documentation",
        author="Aaditya Kumawat",
        subject="Design science research manuscript",
    )

    # Page 1: a full-width block for the title and abstract, then two columns.
    TITLE_H = 352
    f_title = Frame(LM, PH - TM - TITLE_H, FULL_W, TITLE_H, id="title",
                    leftPadding=0, rightPadding=0, topPadding=0,
                    bottomPadding=0)
    body_h = PH - TM - BM - TITLE_H - 10
    f1L = Frame(LM, BM, COL_W, body_h, id="c1", leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0)
    f1R = Frame(LM + COL_W + GUTTER, BM, COL_W, body_h, id="c2",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    # Ordinary pages: two full-height columns.
    full_h = PH - TM - BM
    f2L = Frame(LM, BM, COL_W, full_h, id="c1", leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0)
    f2R = Frame(LM + COL_W + GUTTER, BM, COL_W, full_h, id="c2",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    # Pages carrying a figure that spans both columns: a wide frame on top,
    # two columns beneath, exactly as journals set wide artwork.
    WIDE_H = 330
    fwT = Frame(LM, PH - TM - WIDE_H, FULL_W, WIDE_H, id="wide",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    rest_h = PH - TM - BM - WIDE_H - 10
    fwL = Frame(LM, BM, COL_W, rest_h, id="c1", leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0)
    fwR = Frame(LM + COL_W + GUTTER, BM, COL_W, rest_h, id="c2",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    doc.addPageTemplates([
        PageTemplate(id="first", frames=[f_title, f1L, f1R], onPage=on_first),
        PageTemplate(id="two", frames=[f2L, f2R], onPage=on_later),
        PageTemplate(id="wide", frames=[fwT, fwL, fwR], onPage=on_later),
    ])
    return doc


# --------------------------------------------------------------------------
# helpers for content
# --------------------------------------------------------------------------


def fig(name, caption, width=COL_W, wide=False):
    """A figure scaled to the given width, with an Elsevier-style caption."""
    path = FIGS / name
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scaled_h = width * h / w
    img = Image(str(path), width=width, height=scaled_h)
    img.hAlign = "CENTER"
    block = [img, Paragraph(caption, CAP)]
    return block if wide else [KeepTogether(block)]


def table(caption_no, caption, rows, widths, note=None):
    data = [[Paragraph(str(cell), TBLH if i == 0 else TBL)
             for cell in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        # Elsevier tables: rules above and below the head, and one at the foot.
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, RULE),
    ]))
    # keepWithNext on both caption lines. Without it a caption can be set at
    # the foot of one column with its table at the head of the next, which is
    # how the previous build orphaned Table 2.
    out = [
        Paragraph(f"<b>Table {caption_no}</b>",
                  ParagraphStyle("th", parent=CAP, keepWithNext=1)),
        Paragraph(caption, ParagraphStyle("tc", parent=CAP, spaceBefore=0,
                                          spaceAfter=4, keepWithNext=1)),
        t,
    ]
    if note:
        out.append(Paragraph(note, ParagraphStyle(
            "tn", parent=CAP, fontSize=6.9, leading=8.6, spaceBefore=3)))
    out.append(Spacer(1, 8))
    return out


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------


def story():
    F = []
    A = F.append

    # ---------------- title block (full-width frame on page 1)
    A(Paragraph(
        "Structural constraint of generative output in confidentiality-bearing "
        "legal documentation: A local audio-to-notes system and its design "
        "principles", TITLE))
    A(Paragraph("Aaditya Kumawat", AUTHORS))
    A(Paragraph(
        "Indian Institute of Technology Delhi, Hauz Khas, New Delhi 110016, "
        "India", AFFIL))
    A(Paragraph(
        "In connection with the Applied Artificial Intelligence Laboratory "
        "(AAIL), HEC Lausanne, University of Lausanne, Switzerland", AFFIL))
    A(Spacer(1, 12))

    info = Paragraph(
        "<b>A R T I C L E &nbsp; I N F O</b><br/><br/>"
        "<i>Keywords:</i><br/>"
        "Legal AI<br/>Hallucination<br/>Constrained decoding<br/>"
        "Confidentiality<br/>Design science research<br/>Design principles<br/>"
        "Speech recognition", INFO_B)

    abstract = Paragraph(
        "<b>A B S T R A C T</b><br/><br/>"
        "Generative language models now produce fluent summaries of spoken "
        "proceedings, but in legal work fluency is not the binding constraint: "
        "a summary that invents a holding is worse than none, and audio of a "
        "privileged proceeding often cannot lawfully be sent to a third-party "
        "service at all. Local tools address the second problem and inherit "
        "the first. We report the design and technical evaluation of Lausanne, "
        "an on-device system converting legal proceeding audio into "
        "speaker-attributed, citation-backed notes on consumer hardware. Its "
        "central move is to constrain <i>where output text originates</i> "
        "rather than instruct the model about what to write: in one mode the "
        "response schema exposed to the model contains only integer "
        "identifiers of source utterances and no free-text field for claims, "
        "enforced at decode time, so note text is re-materialised by lookup "
        "and fabrication is inexpressible rather than improbable. We further "
        "show that confidentiality claims must audit model identity and not "
        "only endpoint address, reporting a failure class in which a "
        "cloud-hosted model reached through a loopback URL satisfies a "
        "locality check while transmitting the entire transcript. We "
        "formalise eleven design principles and report a "
        "criticality-disaggregated word error rate substantially higher on "
        "legally load-bearing tokens than in aggregate, identifying an error "
        "surface upstream of the hallucination literature: extraction "
        "guarantees faithfulness to the transcript, not to the proceeding.",
        ABS)

    block = Table([[info, abstract]], colWidths=[COL_W * 0.62, FULL_W - COL_W * 0.62 - 14])
    block.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 14),
        ("LEFTPADDING", (1, 0), (1, 0), 14),
        ("RIGHTPADDING", (1, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.7, RULE),
        ("LINEBEFORE", (1, 0), (1, 0), 0.4, colors.HexColor("#AAAAAA")),
    ]))
    A(block)

    A(NextPageTemplate("two"))   # page 2 onward: two columns, no title frame
    A(FrameBreak())              # into the two columns of page 1

    # ---------------- 1. Introduction
    A(Paragraph("1. Introduction", H1))
    A(Paragraph(
        "Legal work is unusually language-intensive, and much of it consists "
        "of converting speech into a written record: hearings, depositions, "
        "client intake, oral argument, internal case discussion. That work is "
        "slow, repetitive from the practitioner's point of view, and therefore "
        "an obvious candidate for automation with contemporary generative "
        "models.", BODY0))
    A(Paragraph(
        "It is also work in which two constraints bind simultaneously, and the "
        "literature has largely treated them separately. The first is "
        "reliability. " + c("Dahl, Magesh, Suzgun, and Ho (2024)") + " document "
        "hallucination rates of 58–88% on specific, verifiable queries about "
        "federal cases, and show that models frequently fail to flag their own "
        "errors; " + c("Magesh et al. (2025)") + " find that even "
        "retrieval-augmented commercial legal research tools hallucinate "
        "between roughly 17% and 33% of the time. The second is "
        "confidentiality. Audio of a privileged proceeding is categorically "
        "restricted: disclosure to a third party is often impermissible "
        "regardless of that party's security posture, because the obligation "
        "attaches to disclosure itself rather than to the risk of breach.", BODY))
    A(Paragraph(
        "Local deployment answers the second constraint and inherits the "
        "first. A system that runs a language model on the practitioner's own "
        "machine transmits nothing, and still writes whatever the model "
        "produces. The prevailing response is to instruct the model not to "
        "fabricate, and then to measure how often it does anyway.", BODY))
    A(Paragraph(
        "This paper takes a different route. We observe that faithfulness is a "
        "property of <i>where output text originates</i>, not of <i>what "
        "selects it</i>, and that these can be separated. In the system "
        "reported here, one operating mode gives selection to a language model "
        "while restricting its entire output surface to integer identifiers "
        "drawn from a numbered transcript. The response schema contains no "
        "free-text field for claims, and the constraint is enforced at decode "
        "time by the inference engine's grammar. Note text is then "
        "reconstructed from the transcript by identifier lookup. Selection is "
        "as good as the model's judgement; wording cannot be anything other "
        "than source.", BODY))
    A(Paragraph(
        "We make three contributions. First, we describe and formalise this "
        "constraint mechanism, and distinguish it from negative prompting, "
        "which steers a model probabilistically rather than removing the "
        "channel. Second, we report a confidentiality failure mode that "
        "endpoint-level auditing does not catch, in which a cloud-hosted model "
        "reached through a loopback address would pass a locality check while "
        "transmitting the whole transcript; we argue that provenance claims "
        "must audit model identity as well as address. Third, we formalise "
        "eleven design principles in the template of "
        + c("Chandra, Seidel, and Gregor (2015)") + ", and report an "
        "evaluation design in which word error rate is disaggregated by "
        "whether a token is legally load-bearing.", BODY))
    A(Paragraph(
        "Section 2 reviews the relevant literature and locates the "
        "contribution among other ways of bounding generative scope. "
        "Section 3 describes the "
        "artefact. Section 4 states the design principles. Section 5 reports "
        "the evaluation. Section 6 discusses two findings we regard as "
        "generalisable, and Section 7 states limitations, of which the most "
        "important is that no legal practitioner has yet used the system.", BODY))

    # ---------------- 2. Background
    A(Paragraph("2. Background", H1))

    A(Paragraph("2.1. Reliability and hallucination in legal settings", H2))
    A(Paragraph(
        "The reliability problem in legal AI is acute and well documented. "
        "Beyond the rates already cited, " + c("Dahl et al. (2024)") + " note "
        "that models tend to accept a user's incorrect legal premises rather "
        "than correct them, and that the users most likely to rely on such "
        "tools — self-represented litigants — are precisely those least able "
        "to detect the errors. The professional consequences are not "
        "hypothetical: sanctions following the citation of fabricated cases "
        "have made legal caution toward generative systems a rational risk "
        "assessment rather than an aversion to be designed around.", BODY0))
    A(Paragraph(
        "Because human evaluation of open-ended legal generation is expensive, "
        "the field increasingly evaluates with models. That paradigm carries "
        "its own reliability questions. Pointwise scoring is poorly calibrated "
        "and prompt-sensitive; checklist decomposition improves "
        "interpretability and reduces susceptibility to stylistic cues "
        + c("(Lee et al., 2024)") + "; and, structurally distinct from "
        "ordinary generation, the text being evaluated is an untrusted input "
        "that can attempt to hijack the evaluator. An unreliable generator "
        "assessed by an exploitable judge yields a benchmark that looks "
        "healthy while concealing failure.", BODY))
    A(Paragraph(
        "Two further findings bear on any evaluation that delegates judgement "
        "to a model. " + c("Park, Greenewald, Alim, Wang, and Azizan (2025)") +
        " show that reward models used to score intermediate reasoning are "
        "poorly calibrated, and argue for quantifying a model's uncertainty "
        "about its own verdicts rather than consuming those verdicts as "
        "point estimates. And " + c("Bashardoust, Feng, Shrestha, and "
        "colleagues (2026)") + " demonstrate that language models allocating "
        "tasks exhibit systematic bias tied to identity labels attached to the "
        "material, which is to say that what surrounds the content being "
        "judged can move the judgement independently of the content itself.",
        BODY))
    A(Paragraph(
        "We take a conservative reading of this literature. Where a property "
        "can be decided without a model, deciding it without a model removes "
        "calibration error, bias, and injection susceptibility together rather "
        "than mitigating each in turn. Section 5.1 describes the measure we "
        "use for that purpose, and Section 3.3 the design property that makes "
        "such a measure available at all.", BODY))

    A(Paragraph("2.2. Confidentiality as a design constraint", H2))
    A(Paragraph(
        "Work on creativity support tools built over foundation models has "
        "largely assumed that model hosting is an implementation detail. "
        + c("Herath, Sanoubar, Krishna, and Shrestha (2026)") + ", developing "
        "design principles for text-to-image creativity support tools in a "
        "commercial game studio, treat interface simplification and prompt "
        "structure as the principal design surfaces, and confidentiality does "
        "not appear among their boundary conditions. That is appropriate for "
        "their setting: concept art is commercially sensitive, not privileged.",
        BODY0))
    A(Paragraph(
        "Where material is privileged, the topology of inference becomes a "
        "first-class property of the artefact, and one the user must be able "
        "to demonstrate rather than merely be assured of. We take this as the "
        "point of departure for the present work, and return in Section 6.2 to "
        "a specific way in which demonstrating it is harder than it appears.",
        BODY))

    A(Paragraph("2.3. Bounding the scope of generative action", H2))
    A(Paragraph(
        "A useful way to locate this paper's contribution is to ask what a "
        "designer bounds when a generative model is powerful but not "
        "trustworthy. The literature on artificial-intelligence-augmented "
        "decision making " + c("(Shrestha, Ben-Menahem, &amp; von Krogh, 2019; "
        "Raisch &amp; Krakowski, 2021)") + " offers several answers, and they "
        "are not alternatives so much as different axes.", BODY0))
    A(Paragraph(
        "One may bound <i>how much</i> the model decides. Shrestha et al. "
        "(2019) distinguish full human decision making, hybrid arrangements, "
        "and full delegation, and treat the choice among them as an "
        "organisational design question. One may bound <i>when</i> the model "
        "acts: " + c("Tripathi, Sharma, Mishra, Kumar, Herath, and Shrestha "
        "(2026)") + ", designing an LLM-augmented resilience analyst, invoke "
        "generative reasoning only where statistical detection is likely to "
        "fail, and describe this explicitly as preserving analytical rigour by "
        "confining the generative component to the part of the problem that "
        "needs it. One may instead <i>adapt</i> the model, as in the "
        "self-editing approach of " + c("Zweiger, Pari, Guo, Akyürek, Kim, and "
        "Agrawal (2025)") + ", where the model generates its own finetuning "
        "directives.", BODY))
    A(Paragraph(
        "The axis this paper occupies is a fourth: bounding <i>what the model "
        "may emit</i>. Rather than restricting the model's authority, its "
        "moment of involvement, or its weights, we restrict the vocabulary of "
        "its response. The distinction matters because the other three bounds "
        "are enforced around the model — by an orchestrator, a trigger "
        "condition, or a training loop — whereas this one is enforced inside "
        "decoding, and therefore holds without the surrounding system having "
        "to be correct. It is also, unlike adaptation, free: it requires no "
        "additional training and no reward signal.", BODY))
    A(Paragraph(
        "We regard these as complementary. A system may well bound scope on "
        "several axes at once, and Section 4 states principles that do so.",
        BODY))

    A(Paragraph("2.4. Design principles as prescriptive knowledge", H2))
    A(Paragraph(
        "We formalise our design knowledge using the template of "
        + c("Chandra et al. (2015)") + ", in which a principle names a "
        "material property of the artefact, the action potential that property "
        "affords its users, and the boundary conditions under which the design "
        "operates. This follows established practice in design science "
        "research " + c("(Gregor, Chandra Kruse, & Seidel, 2020)") + " and "
        "permits reusability assessment " + c("(Iivari, Hansen, & "
        "Haj-Bolouri, 2021)") + ".", BODY0))
    A(Paragraph(
        "We are explicit that this is design science research and not action "
        "design research " + c("(Sein, Henfridsson, Purao, Rossi, & Lindgren, "
        "2011)") + ". Action design research requires reciprocal shaping "
        "between artefact and organisational context, and its evidence comes "
        "from practitioners using the artefact in their own setting. No legal "
        "practitioner has used this system. Section 7 states what that "
        "costs and what would close it.", BODY))

    # ---------------- 3. Artefact
    A(Paragraph("3. The artefact", H1))
    A(Paragraph(
        "Lausanne converts audio of a legal proceeding into speaker-attributed, "
        "time-stamped, citation-backed notes. Every model executes on the "
        "user's own machine. The reference configuration is a consumer laptop "
        "with an 8 GB GPU and 16 GB of system memory — chosen deliberately, "
        "since a confidentiality guarantee that requires a data-centre GPU is "
        "a weaker guarantee than one that runs on hardware a solo practitioner "
        "already owns.", BODY0))

    A(Paragraph("3.1. Staged execution and the resource budget", H2))
    A(Paragraph(
        "The system comprises five stages (Fig. 1). Each executes as an "
        "isolated operating-system process communicating only through "
        "schema-validated JSON persisted to disk.", BODY0))
    A(Paragraph(
        "This is not architectural preference. The three model-bearing stages "
        "require approximately 12 GB of accelerator memory in combination "
        "against roughly 7.3 GB usable after the display reservation, so they "
        "cannot be co-resident. Releasing that memory is harder than it "
        "appears: common deep-learning runtimes do not return accelerator "
        "memory within a process, the accelerator context persists until the "
        "process exits, and local inference servers retain models between "
        "calls by default. Process termination is the only eviction primitive "
        "that reliably succeeds. Two further properties follow at no cost: "
        "recovery, since every intermediate is durable and a late failure does "
        "not invalidate earlier computation; and independent testability, "
        "since each stage is a function from one validated artefact to "
        "another and can be exercised with synthetic input on hardware with no "
        "accelerator at all.", BODY))
    A(Paragraph(
        "A further consequence deserves emphasis for practitioners. On the "
        "Windows platform, exceeding accelerator memory does not raise an "
        "error; the driver spills to shared system memory and the pipeline "
        "completes correctly but roughly fifty times slower. The failure being "
        "guarded against is therefore not a crash but a silent degradation, "
        "which a user has no obvious means of diagnosing.", BODY))

    for flow in fig("fig1_pipeline.png",
                    "<b>Fig. 1.</b> Staged execution of the pipeline. Each "
                    "stage runs as a separate operating-system process and "
                    "communicates only through schema-validated JSON on disk. "
                    "Combined peak demand exceeds available accelerator "
                    "memory, which is why sequencing rather than co-residence "
                    "is required."):
        A(flow)

    A(Paragraph("3.2. Three modes of note generation", H2))
    A(Paragraph(
        "The system offers three modes over identical input and an identical "
        "output contract (Fig. 2). They differ in one respect: the path by "
        "which text reaches the reader.", BODY0))
    A(Paragraph(
        "<i>Rule-based extraction</i> uses no neural model. Utterances are "
        "scored by damped corpus frequency of their content words and selected "
        "by cue patterns for obligations and rulings. It is fully "
        "deterministic and cannot fabricate, because every emitted line is a "
        "byte-exact sentence of the transcript.", BODY))
    A(Paragraph(
        "<i>Abstractive generation</i> is free generation with enforced "
        "per-claim citation. It is fluent and inherits the full hallucination "
        "risk documented in Section 2.1. It is included because one cannot "
        "argue that extraction is safer without measuring what it is safer "
        "than.", BODY))
    A(Paragraph(
        "<i>Model-selected extraction</i> is the contribution, and is "
        "described separately below.", BODY))

    A(Paragraph("3.3. Structural constraint of the output vocabulary", H2))
    A(Paragraph(
        "Presenting only the first two modes offers the reader a false choice: "
        "extraction that cannot fabricate but selects by word frequency, or "
        "generation that selects well and may invent. The dichotomy dissolves "
        "once faithfulness is recognised as a property of the origin of the "
        "text rather than of the selector.", BODY0))
    A(Paragraph(
        "In model-selected extraction, the model receives a numbered, "
        "speaker-attributed transcript and returns a response conforming to a "
        "schema whose expressible vocabulary is lists of integers. No "
        "free-text field for claims exists anywhere in that schema. The "
        "constraint is enforced at decode time by the inference engine's "
        "grammar, so it is not an instruction the model may disregard under "
        "distribution shift. Note text is then reconstructed from the "
        "transcript by identifier lookup.", BODY))
    A(Paragraph(
        "This differs in kind, not degree, from negative prompting, in which "
        "the user enumerates what should not appear and the model is steered "
        "away from it. Steering is probabilistic. Here the undesired output is "
        "not expressible at any temperature.", BODY))
    A(Paragraph(
        "The design also yields a measurement at no additional cost. Models "
        "occasionally return identifiers absent from the transcript. These are "
        "rejected and counted, and the rate at which a model invents "
        "identifiers — under conditions where identifiers are the only thing "
        "it can emit — is a direct behavioural proxy for its disposition to "
        "fabricate.", BODY))

    for flow in fig("fig2_modes.png",
                    "<b>Fig. 2.</b> The three note-generation modes, which "
                    "differ only in where output text originates. In (b) the "
                    "model's entire response vocabulary is integer "
                    "identifiers; the emitted text is re-materialised from the "
                    "transcript by lookup, so its wording is byte-exact source "
                    "while its selection is the model's. Fabrication is "
                    "inexpressible in (a) and (b), and possible in (c)."):
        A(flow)

    A(Paragraph("3.4. Provenance and enforced confidentiality", H2))
    A(Paragraph(
        "Executing locally is necessary for the confidentiality claim and not "
        "sufficient, because the claim must survive being questioned — by a "
        "supervising partner, an opposing party, or a client. Each run "
        "therefore emits a manifest recording a cryptographic hash of the "
        "input, the model and execution endpoint of every stage, hashes of all "
        "artefacts produced, and a classification of every network endpoint "
        "contacted that separates <i>data egress</i>, in which run content "
        "leaves the machine, from <i>asset retrieval</i>, in which model "
        "weights are fetched.", BODY0))
    A(Paragraph(
        "That separation is easily got wrong and consequential. Downloading "
        "model weights is outbound traffic disclosing nothing about the "
        "proceeding; sending a transcript to a hosted endpoint discloses "
        "everything. A record reporting both as network activity trains its "
        "reader to ignore it.", BODY))
    A(Paragraph(
        "The classification is enforced rather than documented (Fig. 4). A run "
        "declared privileged refuses, before any inference occurs, to execute "
        "against an endpoint that is not on the host — and, for reasons "
        "developed in Section 6.2, against a model that is not on the host "
        "either.", BODY))

    A(Paragraph("3.5. Verification as the normal mode of reading", H2))
    A(Paragraph(
        "Every emitted claim carries a citation resolving to speaker, "
        "timestamp, and utterance identifier. That is necessary and "
        "insufficient, because a citation is worth what it costs to follow. A "
        "timestamped reference in a document requires the reader to open a "
        "player, seek, and re-locate the passage — perhaps thirty seconds per "
        "claim, across dozens of claims. At that price verification is "
        "available in principle and skipped in practice, which returns the "
        "reader to trusting the model.", BODY0))
    A(Paragraph(
        "The system's interface therefore renders every note line as a "
        "control. Selecting it highlights the cited utterances, scrolls the "
        "record to them, and positions the recording at the moment they were "
        "spoken. The change is behavioural rather than informational: it "
        "determines whether checking happens at all. The interface is served "
        "from the loopback interface by a standard-library HTTP server, "
        "refuses to bind a routable address, and routes every run through the "
        "same confidentiality guard as the command line — an interface may "
        "simplify what the user must decide, but must not widen what the "
        "system will do.", BODY))

    # ---------------- 4. Design principles
    A(Paragraph("4. Design principles", H1))
    A(Paragraph(
        "Table 1 states the thirteen principles. Five have no counterpart in "
        "the comparable literature on creativity support tools; three of those "
        "arise from confidentiality and its consequences, one from the fact "
        "that legal documents circulate beyond their author, and one from the "
        "fact that the input is speech attributed to parties whose interests "
        "oppose one another.", BODY0))

    for fl in table(
        "1",
        "Design principles for local, confidentiality-preserving creativity "
        "support tools in legal documentation, with their relation to the "
        "comparable literature.",
        [
            ["DP", "Material property", "Relation to prior work"],
            ["DP1", "On-device inference with recorded execution tier",
             "No counterpart"],
            ["DP2", "Process guidance with environment verifiability",
             "Extends prior process-guidance principle"],
            ["DP3-a", "Response schema restricted to source identifiers",
             "Strengthens negative prompting"],
            ["DP3-b", "Mandatory per-claim provenance", "No counterpart"],
            ["DP3-c", "Modal disclosure of quotation versus paraphrase",
             "No counterpart"],
            ["DP3-d", "One-action verification of any claim",
             "No counterpart"],
            ["DP4", "Graduated autonomy across generation modes",
             "Reframes an accepted compromise"],
            ["DP5", "Model-free, criticality-disaggregated metrics",
             "Strengthens prior metrics principle"],
            ["DP6", "Adversarially hardened evaluation",
             "Extends prior metrics principle"],
            ["DP6-b", "Interface may simplify, not widen",
             "Constrains prior interface principle"],
            ["DP7", "Resource-bounded stage isolation", "No counterpart"],
            ["DP8", "Abstention when a model-based verdict is unstable",
             "Extends prior metrics principle"],
            ["DP9", "Counterfactual auditing of identity labels",
             "No counterpart"],
        ],
        [COL_W * 0.11, COL_W * 0.52, COL_W * 0.37],
        note="Full statements in the Chandra et al. (2015) template — material "
             "property, action potential, boundary condition — are given in the "
             "supplementary material.",
    ):
        A(fl)

    # ---------------- 5. Evaluation
    A(Paragraph("5. Evaluation", H1))

    A(Paragraph("5.1. Data and measures", H2))
    A(Paragraph(
        "Evaluation follows FEDS " + c("(Venable, Pries-Heje, & Baskerville, "
        "2016)") + ". Two of four episodes are complete: component simulation "
        "and technical evaluation against ground truth. The two naturalistic "
        "episodes — a practitioner task study and a principle reusability "
        "assessment — have not been conducted.", BODY0))
    A(Paragraph(
        "Gold data are United States Supreme Court oral arguments published by "
        "Oyez under a CC-BY-NC licence, aligned to the official transcripts "
        "with per-turn speaker attribution and timings. A single corpus "
        "therefore supplies ground truth for both transcription and speaker "
        "diarization, and the material is public, so a higher-capacity model "
        "tier can be benchmarked without privileged content reaching rented "
        "hardware.", BODY))
    A(Paragraph(
        "Measures are: word error rate, and separately word error rate "
        "restricted to legally load-bearing tokens — case names, citation "
        "components, statutory references, Latin terminology, and mid-sentence "
        "proper nouns; diarization error rate, scoring overlapping speech "
        "rather than excluding it, since interruption from the bench is "
        "constant and excluding it would flatter the system on the hardest "
        "material; verbatim rate, the proportion of note lines that are "
        "byte-exact source spans, decided by string comparison with no model "
        "involved; judged support rate for abstractive claims, using a "
        "checklist evaluator with atomic binary items instantiated from a "
        "different model family than the generator; evaluator injection "
        "compliance, measured by planting adversarial instructions in the "
        "evaluated content and counting how often the evaluator obeys; "
        "evaluator abstention rate, the proportion of claims on which the "
        "self-consistency panel split and no verdict was recorded; and "
        "selection stability under relabelling, described in Section 5.3.", BODY))
    A(Paragraph(
        "The verbatim measure deserves note. Because it requires no inference, "
        "it settles faithfulness for the two extractive modes with zero "
        "evaluator error, and confines the fallible model-based judge to the "
        "one mode where nothing else can decide the question.", BODY))
    A(Paragraph(
        "Where the judge cannot be avoided, its verdicts are not consumed as "
        "point estimates. Self-consistency voting damps prompt sensitivity but "
        "discards the dispersion of the samples it aggregates: a unanimous "
        "verdict and a two-to-one verdict emerge from the vote "
        "indistinguishable. Since model-based scorers are known to express "
        "confidence beyond their accuracy " + c("(Park et al., 2025)") + ", "
        "each checklist item carries the fraction of the panel that agreed, "
        "and a claim whose verdict rests on a split panel is recorded as "
        "undecided rather than resolved by one sample. The asymmetry matters: "
        "a claim fails on one stably false item and passes only when every "
        "item passes stably, so abstention can never soften a failure into a "
        "pass. Both denominators are published — the support rate counting "
        "abstentions against the system, and the rate over settled claims "
        "alone — so an evaluator that declines every hard case cannot post a "
        "flattering score unobserved.", BODY))

    A(Paragraph("5.2. Results", H2))
    A(Paragraph(
        "Table 2 reports measures on a 62-minute US Supreme Court oral "
        "argument (<i>New York State Rifle &amp; Pistol Association v. City of "
        "New York</i>, No. 18-280), transcribed and summarised end to end on "
        "the consumer laptop described above. Ten speakers were present and "
        "ten were recovered.", BODY0))

    for fl in table(
        "2",
        "Measures on one hour of Supreme Court oral argument. Verbatim rate is "
        "decided without inference and is a structural property of the "
        "extractive modes, not an empirical finding.",
        [
            ["Measure", "Value"],
            ["Word error rate, all tokens", "6.86%"],
            ["Substitution rate, all tokens", "1.01%"],
            ["Substitution rate, legally load-bearing tokens", "3.07%"],
            ["<b>Ratio</b>", "<b>3.05×</b>"],
            ["Diarization error rate (overlap scored)", "7.92%"],
            ["&nbsp;&nbsp;of which speaker confusion", "0.79%"],
            ["Speakers, reference / recovered", "10 / 10"],
            ["Real-time factor, transcription", "0.016"],
            ["Verbatim rate, rule-based extraction", "100%"],
            ["Verbatim rate, model-selected extraction", "100%"],
            ["Verbatim rate, free generation", "0%"],
            ["Invalid identifiers emitted", "0"],
            ["Judged support, free generation", "21.9%"],
            ["&nbsp;&nbsp;evaluator abstentions", "31.2%"],
        ],
        [COL_W * 0.68, COL_W * 0.32],
    ):
        A(fl)

    for fl in fig("fig3_wer.png",
                  "<b>Fig. 3.</b> The same hour of audio, scored two ways. "
                  "Counting every error class (left), legally load-bearing "
                  "tokens appear marginally safer than average — because "
                  "deletions, which arise from the verbatim transcript "
                  "recording false starts the recogniser drops, swamp the "
                  "count and fall on function words. Counting only "
                  "substitutions (right), where a different word was written, "
                  "the relationship inverts threefold. The left panel is the "
                  "conventional measure and the misleading one."):
        A(fl)

    A(Paragraph(
        "Three results are worth isolating. First, and least expected: "
        "measured the conventional way, legally load-bearing tokens are "
        "<i>no worse</i> than average on this recording — 5.90% against 6.86%. "
        "That figure is an artefact. A verbatim court transcript records false "
        "starts and repetitions which the recogniser silently cleans up, so "
        "deletions dominate the error count — 628 of 797 here — and they fall "
        "on the repeated function words of an interrupted sentence, never on "
        "a case name. Restricting to substitutions, where the model wrote a "
        "<i>different</i> word, the relationship inverts: 1.01% overall "
        "against 3.07% on load-bearing tokens, a factor of three. Aggregate "
        "word error rate does not merely understate this risk; on this corpus "
        "it reverses its sign.", BODY))
    A(Paragraph(
        "The errors themselves are the ones the domain cannot absorb: "
        "<i>Clement</i> transcribed as <i>Coleman</i>, <i>Wall</i> as "
        "<i>Wohl</i>, <i>precedents</i> as <i>precedence</i>. Each is a "
        "fluent, plausible word that a reader skimming a summary would not "
        "flag, and each misidentifies counsel or misstates a doctrine.", BODY))
    A(Paragraph(
        "Second, verbatim rate is 100% for both extractive modes and 0% for "
        "free generation. The first is structural rather than empirical — any "
        "value below 100% there indicates a defect in the re-materialisation "
        "step, and the test suite asserts it as such — but the contrast is the "
        "point. On the same hour of audio, the constrained mode is faithful by "
        "construction while the free mode required an evaluator, and that "
        "evaluator supported 21.9% of its claims.", BODY))
    A(Paragraph(
        "Third, the evaluator abstained on 31.2% of claims, and the "
        "abstentions are not uniformly distributed. Seven of the ten fall on "
        "one checklist item — whether the claim asserts a legal conclusion the "
        "cited lines do not themselves state. That is both the hardest item to "
        "adjudicate and the one whose failure carries the most consequence in "
        "legal work, and a majority vote would have reported all seven as "
        "settled. At this abstention rate the support figure is weak evidence "
        "and is reported as such.", BODY))

    A(Paragraph("5.3. Even-handedness under relabelling", H2))
    A(Paragraph(
        "Every measure above asks whether the notes are correct. None asks "
        "whether they are even-handed. A note set that quotes the bench "
        "flawlessly and omits counsel's strongest point records a perfect "
        "verbatim rate, passes every checklist item, and registers no "
        "transcription or diarization error. It is nonetheless defective for "
        "the use it is put to, and the defect is invisible to the entire "
        "apparatus described in Section 5.1.", BODY0))
    A(Paragraph(
        "Identity labels attached to material are known to move model "
        "decisions independently of the material itself " +
        c("(Bashardoust et al., 2026)") + ". In a transcript the labels are "
        "the speakers, and the resulting question is answerable by "
        "construction: a transcript can be relabelled without altering a byte "
        "of what was said. Selection is therefore re-run over transcripts "
        "whose text is identical and whose speaker labels have been "
        "anonymised, and separately permuted so that the same labels attach to "
        "different people. Any difference in what is selected is attributable "
        "to the labels alone.", BODY))
    A(Paragraph(
        "The measurement carries its own noise floor. Two identical runs of "
        "the same selection do not return the same set, so a stability figure "
        "means nothing on its own; a control condition with labels untouched "
        "is run alongside, and no result counts as evidence unless it exceeds "
        "what resampling alone produced. A control that is itself unstable "
        "therefore raises the threshold rather than lowering it. The reported "
        "determination is gated on movement in the per-speaker distribution "
        "rather than on set overlap, because a selector may exchange half its "
        "sentences while leaving each speaker's share untouched — that is "
        "instability, not bias, and conflating the two would manufacture "
        "findings.", BODY))
    for fl in table(
        "3",
        "Selection under three labellings of one transcript. Text is "
        "byte-identical across conditions and substitute labels are "
        "width-matched, so every condition is split into the same windows.",
        [
            ["Condition", "Jaccard", "Share shift", "Evidence"],
            ["control (labels unchanged)", "1.000", "0.000", "floor"],
            ["anonymised", "0.177", "0.404", "<b>yes</b>"],
            ["permuted", "0.407", "0.036", "no"],
        ],
        [COL_W * 0.40, COL_W * 0.18, COL_W * 0.22, COL_W * 0.20],
        note="Jaccard is overlap with the baseline selection; share shift is "
             "the largest change in any speaker's share of the notes.",
    ):
        A(fl)

    A(Paragraph(
        "The control run returned the identical forty utterances, so on this "
        "generator the noise floor is exactly zero and any movement is "
        "attributable to the labels alone. Both manipulated conditions move "
        "the selection far past it, and they separate in an informative way. "
        "Permuting identities changes <i>which</i> sentences are chosen — "
        "barely two fifths survive — while leaving each speaker's share of the "
        "notes almost untouched. Anonymising them shifts one speaker's share "
        "by forty percentage points, and collapses the note set structurally: "
        "recorded rulings fall from twenty to six, obligations from seven to "
        "three.", BODY))
    A(Paragraph(
        "The selector is therefore not favouring a particular label — the "
        "permutation would have exposed that. It is using speaker identity as "
        "structure, and removing that structure redistributes its attention "
        "wholesale. For an artefact whose purpose is an even-handed record of "
        "an adversarial proceeding, that is a property worth knowing, and no "
        "measure in Section 5.2 would have revealed it: every condition here "
        "scores 100% verbatim with zero invalid identifiers.", BODY))
    A(Paragraph(
        "One case is one case. This is a demonstration that the property is "
        "measurable and that this selector has it, not an estimate of its "
        "magnitude across proceedings.", BODY))

    # ---------------- 6. Discussion
    A(Paragraph("6. Discussion", H1))

    A(Paragraph("6.1. Two error surfaces, not one", H2))
    A(Paragraph(
        "Visual-design applications of generative models have one error "
        "surface: the model produces something wrong. A speech-first legal "
        "pipeline has two. A speech recogniser can mishear a case name, a "
        "citation, or a party, and every downstream stage then operates "
        "faithfully on a corrupted record.", BODY0))
    A(Paragraph(
        "This has an implication the hallucination literature does not reach. "
        "Extraction guarantees faithfulness to <i>the transcript</i>, not to "
        "<i>the proceeding</i>. A structurally unfabricatable summary of a "
        "transcript that misheard a party name is still wrong, and no "
        "summarisation method recovers it. Work measuring fabrication in "
        "generation " + c("(Dahl et al., 2024; Magesh et al., 2025)") + " is "
        "blind to this surface by construction, because it evaluates text "
        "against a correct source. Where the source is itself produced by a "
        "model, the guarantee an extractive method offers is narrower than it "
        "first appears, and criticality-disaggregated error measurement is how "
        "the gap becomes visible.", BODY))

    A(Paragraph("6.2. Endpoint locality is necessary but not sufficient", H2))
    A(Paragraph(
        "During development we identified a confidentiality failure that we "
        "believe generalises. Local inference servers have begun to offer "
        "models that execute on the provider's infrastructure while being "
        "requested through the ordinary local endpoint. The address is "
        "loopback; the computation, and the prompt, are not.", BODY0))
    A(Paragraph(
        "An implementation that audits endpoint address alone — as ours "
        "initially did — will report <i>no data egress</i> about a run that "
        "egressed everything. This is the most dangerous shape a "
        "confidentiality defect can take, because the attestation is not "
        "merely absent but affirmatively and falsely reassuring, and is "
        "therefore trusted. We now treat model identity as part of the "
        "locality check: a run classified privileged refuses a cloud-tagged "
        "model regardless of the URL through which it is reached, and the "
        "manifest records such a model as egress by name.", BODY))
    A(Paragraph(
        "The general lesson is that a provenance claim is only as good as the "
        "property it actually inspects, and that the property which is easy to "
        "inspect is not always the property that matters. We suggest this "
        "belongs in the responsible-deployment literature alongside the more "
        "familiar concerns about training data and retention.", BODY))

    for fl in fig("fig4_guard.png",
                  "<b>Fig. 4.</b> The confidentiality guard as a decision. "
                  "The lower branch is the one that looks safe: a model reached "
                  "at a loopback address that nonetheless executes elsewhere "
                  "would satisfy an endpoint-only check."):
        A(fl)

    A(Paragraph("6.3. Open weights as a precondition, not a preference", H2))
    A(Paragraph(
        "Nothing in this design works with a closed model behind a hosted "
        "API. Constrained decoding requires access to the decoding loop; "
        "on-device execution requires distributable weights; and an "
        "attestation that names the executing model is meaningless if that "
        "model is a remote service whose identity and version the user cannot "
        "verify. " + c("Shrestha, von Krogh, and Feuerriegel (2023)") + " "
        "argue for building AI according to open-source principles on grounds "
        "of accessibility, collaboration, responsibility, and "
        "interoperability. The system reported here is an instance of that "
        "argument in which the dependency is unusually stark: the "
        "confidentiality property is not merely easier to achieve with open "
        "weights, it is unavailable without them.", BODY0))
    A(Paragraph(
        "We note this because the trade-off is usually framed as capability "
        "against openness, with open-weight models the weaker option. In this "
        "setting the framing inverts. A larger closed model cannot deliver the "
        "guarantee at all, so the relevant comparison is not between model "
        "capabilities but between a system that can make a confidentiality "
        "claim and one that cannot.", BODY))

    A(Paragraph("6.4. Implications for practice", H2))
    A(Paragraph(
        "For practitioners, the practical claim is narrow and we prefer to "
        "keep it so. A system of this kind does not remove the obligation to "
        "check its output; it changes the cost of checking, and constrains the "
        "class of error that checking must catch. Where the mode in use makes "
        "fabrication inexpressible, the residual risks are selection quality "
        "and upstream transcription error, both of which a reader can assess "
        "by reading the record. Where free generation is used, the full "
        "hallucination risk returns and the document says so on its face.",
        BODY0))

    # ---------------- 7. Limitations
    A(Paragraph("7. Limitations and future research", H1))
    A(Paragraph(
        "Evidence for the principles is uneven, and the unevenness is "
        "systematic. Those concerning properties verifiable by measurement are "
        "supported by Section 5. Those concerning user comprehension and "
        "behaviour — process guidance, modal disclosure, graduated autonomy, "
        "and one-action verification — are argued from design rationale and "
        "not demonstrated. The claim that one-action verification changes "
        "whether practitioners verify is the sharpest instance: it is an "
        "empirical claim about behaviour that construction cannot establish.",
        BODY0))
    A(Paragraph(
        "The corpus is a floor rather than a representative sample. Supreme "
        "Court oral argument has unusually good recording quality and "
        "unusually articulate speakers, and involves no cross-examination. "
        "Depositions, trial-court proceedings, and client intake will be "
        "worse; reported error rates are a lower bound and should be read as "
        "one.", BODY))
    A(Paragraph(
        "Whether the selected utterances are the <i>right</i> ones requires "
        "annotation by qualified practitioners. The harness generates an "
        "annotation template and declines to report precision and recall until "
        "it is completed, rather than reporting zeros.", BODY))
    A(Paragraph(
        "The configuration is English-language and oriented to United States "
        "appellate practice. Non-Anglophone jurisdictions would require native "
        "audio, native transcription evaluation, and cue patterns rewritten "
        "for local court practice. The principles are stated to hold within "
        "their boundary conditions, not universally.", BODY))
    A(Paragraph(
        "Finally, the attestation is a record of the pipeline's own behaviour "
        "and not a network capture. It cannot speak for an operating-system "
        "telemetry agent or a backup client indexing the working directory. We "
        "document it as such rather than overstating what it establishes.",
        BODY))
    A(Paragraph(
        "The obvious next study is an action design research engagement in a "
        "legal organisation: two build–intervene–evaluate cycles with "
        "practising professionals, a within-subjects comparison across the "
        "three modes on real proceeding audio measuring verification "
        "behaviour, time to a usable work product, and stated willingness to "
        "rely on the output; and a reusability assessment with developers of "
        "documentation tools in adjacent high-stakes domains. Clinical "
        "records, regulatory filings, and incident investigation share the "
        "confidentiality and provenance conditions and would test the boundary "
        "conditions as stated.", BODY))
    A(Paragraph(
        "For the design of that study, " + c("Feng, Feuerriegel, and Shrestha "
        "(2026)") + " provide a directly relevant template: a user study of "
        "how contextualised explanations affect the way people engage with "
        "system output. The question posed by DP3-d — whether reducing the "
        "cost of verification changes whether verification happens — is of the "
        "same form, and would be measured the same way.", BODY))

    # ---------------- 8. Conclusion
    A(Paragraph("8. Conclusion", H1))
    A(Paragraph(
        "Legal AI has moved from feature-based classifiers to general-purpose "
        "models capable of passing the bar examination and, in the same "
        "breath, of inventing the cases they cite. We have argued that for one "
        "important class of legal task the reliability problem admits a "
        "structural rather than a behavioural solution: constrain where output "
        "text can originate, and fabrication ceases to be a risk to be "
        "measured and becomes a state that cannot be reached. We have also "
        "shown that the confidentiality guarantees such systems advertise "
        "depend on auditing the right property, and that the intuitive "
        "property — endpoint address — is not it.", BODY0))
    A(Paragraph(
        "Neither result removes the human professional from the loop, and "
        "neither is intended to. Responsibility for anything filed, advised, "
        "or decided remains where it has always been.", BODY))

    # ---------------- references
    A(Paragraph("References", H1))
    refs = [
        "Bashardoust, A., Feng, Y., Shrestha, Y. R., &amp; colleagues (2026). "
        "When your boss is an AI: Identity label bias in LLM task allocation "
        "decisions. <i>Working paper</i>, University of Lausanne.",

        "Chandra, L., Seidel, S., &amp; Gregor, S. (2015). Prescriptive "
        "knowledge in IS research: Conceptualizing design principles in terms "
        "of materiality, action, and boundary conditions. <i>48th Hawaii "
        "International Conference on System Sciences</i>, 4039–4048.",

        "Dahl, M., Magesh, V., Suzgun, M., &amp; Ho, D. E. (2024). Large legal "
        "fictions: Profiling legal hallucinations in large language models. "
        "<i>Journal of Legal Analysis</i>, 16(1), 64–93.",

        "Gregor, S., Chandra Kruse, L., &amp; Seidel, S. (2020). Research "
        "perspectives: The anatomy of a design principle. <i>Journal of the "
        "Association for Information Systems</i>, 21(6), 1622–1652.",

        "Feng, Y., Feuerriegel, S., &amp; Shrestha, Y. R. (2026). "
        "Contextualizing recommendation explanations with LLMs: A user study. "
        "<i>Proceedings of the Twentieth International AAAI Conference on Web "
        "and Social Media</i>.",

        "Guo, J., Li, N., Xi, M., Yang, W., &amp; Duan, S. (2024). Speaker "
        "diarization: A review of objectives and methods. <i>Applied Soft "
        "Computing</i>, 111238.",

        "Herath, S., Sanoubar, A., Sanoubar, Y., &amp; Shrestha, Y. R. (2026). "
        "Design principles for text-to-image generative AI creativity support "
        "tools for visual design: An action design research study. "
        "<i>European Journal of Information Systems</i>, 35(4), 679–704.",

        "Iivari, J., Rotvit Perlt Hansen, M., &amp; Haj-Bolouri, A. (2021). A "
        "proposal for minimum reusability evaluation of design principles. "
        "<i>European Journal of Information Systems</i>, 30(3), 286–303.",

        "Lee, Y., Kim, J., Kim, S., Cho, H., &amp; Kim, Y. (2024). CheckEval: "
        "A reliable LLM-as-a-judge framework for evaluating text generation "
        "using checklists. <i>arXiv preprint</i> arXiv:2403.18771.",

        "Park, Y.-J., Greenewald, K., Alim, K., Wang, H., &amp; Azizan, N. "
        "(2025). Know what you don't know: Uncertainty calibration of process "
        "reward models. <i>arXiv preprint</i> arXiv:2506.09338.",

        "Raisch, S., &amp; Krakowski, S. (2021). Artificial intelligence and "
        "management: The automation–augmentation paradox. <i>Academy of "
        "Management Review</i>, 46(1), 192–210.",

        "Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., &amp; "
        "Ho, D. E. (2025). Hallucination-free? Assessing the reliability of "
        "leading AI legal research tools. <i>Journal of Empirical Legal "
        "Studies</i>, 22(2), 216–242.",

        "Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., &amp; "
        "Sutskever, I. (2023). Robust speech recognition via large-scale weak "
        "supervision. <i>Proceedings of the 40th International Conference on "
        "Machine Learning</i>, 28492–28518.",

        "Sein, M. K., Henfridsson, O., Purao, S., Rossi, M., &amp; Lindgren, "
        "R. (2011). Action design research. <i>MIS Quarterly</i>, 35(1), 37–56.",

        "Shrestha, Y. R., Ben-Menahem, S. M., &amp; von Krogh, G. (2019). "
        "Organizational decision-making structures in the age of artificial "
        "intelligence. <i>California Management Review</i>, 61(4), 66–83.",

        "Shrestha, Y. R., von Krogh, G., &amp; Feuerriegel, S. (2023). "
        "Building open-source AI. <i>Nature Computational Science</i>, 3, "
        "908–911.",

        "Shrestha, Y. R., Krishna, V., &amp; von Krogh, G. (2021). Augmenting "
        "organizational decision-making with deep learning algorithms: "
        "Principles, promises, and challenges. <i>Journal of Business "
        "Research</i>, 123, 588–603.",

        "Tripathi, P., Sharma, H., Mishra, S., Kumar, B., Herath, S., &amp; "
        "Shrestha, Y. R. (2026). Designing an LLM-augmented resilience analyst "
        "for disruption-driven demand adjustment. <i>PACIS 2026 "
        "Proceedings</i>, 5.",

        "Venable, J., Pries-Heje, J., &amp; Baskerville, R. (2016). FEDS: A "
        "framework for evaluation in design science research. <i>European "
        "Journal of Information Systems</i>, 25(1), 77–89.",

        "Willard, B. T., &amp; Louf, R. (2023). Efficient guided generation "
        "for large language models. <i>arXiv preprint</i> arXiv:2307.09702.",

        "Zweiger, A., Pari, J., Guo, H., Akyürek, E., Kim, Y., &amp; Agrawal, "
        "P. (2025). Self-adapting language models. <i>arXiv preprint</i> "
        "arXiv:2506.10943.",
    ]
    for r in refs:
        A(Paragraph(r, REF))

    return F


def main():
    doc = build_doc()
    doc.build(story())
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
