"""
Stage 5 - Render notes to Markdown, plain text, and Word.

One rendering decision is substantive rather than cosmetic. Every note line
carries a `verbatim` flag and a citation. The renderer displays verbatim lines
as quotations with a timestamp, and paraphrased (abstractive) lines in plain
type with a "paraphrase" marker. A reader can therefore tell at a glance which
parts of the document are the record and which parts are a model's account of
the record.

That distinction is the practical form of the survey's Section 6 argument. A
note file that renders both identically is asking the reader to trust a
generative model on a matter of fact, with no way to see where that trust is
being spent.

Run standalone:
    python src/stages/render.py --merged work/h/merged.json \\
        --notes work/h/notes_llm.json --outdir output/hearing
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.schemas import MergedTranscript, NoteItem, NoteMode, NoteSet  # noqa: E402
from src.util.io import ensure_dir, read_json                           # noqa: E402
from src.util.logging import get_logger                                 # noqa: E402

log = get_logger("render")


MODE_BANNER = {
    NoteMode.EXTRACTIVE_RULE: (
        "Extractive (rule-based). Every line below is a verbatim sentence "
        "from the transcript, selected by word-frequency scoring and cue "
        "patterns. No language model was used. Fabrication is structurally "
        "impossible in this mode."
    ),
    NoteMode.EXTRACTIVE_LLM: (
        "Extractive (model-selected). A local language model chose which "
        "transcript lines to surface, but could only return line identifiers; "
        "the text below was re-materialised from the transcript. Every line is "
        "verbatim. Selection may be imperfect; wording cannot be."
    ),
    NoteMode.ABSTRACTIVE: (
        "Abstractive (model-generated). The lines below were WRITTEN by a "
        "local language model and are paraphrases, not quotations. Each "
        "carries citations into the record. Verify against the transcript "
        "before relying on any of it."
    ),
}


def _fmt_time(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def _cite_str(item: NoteItem) -> str:
    if not item.citations:
        return "*(uncited)*"
    parts = [
        f"{c.speaker} @ {_fmt_time(c.start)}" for c in item.citations[:3]
    ]
    more = "" if len(item.citations) <= 3 else f" +{len(item.citations) - 3}"
    return f"[{'; '.join(parts)}{more}]"


def _render_items(items: List[NoteItem], numbered: bool = False) -> List[str]:
    lines = []
    for i, item in enumerate(items, 1):
        bullet = f"{i}." if numbered else "-"
        if item.verbatim:
            lines.append(f'{bullet} "{item.text}" {_cite_str(item)}')
        else:
            lines.append(f"{bullet} {item.text} {_cite_str(item)} *(paraphrase)*")
    return lines or ["- *(none identified)*"]


def to_markdown(merged: MergedTranscript, notes: NoteSet) -> str:
    name = pathlib.Path(merged.audio_path).name
    out: List[str] = []
    out.append(f"# Proceeding Notes - {name}")
    out.append("")
    out.append(f"> **Mode:** `{notes.mode.value}`  ")
    out.append(f"> {MODE_BANNER.get(notes.mode, '')}")
    out.append("")
    out.append(
        f"**Duration:** {_fmt_time(merged.audio_duration)} | "
        f"**Speakers:** {len(merged.speaker_labels)} | "
        f"**Utterances:** {len(merged.utterances)}"
    )
    if notes.model:
        out.append(
            f"**Model:** `{notes.model}` (tier: `{notes.tier}`) | "
            f"**LLM calls:** {notes.llm_calls} | "
            f"**Note stage:** {notes.elapsed_seconds:.1f}s"
        )
    out.append("")

    if notes.validation_warnings:
        out.append("> [!WARNING]")
        out.append(
            f"> The model produced {len(notes.validation_warnings)} output(s) "
            f"that failed validation and were discarded. See "
            f"`validation_warnings` in the notes JSON."
        )
        out.append("")

    out.append("## Summary")
    out.append("")
    out.extend(_render_items(notes.summary))
    out.append("")

    out.append("## Key Points")
    out.append("")
    out.extend(_render_items(notes.key_points))
    out.append("")

    out.append("## Decisions and Rulings")
    out.append("")
    out.extend(_render_items(notes.decisions))
    out.append("")

    out.append("## Action Items and Obligations")
    out.append("")
    out.extend(_render_items(notes.action_items, numbered=True))
    out.append("")

    if notes.keywords:
        out.append("## Keywords")
        out.append("")
        out.append(", ".join(f"`{k}`" for k in notes.keywords))
        out.append("")

    out.append("## Time-stamped Transcript")
    out.append("")
    for u in merged.utterances:
        flag = ""
        if u.confidence is not None and u.confidence < 0.6:
            flag = " ⚠"
        out.append(f"**{u.timestamp()} {u.speaker}{flag}:** {u.text}")
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "*Generated locally. No audio, transcript, or note content was "
        "transmitted to any external service. This document is an aid to "
        "review, not a substitute for the record.*"
    )
    return "\n".join(out)


def to_plain_transcript(merged: MergedTranscript, timestamps: bool) -> str:
    lines = []
    for u in merged.utterances:
        if timestamps:
            lines.append(f"{u.timestamp()} {u.speaker}: {u.text}")
        else:
            lines.append(f"{u.speaker}: {u.text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Word export
# --------------------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"\*(.+?)\*")


def _add_rich(paragraph, text: str) -> None:
    """
    Convert **bold** / *italic* markers into real Word runs.

    Section 8.2 notes that the original prototype converted ** to bold and
    underline; underline is dropped here because in a legal document
    underlining conventionally marks a case name, and overloading it for
    emphasis is actively misleading.
    """
    pos = 0
    pattern = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
    for m in pattern.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        if m.group(1) is not None:
            paragraph.add_run(m.group(1)).bold = True
        else:
            paragraph.add_run(m.group(2)).italic = True
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def to_docx(merged: MergedTranscript, notes: NoteSet, path: pathlib.Path) -> None:
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        log.warning("python-docx not installed; skipping .docx export")
        return

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    doc.add_heading(f"Proceeding Notes - {pathlib.Path(merged.audio_path).name}", 0)

    banner = doc.add_paragraph()
    run = banner.add_run(f"Mode: {notes.mode.value}. ")
    run.bold = True
    banner.add_run(MODE_BANNER.get(notes.mode, ""))
    for r in banner.runs:
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    meta = doc.add_paragraph()
    meta.add_run(
        f"Duration {_fmt_time(merged.audio_duration)}  |  "
        f"{len(merged.speaker_labels)} speakers  |  "
        f"{len(merged.utterances)} utterances"
    ).font.size = Pt(9)
    if notes.model:
        meta.add_run(
            f"  |  Model {notes.model} ({notes.tier})"
        ).font.size = Pt(9)

    if notes.validation_warnings:
        warn = doc.add_paragraph()
        wr = warn.add_run(
            f"⚠ {len(notes.validation_warnings)} model output(s) failed "
            f"validation and were discarded."
        )
        wr.bold = True
        wr.font.color.rgb = RGBColor(0xB0, 0x00, 0x00)
        wr.font.size = Pt(9)

    def section(title: str, items: List[NoteItem], numbered: bool = False) -> None:
        doc.add_heading(title, level=1)
        if not items:
            p = doc.add_paragraph()
            p.add_run("(none identified)").italic = True
            return
        for item in items:
            style_name = "List Number" if numbered else "List Bullet"
            p = doc.add_paragraph(style=style_name)
            if item.verbatim:
                q = p.add_run(f'"{item.text}"')
                q.italic = True
            else:
                _add_rich(p, item.text)
            cite = p.add_run(f"  {_cite_str(item).replace('*', '')}")
            cite.font.size = Pt(8)
            cite.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    section("Summary", notes.summary)
    section("Key Points", notes.key_points)
    section("Decisions and Rulings", notes.decisions)
    section("Action Items and Obligations", notes.action_items, numbered=True)

    if notes.keywords:
        doc.add_heading("Keywords", level=1)
        doc.add_paragraph(", ".join(notes.keywords))

    doc.add_page_break()
    doc.add_heading("Time-stamped Transcript", level=1)
    for u in merged.utterances:
        p = doc.add_paragraph()
        head = p.add_run(f"{u.timestamp()} {u.speaker}: ")
        head.bold = True
        head.font.size = Pt(9)
        p.add_run(u.text)

    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run(
        "Generated locally; no content transmitted externally. "
        "An aid to review, not a substitute for the record."
    )
    fr.italic = True
    fr.font.size = Pt(8)
    fr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    doc.save(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="Render notes")
    ap.add_argument("--merged", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="notes")
    ap.add_argument("--no-docx", action="store_true")
    args = ap.parse_args()

    merged = read_json(args.merged, MergedTranscript)
    notes = read_json(args.notes, NoteSet)
    outdir = ensure_dir(args.outdir)
    prefix = f"{args.prefix}_{notes.mode.value}"

    md_path = outdir / f"{prefix}.md"
    md_path.write_text(to_markdown(merged, notes), encoding="utf-8")

    (outdir / "transcript.txt").write_text(
        to_plain_transcript(merged, timestamps=False), encoding="utf-8"
    )
    (outdir / "transcript_timestamped.txt").write_text(
        to_plain_transcript(merged, timestamps=True), encoding="utf-8"
    )

    written = [md_path.name, "transcript.txt", "transcript_timestamped.txt"]
    if not args.no_docx:
        docx_path = outdir / f"{prefix}.docx"
        to_docx(merged, notes, docx_path)
        if docx_path.exists():
            written.append(docx_path.name)

    log.info("wrote %s to %s", ", ".join(written), outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
