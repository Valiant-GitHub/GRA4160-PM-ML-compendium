"""Phase 3 infra: extract page-mapped tables of contents from ESL & ISL.

Sub-agents use these to jump straight to a section (Rule 1: TOC navigation, not
blind full-text search) before reading the derivation pages for Mode B.
Uses the embedded PDF bookmarks (doc.get_toc()).
"""
from pathlib import Path
import fitz  # pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "phase0"

TARGETS = {
    "esl": "The Elements of Statistical Learning 2.E..pdf",
    "isl": "An Introduction to Statistical Learning Python.pdf",
}

for slug, fn in TARGETS.items():
    path = ROOT / fn
    if not path.exists():
        print(f"{slug}: MISSING {fn}")
        continue
    doc = fitz.open(path)
    toc = doc.get_toc()  # list of [level, title, page]
    lines = [f"# {slug.upper()} — Table of Contents", f"# file: {fn}",
             f"# pages: {doc.page_count}",
             "# Use these page numbers with Read(pages=...) to reach a section, then read the derivation.",
             ""]
    if toc:
        for lvl, title, page in toc:
            indent = "  " * max(0, lvl - 1)
            lines.append(f"{indent}- p{page}: {title}")
    else:
        lines.append("(no embedded bookmarks — fall back to reading the printed TOC pages)")
    (OUT / f"{slug}_toc.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{slug}: {len(toc)} TOC entries across {doc.page_count} pages -> phase0/{slug}_toc.md")
    doc.close()
