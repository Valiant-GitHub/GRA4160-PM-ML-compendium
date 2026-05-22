"""Phase 0 PDF -> text extraction (pymupdf primary).

Dumps lecture slides, past-exam papers, and guideline PDFs to per-page text
files under phase0/raw_text/ with [PAGE n] markers for citation anchors.
ESL/ISL are intentionally NOT extracted (Phase 0.5: queried per-method later).
Honors C11 (no __MACOSX / ._* inputs exist among these targets anyway).
"""
import re
import sys
from pathlib import Path

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "phase0" / "raw_text"
OUT.mkdir(parents=True, exist_ok=True)

SLIDES = ROOT / "Lecture slides"
EXAMS = SLIDES / "Past exams"


def lecture_targets():
    out = []
    for p in sorted(SLIDES.glob("*.pdf")):
        m = re.match(r"lecture(\d+)", p.name, re.IGNORECASE)
        if m:
            out.append((f"lecture_{int(m.group(1)):02d}", p))
    return out


def exam_targets():
    out = []
    for p in sorted(EXAMS.glob("*.pdf")):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", p.stem).strip("_").lower()
        out.append((f"exam_{slug}", p))
    return out


def extract(slug, path):
    doc = fitz.open(path)
    n = doc.page_count
    chunks = [f"# SOURCE: {path.name}\n# PAGES: {n}\n"]
    nonempty = 0
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text").strip()
        if txt:
            nonempty += 1
        chunks.append(f"\n===== [PAGE {i}] =====\n{txt}\n")
    doc.close()
    dest = OUT / f"{slug}.txt"
    dest.write_text("".join(chunks), encoding="utf-8")
    return n, nonempty, len("".join(chunks))


def main():
    targets = lecture_targets() + exam_targets()
    print(f"Extracting {len(targets)} PDFs -> {OUT}\n")
    for slug, path in targets:
        try:
            n, nonempty, size = extract(slug, path)
            flag = "  <-- LOW TEXT (likely image-only)" if nonempty < max(1, n * 0.3) else ""
            print(f"{slug:32s} pages={n:3d} nonempty={nonempty:3d} chars={size:7d}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{slug:32s} ERROR: {e}", file=sys.stderr)
    print("\nDone.")


if __name__ == "__main__":
    main()
