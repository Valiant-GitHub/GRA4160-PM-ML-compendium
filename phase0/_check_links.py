"""Phase 6.6 — cross-reference / link checker over the rendered _site.

Walks every rendered HTML page, extracts internal href/src targets (skips http,
mailto, #-only same-page handled separately), resolves them relative to the page,
and verifies (a) the target file exists and (b) any #anchor exists as an id in the
target file. Also reports any remote http(s) <script>/<link> (6.5 offline cross-check).
"""
import re
from pathlib import Path
from urllib.parse import urldefrag, unquote

SITE = Path(__file__).resolve().parent.parent / "_site"
html_files = list(SITE.rglob("*.html"))

# Build an id index per file (ids and name anchors)
id_re = re.compile(r'\bid="([^"]+)"')
ids = {}
for f in html_files:
    txt = f.read_text(encoding="utf-8", errors="ignore")
    ids[f.resolve()] = set(id_re.findall(txt))

href_re = re.compile(r'(?:href|src)="([^"]+)"')
broken, remote, ok = [], [], 0
for f in html_files:
    txt = f.read_text(encoding="utf-8", errors="ignore")
    for raw in href_re.findall(txt):
        if raw.startswith(("http://", "https://")):
            # only flag remote <script src> / <link href> (assets); external <a> links are allowed
            # crude check: is this raw inside a script/link tag? approximate by scanning context
            remote.append((f.name, raw))
            continue
        if raw.startswith(("mailto:", "data:", "javascript:", "tel:")):
            continue
        target, frag = urldefrag(raw)
        if target == "":
            # same-page anchor
            if frag and frag not in ids[f.resolve()]:
                broken.append((f.name, raw, "missing same-page anchor"))
            else:
                ok += 1
            continue
        tgt_path = (f.parent / unquote(target)).resolve()
        if not tgt_path.exists():
            broken.append((f.name, raw, "missing file"))
            continue
        if frag:
            if tgt_path.suffix == ".html" and frag not in ids.get(tgt_path, set()):
                broken.append((f.name, raw, f"missing anchor #{frag}"))
                continue
        ok += 1

# Filter remote to only those that look like asset includes (script/link), not <a>
asset_remote = []
for f in html_files:
    txt = f.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r'<(script|link)\b[^>]*\b(?:src|href)="(https?://[^"]+)"', txt):
        asset_remote.append((f.name, m.group(2)))

print(f"HTML pages scanned: {len(html_files)}")
print(f"internal links OK: {ok}")
print(f"BROKEN internal links: {len(broken)}")
for b in broken[:50]:
    print("   ", b)
print(f"REMOTE asset includes (should be 0 for offline): {len(asset_remote)}")
for r in asset_remote[:20]:
    print("   ", r)
