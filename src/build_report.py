"""Render report/REPORT.md to a print-ready report/REPORT.html and report/REPORT.pdf.

- `<!-- include-md: tables/x.md -->` and `<!-- include-csv: tables/x.csv | col1,col2 -->` are
  expanded from report/ before rendering (keeps REPORT.md readable, PDF complete).
- Images become numbered <figure>s with their alt text as caption.
- The PDF is printed by headless Google Chrome (A4, page numbers).

Usage: python src/build_report.py [--no-pdf]
"""
import argparse
import html
import re
import shutil
import subprocess
from pathlib import Path

import mistune
import pandas as pd

from common import ROOT

REPORT = ROOT / "report"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm;
        @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #6e6656; } }
@page :first { @bottom-center { content: none; } }
:root { --ink:#1d2433; --soft:#4a4a52; --muted:#6e6656; --navy:#1e3459; --brass:#8f6c1f; --rule:#d9cbaa; --paper2:#f6f1e6; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.6pt; line-height: 1.5; color: var(--ink); margin: 0; }
h1, h2, h3 { font-family: Georgia, serif; color: var(--navy); line-height: 1.25; break-after: avoid; }
h2 { font-size: 15pt; margin: 22pt 0 6pt; padding-bottom: 3pt; border-bottom: 1.5px solid var(--rule); }
h3 { font-size: 12pt; margin: 14pt 0 4pt; }
p, li { text-align: justify; hyphens: auto; }
a { color: var(--navy); }
code { font-family: Menlo, monospace; font-size: 8.6pt; background: var(--paper2); padding: 0 3px; border-radius: 3px; }
pre { background: var(--paper2); border: 1px solid var(--rule); border-radius: 4px; padding: 8px 10px; font-size: 8.4pt; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt; font-size: 8.8pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: var(--navy); color: #fff; font-weight: bold; text-align: left; padding: 4px 6px; }
td { border-bottom: 1px solid #e6dfcf; padding: 3px 6px; vertical-align: top; }
tbody tr:nth-child(even) td { background: #faf7f0; }
.small-table table { font-size: 7.6pt; }
.small-table td, .small-table th { padding: 2px 4px; }
figure { margin: 10pt 0 14pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 112mm; }
figure.screenshot img { max-height: 150mm; border: 1px solid var(--rule); }
figcaption { font-size: 9pt; color: var(--soft); margin-top: 4pt; text-align: center; font-style: italic; }
figcaption b { font-style: normal; color: var(--ink); }
hr { border: 0; border-top: 1px solid var(--rule); }
.page-break { break-after: page; }
.title-page { height: 240mm; display: flex; flex-direction: column; justify-content: center; text-align: center;
              border-top: 4px double var(--brass); border-bottom: 4px double var(--brass); padding: 0 12mm; }
.title-page h1 { font-size: 23pt; margin: 0 0 10pt; }
.title-page h3 { font-weight: normal; font-style: italic; color: var(--soft); margin: 0 0 30pt; font-size: 12.5pt; }
.title-page p { text-align: center; line-height: 1.9; }
"""


def expand_includes(md):
    def inc_md(m):
        return (REPORT / m.group(1).strip()).read_text()

    def inc_csv(m):
        path, cols = [x.strip() for x in m.group(1).split("|")]
        df = pd.read_csv(REPORT / path, dtype=str).fillna("")[cols.split(",")]
        head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
        return head + "\n".join("| " + " | ".join(v.replace("|", "/") for v in r) + " |" for r in df.itertuples(index=False))

    md = re.sub(r"<!--\s*include-md:\s*(.+?)\s*-->", inc_md, md)
    md = re.sub(r"<!--\s*include-csv:\s*(.+?)\s*-->", inc_csv, md)
    return md


def figures(body):
    def fig(m):
        src, alt = m.group(1), html.unescape(m.group(2))
        cls = ' class="screenshot"' if "app_" in src else ""
        cap = re.sub(r"^(Figure \d+\.)", r"<b>\1</b>", html.escape(alt))
        return f'<figure{cls}><img src="{src}" alt="{html.escape(alt)}"><figcaption>{cap}</figcaption></figure>'
    return re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>', fig, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    md = expand_includes((REPORT / "REPORT.md").read_text())
    render = mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])
    body = figures(render(md))
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>IPC Statute Identification: Report</title><style>{CSS}</style></head><body>{body}</body></html>")
    out_html = REPORT / "REPORT.html"
    out_html.write_text(page)
    print("wrote", out_html.relative_to(ROOT))
    if args.no_pdf:
        return
    chrome = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        raise SystemExit("Google Chrome/Chromium not found: open report/REPORT.html in a browser and print to PDF.")
    out_pdf = REPORT / "REPORT.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={out_pdf}", out_html.as_uri()],
                   check=True, capture_output=True, timeout=180)
    print("wrote", out_pdf.relative_to(ROOT), f"({out_pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
