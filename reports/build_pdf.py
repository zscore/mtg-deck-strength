"""Render a report's Markdown to an arXiv-style PDF: python reports/build_pdf.py "reports/<name>.md" [more.md ...]

Uses pandoc (~/.local/bin) + XeLaTeX from TinyTeX (~/Library/TinyTeX). ```latex code blocks become display math.
Writes the PDF next to the Markdown file.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
PANDOC = shutil.which("pandoc") or str(HOME / ".local/bin/pandoc")
TEX_BIN = HOME / "Library/TinyTeX/bin/universal-darwin"


def build(md: Path) -> Path:
    text = md.read_text()
    text = re.sub(r"```latex\n(.*?)\n```", lambda m: f"$$\n{m.group(1)}\n$$", text, flags=re.S)
    for ch, tex in (("\u2192", r"$\rightarrow$"), ("\u2248", r"$\approx$")):  # glyphs the text font lacks
        text = text.replace(ch, tex)
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()
    meta = []
    # "<date> · <author>" byline right under the title becomes the title block
    by = next((i for i, l in enumerate(lines[1:6], 1) if " · " in l), None)
    if by is not None:
        date, author = (x.strip() for x in lines[by].split(" · ", 1))
        meta = ["-M", f"date={date}", "-M", f"author={author}"]
        lines = lines[:by] + lines[by + 1:]
    body = "\n".join(lines[1:])
    out = md.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "paper.md"
        src.write_text(body)
        cmd = [PANDOC, str(src), "-o", str(out), "--pdf-engine=xelatex",
               "-M", f"title={title}", *meta,
               "-V", "documentclass=article", "-V", "fontsize=10pt", "-V", "geometry:margin=1in",
               "-V", "mainfont=Palatino", "-V", "mathfont=STIX Two Math", "-V", "monofont=Menlo",
               "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
               "-V", "tables=true"]
        env = {"PATH": f"{TEX_BIN}:{Path(PANDOC).parent}:/usr/bin:/bin"}
        subprocess.run(cmd, check=True, env=env)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(build(Path(p)))
