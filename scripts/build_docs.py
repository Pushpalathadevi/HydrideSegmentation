"""Build the MicroSeg documentation site.

This helper wraps the Sphinx HTML build, the single-page HTML build, and a
browser-based PDF export path for offline review.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = DOCS / "_build"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _build_html() -> Path:
    out = BUILD / "html"
    _run([sys.executable, "-m", "sphinx", "-b", "html", str(DOCS), str(out)])
    return out


def _build_singlehtml() -> Path:
    out = BUILD / "singlehtml"
    _run([sys.executable, "-m", "sphinx", "-b", "singlehtml", str(DOCS), str(out)])
    return out


def _build_pdf(singlehtml_dir: Path) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "playwright is required for PDF export. Install it and run `python -m playwright install chromium`."
        ) from exc

    html_path = singlehtml_dir / "index.html"
    if not html_path.exists():
        raise FileNotFoundError(f"singlehtml build missing: {html_path}")

    pdf_dir = BUILD / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "microseg-docs.pdf"
    url = html_path.as_uri()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 2000})
        page.goto(url, wait_until="networkidle")
        page.emulate_media(media="print")
        page.pdf(path=str(pdf_path), print_background=True, prefer_css_page_size=True)
        browser.close()

    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MicroSeg documentation site")
    parser.add_argument("--html-only", action="store_true", help="Build only the HTML site")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF generation even when HTML is built")
    args = parser.parse_args()

    html_out = _build_html()
    print(f"html: {html_out}")

    if not args.html_only and not args.skip_pdf:
        singlehtml_out = _build_singlehtml()
        print(f"singlehtml: {singlehtml_out}")
        pdf_path = _build_pdf(singlehtml_out)
        print(f"pdf: {pdf_path}")


if __name__ == "__main__":
    main()
