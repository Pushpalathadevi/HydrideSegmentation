# Vendored KaTeX

KaTeX renders the mathematics on the help page with the same typography as a
LaTeX document. It is vendored here rather than loaded from a CDN because this
application is deployed on air-gapped intranet hosts that cannot reach one, and
because the help page is required to work with no external requests at all.

- **Version:** 0.16.11
- **Upstream:** https://github.com/KaTeX/KaTeX
- **Licence:** MIT, see `LICENSE` in this folder.

## What is here, and what was left out

| File | Purpose |
|---|---|
| `katex.min.css` | Layout and `@font-face` rules |
| `katex.min.js` | The renderer |
| `auto-render.min.js` | Finds `\(...\)` and `\[...\]` in the page and renders them |
| `fonts/*.woff2` | The 20 KaTeX font faces |

Only the **woff2** font format is shipped. Upstream also ships `woff` and `ttf`
copies of every face, which triples the payload to about 1 MB for the benefit of
browsers released before 2014. Those formats are not needed for any browser this
is deployed against, so the `@font-face` rules in `katex.min.css` were edited to
drop their `url(...)` entries. That edit is the **only** modification to any
upstream file.

The `mhchem`, `copy-tex`, and `render-a11y-string` extensions are not used and
were not copied.

## Upgrading

1. `npm install katex@<version> --no-save`
2. Copy `dist/katex.min.css`, `dist/katex.min.js`,
   `dist/contrib/auto-render.min.js`, `dist/fonts/*.woff2`, and `LICENSE` here.
3. Re-apply the woff/ttf strip:

   ```bash
   python -c "import re,pathlib; p=pathlib.Path('katex.min.css'); c=p.read_text(encoding='utf-8'); p.write_text(re.sub(r',url\(fonts/[^)]+\.woff\) format\(\"woff\"\),url\(fonts/[^)]+\.ttf\) format\(\"truetype\"\)','',c), encoding='utf-8', newline='')"
   ```

4. Update the version above and run the test suite. `tests/test_phase35_web_app.py`
   checks that no page references an external host and that every font the CSS
   asks for is actually present, so a partial copy fails there rather than in
   front of a user on a host with no network.
