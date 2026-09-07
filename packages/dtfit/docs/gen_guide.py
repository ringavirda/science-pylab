"""Render the guide pages of the docs site from their wiki sources.

Run from this directory or the repo root:

    python docs/gen_guide.py          # rewrite the pages
    python docs/gen_guide.py --check  # print a diff per page, change nothing

Each page in PAGES is copied from wiki/, with a note pointing back to the
wiki inserted after the title and every wiki-style link rewritten: a page
name that has a docs counterpart becomes a relative link to it, any other
page name becomes its GitHub wiki URL. Image and external links are left
as they are; the figure files under guide/figures/ are copied by hand.
"""
import difflib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
WIKI = ROOT / "wiki"
DOCS = pathlib.Path(__file__).resolve().parent
WIKI_URL = "https://github.com/ringavirda/science-nonline/wiki/"
REPO_URL = "https://github.com/ringavirda/science-nonline/blob/main/"

# wiki page -> docs page (relative to docs/)
PAGES = {
    "Methods-Image": "guide/image.md",
    "Methods-LSI": "guide/lsi.md",
    "Methods-EAC": "guide/eac.md",
    "Guides-Choosing-a-Method": "guide/choosing-a-method.md",
    "Comparison": "comparison.md",
}
# wiki pages with a docs counterpart that is not generated from the wiki
LINKS = {
    "API-Fitting": "api/batch-fitting.md",
    "API-Models": "api/models.md",
    "API-Auto": "api/forecasting.md",
    "API-Streaming": "api/streaming.md",
    "API-Estimator": "api/sklearn.md",
}
LINK = re.compile(r"\]\(([A-Za-z0-9-]+)(#[^)]*)?\)")
NOTE = (
    "!!! note\n"
    "    Adapted from the project [wiki]({url}). The wiki has the full set of"
    " method, domain and case-study pages.\n"
)
# lines that only make sense on the wiki copy
WIKI_ONLY = ("> Also published in the dtfit docs site",)


def rewrite(text, page, target):
    targets = dict(LINKS)
    targets.update(PAGES)
    here = (DOCS / target).parent

    def sub(m):
        name, anchor = m.group(1), m.group(2) or ""
        if name in targets:
            rel = os.path.relpath(DOCS / targets[name], here)
            return "]({}{})".format(rel, anchor)
        return "]({}{}{})".format(WIKI_URL, name, anchor)

    lines = [l for l in text.splitlines() if not l.startswith(WIKI_ONLY)]
    body = LINK.sub(sub, "\n".join(lines))
    body = body.replace("(" + REPO_URL + "packages/dtfit/docs/gen_comparison.py)",
                        "(gen_comparison.py)")
    title, _, rest = body.partition("\n")
    return "{}\n\n{}{}\n".format(title, NOTE.format(url=WIKI_URL + page), rest.lstrip("\n").rstrip("\n"))


def main(argv):
    check = "--check" in argv
    for page, target in PAGES.items():
        src = WIKI / (page + ".md")
        if not src.exists():
            print("missing {}".format(src))
            continue
        new = rewrite(src.read_text(encoding="utf-8"), page, target)
        dst = DOCS / target
        old = dst.read_text(encoding="utf-8") if dst.exists() else ""
        if new == old:
            print("unchanged {}".format(target))
            continue
        if check:
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(True), new.splitlines(True), target, target + " (rendered)"))
        else:
            dst.write_text(new, encoding="utf-8")
            print("rewrote {}".format(target))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
