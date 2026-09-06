# Comment Cleanup and Structural Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip changelog-narration and machine-written register from ~10,000 lines of comments and docstrings across three Python packages, then split the five modules that have outgrown a single file.

**Architecture:** Two phases. Phase 1 edits prose only, in every in-scope file, gated by an AST-equality checker that proves no logic moved. Phase 2 restructures five named oversized modules, gated by the test suites. Public API is frozen across both, so `packages/dtfit/docs/api/*.md` needs no edits.

**Tech Stack:** Python 3.10+, pytest, ruff (line-length 79), `ast` from the standard library. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-comment-cleanup-design.md`

## Global Constraints

These apply to every task. Copied verbatim from the spec.

**Prose policy -- cut:**
- Changelog narration: "the old ... veto", "previously only exercised via...", "unchanged from the historical...". Git holds this history.
- Justification of code that was never written: "Kept independent of...", "Deliberately permissive", "so the recommender never silently drops...".
- Marketing register: "answers the question the domain studies say is the whole game", "faithfully mirroring", "the real win".
- Restatement of the line below: `# stderr_ is therefore real & finite` above an assertion on `stderr_`.
- Banner rules: the 255 `# ---------- #` separators, where the function or test name below already carries the label.

**Prose policy -- keep, tightened:**
- `Args:` / `Returns:` / `Raises:` on every docstring that has them. These stay everywhere, not only on published symbols. Made specific and concise: `window_size: Samples per window.` rather than a paragraph on why the default is 50.
- Non-obvious *why*: numerical-stability tricks, the local-ENU-metres choice in `compare_real.py`, the `pyright: ignore` on scipy's untyped `.statistic`.
- Mathematical anchors: the Legendre orthonormal weight `R_j` proportional to `(2j+1)`, quadrature choices, and similar derivation notes.

**Register:**
- The 911 `--` sites are recast to plain punctuation (comma, colon, full stop) in preference to substituting a Unicode em-dash. A dash-heavy register is itself part of what reads as machine-written.
- `**bold**` and `*italic*` are dropped from prose (763 sites).
- Plain declarative sentences.
- Ruff's 79-column limit holds throughout.

**Hard rules:**
- **Nothing in this repository is committed by an implementer or by the
  controller.** No `git add`, no `git commit`, no `git rm`, no branch or tag
  operations. Work is left in the working tree for the repository owner to
  review and commit. This overrides any commit instruction elsewhere in this
  plan or in any skill template.
- Public API is frozen. No renaming, adding, or removing any symbol exported from a package `__init__.py`. If a change appears to require one, stop and raise it.
- Phase 1 changes no logic. `tools/prose_guard.py` must report OK for every file before a Phase 1 commit.
- Out of scope: `dis/`, `papers/` (git-ignored, `.gitignore:189-190`), `wiki/`, Jupyter notebooks, `packages/dtfit/docs/api/*.md`, and the React Native app under `packages/dtfit-hardware/mobile/`.

**Baselines to preserve (recorded 2026-08-27, all green):**

| suite | result | time |
|---|---|---|
| `packages/dtfit/tests` | 906 passed, 2 skipped, 17 xfailed | 43.30s |
| `packages/dtfit-experimental/tests` | 72 passed | 8.72s |
| `packages/dtfit-hardware/tests` | 5 passed | 1.27s |

Ruff, mypy, and the docs build are equally part of this project's definition of green:

| gate | command | run from | baseline |
|---|---|---|---|
| ruff | `python -m ruff check packages/` | repo root | `All checks passed!` |
| mypy | `python -m mypy` | `packages/dtfit/` | `Success: no issues found in 45 source files` |
| docs | `python -m mkdocs build --strict` | `packages/dtfit/` | exit 0 |

**Toolchain gotchas -- read before running anything.**

1. **mypy and mkdocs must be run from `packages/dtfit/`.** Their configuration
   lives in `packages/dtfit/pyproject.toml` (`[tool.mypy] files = ["src/dtfit"]`,
   `ignore_missing_imports = true`). Run mypy from the repo root instead and the
   config is never loaded: it reports 38 phantom `[import-untyped]` errors for
   sympy and scipy.stats. Those are an artifact of the wrong working directory,
   not a regression. Do not chase them, and do not install stubs to silence them.

2. **Always invoke tools as `.venv/Scripts/python.exe -m <tool>`, never as a bare
   `pytest` / `mypy` / `mkdocs` command.** This repo was moved on disk
   (`F:/repos/fallen-traces/science-nonline` -> `F:/repos/science-nonline`).
   The editable-install `.pth` files were repaired, but the
   `.venv/Scripts/*.exe` console-script launchers still embed the old
   interpreter path in their shebang and **fail silently with exit 1**.
   `ruff.exe` is a native binary and survives; nothing else does.

3. **Read pytest's exit status from `PIPESTATUS`**, never from a pipeline ending
   in `tail`. A pipeline reports `tail`'s status, which masked a real pytest
   failure during baselining.

4. **`mkdocs build --strict` is the only gate that catches a malformed
   docstring.** mkdocstrings renders `Args:`/`Returns:`/`Raises:` blocks from
   `packages/dtfit/src`; a broken indent in a rewritten block fails the strict
   build while pytest and ruff both stay green. Every task that edits
   `packages/dtfit/src` must run it.

---

## File Structure

**Created by this plan:**
- `tools/prose_guard.py` -- the AST-equality checker. Repo-root because it spans all three packages.
- `tools/test_prose_guard.py` -- its tests.

**Modified in Phase 1:** every `.py` under `packages/dtfit/src`, `packages/dtfit/tests`, `packages/dtfit-experimental/`, `packages/dtfit-hardware/src`, plus the 14 firmware `.ino`/`.h` files.

**Modified in Phase 2:** the five oversized modules named in Tasks 17-21.

**Deleted:** `img/` (7 tracked files), `__pycache__/_diag_variants.cpython-314.pyc`.

---

## Phase 0 -- Tooling

### Task 1: Build the prose guard

**Files:**
- Create: `tools/prose_guard.py`
- Test: `tools/test_prose_guard.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `strip_prose(src: str) -> str` returning a normalized AST dump with all docstrings removed; `check(before: str, after: str) -> bool`; and a CLI `python tools/prose_guard.py <git-ref> <path>...` exiting 0 when every path is prose-identical to that ref, 1 otherwise. Every Phase 1 task calls the CLI.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the Phase 1 prose guard."""
import textwrap

import pytest

from prose_guard import check, strip_prose


def _d(s):
    return textwrap.dedent(s)


def test_docstring_rewrite_passes():
    before = _d('''
        def f(x):
            """Compute the thing -- and explain **why** at length."""
            return x + 1
    ''')
    after = _d('''
        def f(x):
            """Add one."""
            return x + 1
    ''')
    assert check(before, after)


def test_comment_rewrite_passes():
    before = _d('''
        # The old veto dropped the whole family, which is why we changed it.
        VALUE = 3
    ''')
    after = _d('''
        # Families are kept; AIC decides.
        VALUE = 3
    ''')
    assert check(before, after)


def test_docstring_removal_passes():
    before = _d('''
        def f():
            """Gone."""
            return 1
    ''')
    after = _d('''
        def f():
            return 1
    ''')
    assert check(before, after)


def test_docstring_only_body_passes():
    before = _d('''
        def f():
            """Long explanation -- with **bold**."""
    ''')
    after = _d('''
        def f():
            """Short."""
    ''')
    assert check(before, after)


def test_reflow_passes():
    before = _d('''
        def f(a, b):
            return a + b
    ''')
    after = _d('''
        def f(
            a,
            b,
        ):
            return a + b
    ''')
    assert check(before, after)


def test_logic_change_detected():
    before = "VALUE = 3\n"
    after = "VALUE = 4\n"
    assert not check(before, after)


def test_string_literal_edit_detected():
    """The failure mode a naive `--` codemod would cause."""
    before = 'SEP = " -- "\n'
    after = 'SEP = " — "\n'
    assert not check(before, after)


def test_regex_literal_edit_detected():
    before = 'PAT = re.compile(r"a--b")\n'
    after = 'PAT = re.compile(r"a—b")\n'
    assert not check(before, after)


def test_bare_string_expression_is_not_a_docstring():
    """A string in a non-docstring position is code, so edits must be caught."""
    before = _d('''
        x = 1
        "not a docstring -- just an expression"
    ''')
    after = _d('''
        x = 1
        "not a docstring, just an expression"
    ''')
    assert not check(before, after)


def test_syntax_error_raises():
    with pytest.raises(SyntaxError):
        strip_prose("def f(:\n")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tools/test_prose_guard.py -v`
Expected: FAIL, collection error `ModuleNotFoundError: No module named 'prose_guard'`.

- [ ] **Step 3: Write the implementation**

```python
"""Prove that an edit touched only prose.

Phase 1 of the comment cleanup rewrites comments and docstrings but must not
move any logic. Comments never reach the AST, so parsing both revisions and
comparing them after stripping docstrings is an exact test of that: it passes a
reworded comment or a reflowed signature, and fails an edit that lands inside a
string literal or regex.

Usage::

    python tools/prose_guard.py HEAD packages/dtfit/src
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_SCOPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def strip_prose(src: str) -> str:
    """Return an AST dump of ``src`` with every docstring removed.

    Args:
        src: Python source text.

    Returns:
        A dump with positions omitted, so reflowing does not register as a
        change.

    Raises:
        SyntaxError: If ``src`` does not parse.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, _SCOPES):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        is_doc = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        if is_doc:
            # A scope whose only statement was the docstring needs a filler so
            # the tree stays valid; both revisions get the same one.
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def check(before: str, after: str) -> bool:
    """Report whether two revisions differ only in prose.

    Args:
        before: Source before the edit.
        after: Source after the edit.

    Returns:
        True if the two are identical once comments and docstrings are
        discarded.

    Raises:
        SyntaxError: If either revision does not parse.
    """
    return strip_prose(before) == strip_prose(after)


def _git_show(ref: str, path: Path) -> str | None:
    """Read ``path`` at ``ref``, or None if it did not exist there."""
    done = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    return done.stdout if done.returncode == 0 else None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    ref, targets = argv[0], argv[1:]
    files: list[Path] = []
    for t in targets:
        p = Path(t)
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])

    failed = skipped = 0
    for f in files:
        if "__pycache__" in f.parts:
            continue
        before = _git_show(ref, f)
        if before is None:
            skipped += 1
            continue
        try:
            if not check(before, f.read_text(encoding="utf-8")):
                print(f"LOGIC CHANGED: {f}")
                failed += 1
        except SyntaxError as exc:
            print(f"SYNTAX ERROR:  {f}: {exc}")
            failed += 1

    checked = len(files) - skipped
    if failed:
        print(f"\n{failed} of {checked} file(s) changed more than prose.")
        return 1
    print(f"OK: {checked} file(s) prose-identical to {ref} ({skipped} new).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tools/test_prose_guard.py -v`
Expected: PASS, 10 passed.

Note: pytest inserts the test file's directory into `sys.path` (rootdir auto-insert), so `from prose_guard import ...` resolves. If it does not, run from the repo root with `PYTHONPATH=tools`.

- [ ] **Step 5: Verify the CLI reports a clean tree**

Run: `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src`
Expected: `OK: 44 file(s) prose-identical to HEAD (0 new).`, exit 0.

- [ ] **Step 6: Report, do not commit**

Leave both files in the working tree, unstaged. Report the file list and the
verification output. Committing is the repository owner's call, not yours.

---

## Phase 1 -- Prose

Every Phase 1 task follows the identical shape below. `<PATHS>` is the task's file set and `<SUITE>` its package suite.

1. Rewrite prose in `<PATHS>` per the Global Constraints.
2. `.venv/Scripts/python.exe tools/prose_guard.py HEAD <PATHS>` -> expect `OK`, exit 0. A `LOGIC CHANGED` line means revert that file's logic edit; the guard is not advisory.
3. `.venv/Scripts/python.exe -m ruff check <PATHS>` -> expect no findings.

   **`ruff check` does NOT enforce the 79-column limit.** `pyproject.toml` sets
   `line-length = 79` but declares no `[tool.ruff.lint] select`, so ruff runs its
   default `E4/E7/E9/F` set, and E501 is not in it. The repo carries 390 E501
   violations today and always has. Do not attempt a repo-wide E501 cleanup: that
   means reflowing code, which is out of scope.

   What IS required is that prose YOU write fits in 79 columns. Check only your
   own added lines:

   ```bash
   git diff -U0 -- <PATHS> > "$SCRATCH/d.txt" && .venv/Scripts/python.exe -c "
import io
n = 0
for ln in io.open(r'<SCRATCH>/d.txt', encoding='utf-8', errors='replace'):
    ln = ln.rstrip('
')
    if ln.startswith('+') and not ln.startswith('+++') and len(ln) - 1 > 79:
        n += 1; print(len(ln) - 1, ln)
print(n, 'overlong added lines')
"
   ```

   Expect `0 overlong added lines`. Pre-existing overlong lines are not yours
   to fix.

   Write the diff to a real file under your scratchpad directory, not `/tmp`
   (a Git Bash path the Windows interpreter cannot open), and substitute its
   absolute path for `<SCRATCH>`.

   The file-with-explicit-encoding form is deliberate. Do NOT use
   `awk 'length>80'` (awk counts BYTES) and do NOT pipe into bare `python -c`
   (on Windows, stdin decodes as the console codepage, not UTF-8). Both
   over-count any line holding a multi-byte character, and this codebase's math
   prose is full of them (`β`, `∫`, `φ`, `·`). Each wrong form reported a
   76-character line as 81.
4. `.venv/Scripts/python.exe -m pytest <SUITE> -q; echo "EXIT=${PIPESTATUS[0]}"` -> expect the recorded baseline, `EXIT=0`.
5. **Only for tasks editing `packages/dtfit/src` (Tasks 2-7)**, from `packages/dtfit/`:

   ```bash
   cd packages/dtfit
   ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"
   ../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"
   cd ../..
   ```

   Expected: `Success: no issues found in 45 source files`, `MYPY_EXIT=0`,
   `MKDOCS_EXIT=0`. A mkdocs failure after a docstring rewrite is almost always
   a broken indent in an `Args:` block -- fix the docstring, not the config.
6. Report the files changed and the verification output. Do not stage or commit.

**Diff your prose against HEAD, block by block — do not only read it forwards.**
This is the stronger of the two checks and it was learned the hard way. Reading
the surviving text forwards cannot reliably find a word that is missing, because
nothing on the page tells you it was ever there: one implementer read all 242 of
its prose lines end to end, reported no losses, and had shipped
"leaving the regenerated structure the same whatever the units of t" (HEAD:
"...matches whatever the units of t **were**"). Its own trailing-off heuristic
missed it too, because the sentence ends on a bare noun rather than a function
word. Against HEAD the same loss is simply a deletion, and obvious. So: for every
comment and docstring you changed, put the HEAD text and your replacement side by
side and confirm every claim in the original either survives or was deliberately
cut.

**Then also read the full prose of every file you touch, end to end.**
Not the diff hunks: the actual docstrings and comments as they now read. Four
separate tasks lost a clause that a hunk-level view hid, in three different ways:
a changelog sentence cut together with the conclusion it shared a semicolon with;
an ordinary tightening that swept up an adjacent claim carrying no changelog word
at all; and a reflow that amputated a clause mid-paragraph, leaving "contiguous
sub-areas are shrink in magnitude" in a published module docstring. `prose_guard`
cannot catch any of these, because prose is not in the AST. Reading is the only
gate there is.

Because the guard proves logic is untouched, a test failure at step 4 means a docstring was load-bearing -- a doctest, or a `__doc__` read at runtime. Investigate rather than reverting wholesale.

### Task 2: dtfit/src -- core and top-level

**Files:** `packages/dtfit/src/dtfit/_core/` (4), plus `__about__.py`, `__init__.py`, `_pandas.py`, `_signal.py`, `auto.py`, `log.py`, `types.py` (7). 11 files.

Known targets: `_spectral.py` carries 60 comment lines in 507; `types.py::FittingResult` has a 49-line docstring; `__init__.py` has a 59-line module docstring; `auto.py:332` holds a changelog comment.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src/dtfit/_core packages/dtfit/src/dtfit/__about__.py packages/dtfit/src/dtfit/__init__.py packages/dtfit/src/dtfit/_pandas.py packages/dtfit/src/dtfit/_signal.py packages/dtfit/src/dtfit/auto.py packages/dtfit/src/dtfit/log.py packages/dtfit/src/dtfit/types.py` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/src` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> 906 passed, 2 skipped, 17 xfailed, `EXIT=0`
- [ ] **Step 5:** From `packages/dtfit/`: `cd packages/dtfit && ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"` -> `Success: no issues found in 45 source files`, `MYPY_EXIT=0`; then `../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"` -> `MKDOCS_EXIT=0`; then `cd ../..`. A mkdocs failure here is almost always a broken indent in a rewritten `Args:` block -- fix the docstring, not the config.
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 3: dtfit/src -- methods

**Files:** `packages/dtfit/src/dtfit/methods/` -- `__init__.py`, `_common.py`, `_dsb.py`, `_eac.py`, `_ensemble.py`, `_lsi.py`, `_modelinput.py` (7), plus `packages/dtfit/src/dtfit/_core/_native.c` (1). 8 files.

`_native.c` is added here to close a scope gap: Phase 1 enumerated only `.py` files and Task 15 covers only the hardware firmware, so this 373-line C source (71 comment lines, 19%) fell between them. It is the only C source in `dtfit`. `prose_guard.py` cannot parse C, so exclude it from the Step 2 guard command and re-read its diff by hand, exactly as Task 15 does for the firmware.

Known targets: `fit_eac` (107-line docstring), `fit_lsi` (105), `_eac.py` module docstring (42). `_eac.py` holds four changelog comments at lines 332, 355, 449, 484 ("historical sorted-name parameter order", "the old fixed default"). Keep the `Args:`/`Returns:`/`Raises:` blocks on both public fitters.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src/dtfit/methods` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/src/dtfit/methods` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** From `packages/dtfit/`: `cd packages/dtfit && ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"` -> `Success: no issues found in 45 source files`, `MYPY_EXIT=0`; then `../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"` -> `MKDOCS_EXIT=0`; then `cd ../..`. A mkdocs failure here is almost always a broken indent in a rewritten `Args:` block -- fix the docstring, not the config.
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 4: dtfit/src -- models

**Files:** `packages/dtfit/src/dtfit/models/` -- `__init__.py`, `_catalog.py`, `_model.py`, `_stochastic.py`, `_suggest.py`. 5 files.

Known targets: `_suggest.py` is the densest file in the package (39 comment lines in 193) and holds the worst single graveyard comment -- the 8-line paragraph in `_detect_categories` explaining why the old `(not monotone)` veto was removed. Cut it to a one-line statement of what the code does. Its module docstring's "answers the question the domain studies say is the whole game" is the marketing register named in the spec. Preserve the `pyright: ignore` note on `spearmanr(...).statistic` -- that is load-bearing. Changelog comments also sit at `_catalog.py:116` and `_model.py:277`.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src/dtfit/models` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/src/dtfit/models` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** From `packages/dtfit/`: `cd packages/dtfit && ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"` -> `Success: no issues found in 45 source files`, `MYPY_EXIT=0`; then `../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"` -> `MKDOCS_EXIT=0`; then `cd ../..`. A mkdocs failure here is almost always a broken indent in a rewritten `Args:` block -- fix the docstring, not the config.
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 5: dtfit/src -- streaming

**Files:** `packages/dtfit/src/dtfit/streaming/` -- `__init__.py`, `_bank.py`, `_base.py`, `_eac.py`, `_lsi.py`. 5 files.

Known targets: `_lsi.py` is the densest file in scope (142 comment lines in 618, 23%) and holds the 112-line `__init__` docstring -- the longest in the repo -- plus the 33-line bulleted module docstring with `* **Observability.**`-style lead-ins. Flatten the bullets to prose, drop the comparative editorialising ("which is why the area filter needs an `adapt_r` rescaling hack"), and keep the full `Args:` block. Keep the `R_j` proportional to `(2j+1)` note. `_eac.py` has 102 comment lines in 535; changelog comments sit at `_base.py:127`, `_eac.py:340`, `_lsi.py:423,599`.

Do not restructure these files -- they are cohesive and public API. Phase 2 leaves them alone by design.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src/dtfit/streaming` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/src/dtfit/streaming` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** From `packages/dtfit/`: `cd packages/dtfit && ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"` -> `Success: no issues found in 45 source files`, `MYPY_EXIT=0`; then `../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"` -> `MKDOCS_EXIT=0`; then `cd ../..`. A mkdocs failure here is almost always a broken indent in a rewritten `Args:` block -- fix the docstring, not the config.
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 6: dtfit/src -- stochastic

**Files:** `packages/dtfit/src/dtfit/stochastic/` -- `__init__.py`, `_estimators.py`, `_filter.py`, `_forecast.py`, `_model.py`, `_simulate.py`, `_stats.py`. 7 files.

Known targets: `_model.py` (128 comment lines in 646) including the changelog comment at line 511 ("the AR(1)-whitened innovations -- the old approach"); `fit_stochastic` has a 61-line docstring; `_filter.py` a 46-line module docstring; `_stats.py` is 15% comment lines.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src/dtfit/stochastic` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/src/dtfit/stochastic` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** From `packages/dtfit/`: `cd packages/dtfit && ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"` -> `Success: no issues found in 45 source files`, `MYPY_EXIT=0`; then `../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"` -> `MKDOCS_EXIT=0`; then `cd ../..`. A mkdocs failure here is almost always a broken indent in a rewritten `Args:` block -- fix the docstring, not the config.
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 7: dtfit/src -- scale, estimators, diagnostics

**Files:** `packages/dtfit/src/dtfit/scale/` (4), `estimators/` (2), `diagnostics/` (3). 9 files.

Known targets: `estimators/_regressor.py` (56 comment lines in 418; `NonlineRegressor` has a 63-line class docstring). `diagnostics/_plot.py` and `models/_stochastic.py` looked orphaned during the audit but are re-exported through their package `__init__` -- do not delete them.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/src/dtfit/scale packages/dtfit/src/dtfit/estimators packages/dtfit/src/dtfit/diagnostics` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/src` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** From `packages/dtfit/`: `cd packages/dtfit && ../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"` -> `Success: no issues found in 45 source files`, `MYPY_EXIT=0`; then `../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"` -> `MKDOCS_EXIT=0`; then `cd ../..`. A mkdocs failure here is almost always a broken indent in a rewritten `Args:` block -- fix the docstring, not the config.
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 8: dtfit/tests -- root and validation

**Files:** `packages/dtfit/tests/` root (`conftest.py`, `test_examples.py`, `test_imports.py`, `test_improvements.py`, `test_logging.py`, `test_pandas_io.py`, `test_readme_examples.py`, `test_result.py`, `test_stochastic.py`) and `validation/` (5). 14 files.

Known targets: `test_improvements.py` opens with a 17-line bulleted module docstring listing every fix it guards, and uses banner rules between tests -- both redundant with the test names; it also has the changelog comment at line 64. Changelog comments at `test_stochastic.py:159,171` and `validation/test_accuracy_matrix.py:77`.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/tests` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/tests` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 9: dtfit/tests -- methods and models

**Files:** `packages/dtfit/tests/methods/` (9), `models/test_models.py` (1). 10 files.

Known targets: changelog comments at `test_auto.py:495`, `test_dsb.py:36`, `test_eac.py:33,153,192`, `test_lsi.py:88`, `test_models.py:146,160,357`. `test_dsb.py` is 21% comment lines; `test_auto.py` has 82 comment lines in 514.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/tests/methods packages/dtfit/tests/models` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/tests` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 10: dtfit/tests -- remaining subdirectories

The stale-reference fix that once made `tests/core/test_native.py` report
`LOGIC CHANGED` is now part of HEAD (commit 4d87a6b), so the guard is clean
again for this directory. Expect a plain `OK`; any `LOGIC CHANGED` here is a
real defect.

**Files:** `packages/dtfit/tests/` -- `accuracy/` (4), `core/` (3), `diagnostics/` (1), `estimators/` (2), `scale/` (5), `streaming/` (2). 17 files.

Known targets: `streaming/test_streaming.py` (90 comment lines in 869) including the changelog comment at line 697.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit/tests` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit/tests` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> baseline, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 11: experimental -- package top level and experiment entry points

**Files:** `packages/dtfit-experimental/src/dtfit_experimental/` -- `__init__.py`, `basis_lsi.py`, `boosting.py`, `information.py`, `joint.py`; and `experiments/` top level -- `__init__.py`, `accuracy_explore.py`, `benchmark.py`, `download_data.py`, `streaming_lsi_benchmark.py`, `validate_methods.py`. 11 files.

Known targets: the package `__init__.py` has a 70-line module docstring including the changelog note at line 81 ("are no longer re-exported from here"). `validate_methods.py` is 15% comment lines. All five entry points do distinct jobs -- do not merge them.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit-experimental/src/dtfit_experimental` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit-experimental/src` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> 72 passed, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 12: experimental -- cases

**Files:** `packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/` -- 21 `.py` files (`__init__.py` plus ten `NN_name/{__init__,backend}.py`).

Known targets: `06_benchmark_ltsf/backend.py` (16% comment lines), `08_gpu_batched_projection/backend.py`.

The `cases/` tree is a deliberate counterpart to `domains/` -- per-adaptation isolation studies versus end-to-end domain studies. Both are live. Do not merge or delete either.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit-experimental/src/dtfit_experimental/experiments/cases` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit-experimental/src` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> 72 passed, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 13: experimental -- common and domains

**Files:** `packages/dtfit-experimental/src/dtfit_experimental/experiments/common/` (6) and `experiments/domains/` (15). 21 files.

Known targets: `domains/realtime_gps/backend.py` (80 comment lines in 598); `domains/forecasting/backend.py` (152 in 1278) and `big_data/backend.py` -- both with ~48-line module docstrings; `domains/common.py` is 15% comment lines; changelog comments at `parameter_estimation/backend.py:343` and `stochastic_series/backend.py:962-963`.

Prose only here. The structural split of `forecasting`, `stochastic_series`, `parameter_estimation`, and `baselines` happens in Phase 2, after Phase 1 is green everywhere.

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit-experimental/src/dtfit_experimental/experiments/common packages/dtfit-experimental/src/dtfit_experimental/experiments/domains` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit-experimental/src` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> 72 passed, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 14: experimental -- tests

**Files:** `packages/dtfit-experimental/tests/` -- `test_adaptations.py`, `test_baselines.py`, `test_classical_stochastic.py`, `test_domain_backends_import.py`, `test_information.py`, `test_stochastic.py`. 6 files.

Known target: changelog comment at `test_classical_stochastic.py:81` ("the previously-missing foils").

- [ ] **Step 1:** Rewrite prose per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit-experimental/tests` -> `OK`
- [ ] **Step 3:** `.venv/Scripts/python.exe -m ruff check packages/dtfit-experimental/tests` -> no findings
- [ ] **Step 4:** `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> 72 passed, `EXIT=0`
- [ ] **Step 5:** Report the files changed and the verification output. Do not stage or commit.

### Task 15: hardware -- Python and firmware

**Files:** `packages/dtfit-hardware/src/dtfit_hardware/` -- `__init__.py`, `backend.py`, `compare_real.py`, `tools/embed_lsi.py` (4 Python); `packages/dtfit-hardware/tests/` -- `test_embed.py`, `test_smoke.py` (2 Python); and the 14 firmware sources under `src/dtfit_hardware/firmware/` (`.ino` and `.h`).

Known targets: `compare_real.py` has a 26-line bulleted module docstring and the changelog comment at line 142; `load_log`'s docstring narrates the v4 log-format change. `tools/embed_lsi.py` is 11% comment lines.

Keep the local-ENU-metres rationale in `compare_real.py` -- the spec names it as load-bearing.

The firmware is C++, so `prose_guard.py` does not cover it. Apply the same prose policy by hand and rely on review. Do not attempt to compile; no Arduino toolchain is assumed. The React Native app under `packages/dtfit-hardware/mobile/` is out of scope.

- [ ] **Step 1:** Rewrite prose in the 6 Python files per Global Constraints.
- [ ] **Step 2:** `.venv/Scripts/python.exe tools/prose_guard.py HEAD packages/dtfit-hardware/src packages/dtfit-hardware/tests` -> `OK`
- [ ] **Step 3:** Rewrite prose in the 14 firmware `.ino`/`.h` files per Global Constraints. No guard available; re-read each diff.
- [ ] **Step 4:** `.venv/Scripts/python.exe -m ruff check packages/dtfit-hardware` -> no findings
- [ ] **Step 5:** `.venv/Scripts/python.exe -m pytest packages/dtfit-hardware/tests -q; echo "EXIT=${PIPESTATUS[0]}"` -> 5 passed, `EXIT=0`
- [ ] **Step 6:** Report the files changed and the verification output. Do not stage or commit.

### Task 16: Organization fixes

**Files:**
- Modify: `packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/analysis/*.md` (16 stale links across 12 files)
- Delete: `img/` (7 tracked files)
- Delete: `__pycache__/_diag_variants.cpython-314.pyc` (untracked)

The 16 links point at `../../src/dtfit/adaptations/*.py`, a pre-monorepo layout the promotion refactor dissolved. Repoint each to where the code lives now, for example `partitioned.py` to `packages/dtfit/src/dtfit/scale/_partitioned.py` and `streaming/` to `packages/dtfit/src/dtfit/streaming/`. Resolve each target by searching for the symbol rather than guessing the path; two of them (`_native.c`, `multiresolution.py`) may have no current equivalent, in which case drop the link and keep the prose.

- [ ] **Step 1: List the exact broken links**

```bash
grep -rn "src/dtfit/adaptations\|src/dtfit/_native.c\|src/dtfit/parallel.py\|src/dtfit/streaming/" \
  packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/analysis/
```

- [ ] **Step 2: Repoint each link to the current path, dropping any with no equivalent**

- [ ] **Step 3: Verify no broken relative links remain in that directory**

```bash
.venv/Scripts/python.exe - <<'EOF'
import os, re
pat = re.compile(r'\[[^\]]*\]\(([^)#][^)]*?)\)')
base = "packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/analysis"
bad = 0
for f in sorted(os.listdir(base)):
    if not f.endswith(".md"):
        continue
    p = os.path.join(base, f)
    for m in pat.finditer(open(p, encoding="utf-8").read()):
        t = m.group(1).split("#")[0].strip()
        if not t or t.startswith(("http", "mailto:")):
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, t))):
            print(f"STILL BROKEN: {p} -> {t}")
            bad += 1
print(f"{bad} broken")
EOF
```

Expected: `0 broken`.

- [ ] **Step 4: Delete the approved files**

```bash
rm -rf img/
rm -f __pycache__/_diag_variants.cpython-314.pyc
```

`img/` is tracked, so this leaves 7 deletions showing in `git status`. Do not
run `git rm` and do not commit -- the owner reviews the deletion before it is
recorded.

- [ ] **Step 5: Confirm nothing referenced `img/`**

```bash
grep -rn "img/" --include='*.md' --include='*.py' --include='*.yml' . \
  | grep -v -e node_modules -e '\.venv' -e '^\./docs/superpowers'
```

Expected: no output. If anything appears, fix the reference before committing.

- [ ] **Step 6: Report, do not commit**

Report the repointed links, the deleted paths, and the `grep` output from Step 5.

---

## Phase 2 -- Structure

Begins only once Tasks 2-16 are complete and all three suites are green. The prose guard does not apply -- logic changes are the point. Public API stays frozen.

Each task extracts cohesive groups into sibling modules, keeps the original module's importable names intact by re-exporting, runs the package suite, and commits.

### Task 17: Split `experiments/domains/forecasting/backend.py`

**Files:**
- Modify: `packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/forecasting/backend.py` (1277 lines)
- Create: sibling modules under the same package, named for the study groups found in Step 2.
- Test: `packages/dtfit-experimental/tests/test_domain_backends_import.py`

**Interfaces:**
- Consumes: nothing from earlier Phase 2 tasks.
- Produces: `backend.py` must keep re-exporting every name it exports today. `forecasting.ipynb` imports it via `importlib.import_module("...domains.forecasting.backend")` and calls attributes off it, so the module's public surface is its contract.

- [ ] **Step 1: Record the current public surface**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.domains.forecasting.backend as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > forecasting_before.txt
cat forecasting_before.txt
```

- [ ] **Step 2: Identify the study groups**

```bash
grep -n "^def \|^class \|^# ---" packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/forecasting/backend.py
```

Group by what each function operates on. Only split where groups are genuinely independent; if the file turns out to be one cohesive study, stop and report that rather than splitting for its own sake.

- [ ] **Step 3: Move each group to its own module, re-exporting from `backend.py`**

- [ ] **Step 4: Verify the public surface is unchanged**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.domains.forecasting.backend as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > forecasting_after.txt
diff forecasting_before.txt forecasting_after.txt && echo "SURFACE UNCHANGED"
```

Expected: `SURFACE UNCHANGED`. Delete both scratch files before committing.

- [ ] **Step 5: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"`
Expected: 72 passed, `EXIT=0`.

- [ ] **Step 6: Clean up scratch and report**

```bash
rm -f forecasting_before.txt forecasting_after.txt
```

Then report the new module list, the `SURFACE UNCHANGED` confirmation, and the
suite output. Do not stage or commit.

### Task 18: Split `experiments/domains/stochastic_series/backend.py`

**Files:**
- Modify: `packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/stochastic_series/backend.py` (1268 lines)
- Create: sibling modules named for the study groups.
- Test: `packages/dtfit-experimental/tests/test_stochastic.py`, `packages/dtfit-experimental/tests/test_classical_stochastic.py`

**Interfaces:**
- Consumes: nothing from Task 17.
- Produces: `backend.py` keeps its full public surface; `stochastic_series.ipynb` imports it by module path.

- [ ] **Step 1: Record the current public surface**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.domains.stochastic_series.backend as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > stochastic_before.txt
```

- [ ] **Step 2: Identify the study groups**

```bash
grep -n "^def \|^class \|^# ---" packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/stochastic_series/backend.py
```

Only split where groups are genuinely independent. Note the changelog comment at lines 962-963 concerns the long-memory router: keep the current behaviour exactly.

- [ ] **Step 3: Move each group to its own module, re-exporting from `backend.py`**

- [ ] **Step 4: Verify the public surface is unchanged**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.domains.stochastic_series.backend as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > stochastic_after.txt
diff stochastic_before.txt stochastic_after.txt && echo "SURFACE UNCHANGED"
```

Expected: `SURFACE UNCHANGED`.

- [ ] **Step 5: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"`
Expected: 72 passed, `EXIT=0`.

- [ ] **Step 6: Clean up scratch and report**

```bash
rm -f stochastic_before.txt stochastic_after.txt
```

Then report the new module list, the `SURFACE UNCHANGED` confirmation, and the
suite output. Do not stage or commit.

### Task 19: Split `experiments/common/baselines.py`

**Files:**
- Modify: `packages/dtfit-experimental/src/dtfit_experimental/experiments/common/baselines.py` (886 lines)
- Create: one module per baseline family, as siblings in `experiments/common/`
- Test: `packages/dtfit-experimental/tests/test_baselines.py`

**Interfaces:**
- Consumes: nothing from Tasks 17-18.
- Produces: every baseline class currently importable from `dtfit_experimental.experiments.common.baselines` stays importable from that exact path. Domain backends import from it directly, so this is a wide contract -- enumerate consumers first.

- [ ] **Step 1: Find every consumer**

```bash
grep -rn "baselines" --include='*.py' --include='*.ipynb' packages/ | grep -v node_modules
```

- [ ] **Step 2: Record the current public surface**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.common.baselines as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > baselines_before.txt
```

- [ ] **Step 3: Split by baseline family, re-exporting all names from `baselines.py`**

Keep `baselines.py` as the façade rather than converting it to a package directory -- consumers import the module path directly, and a file-to-package swap risks shadowing during the transition.

- [ ] **Step 4: Verify the public surface is unchanged**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.common.baselines as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > baselines_after.txt
diff baselines_before.txt baselines_after.txt && echo "SURFACE UNCHANGED"
```

Expected: `SURFACE UNCHANGED`.

- [ ] **Step 5: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"`
Expected: 72 passed, `EXIT=0`.

- [ ] **Step 6: Clean up scratch and report**

```bash
rm -f baselines_before.txt baselines_after.txt
```

Then report the new module list, the `SURFACE UNCHANGED` confirmation, and the
suite output. Do not stage or commit.

### Task 20: Split `dtfit_hardware/compare_real.py`

**Files:**
- Modify: `packages/dtfit-hardware/src/dtfit_hardware/compare_real.py` (872 lines)
- Create: `packages/dtfit-hardware/src/dtfit_hardware/_log.py` (CSV loading and ENU projection), `_score.py` (forecast and dropout-coasting metrics), `_report.py` (formatting)
- Test: `packages/dtfit-hardware/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing from Tasks 17-19.
- Produces: `load_log(path: str) -> dict` and `report(path: str)` stay importable from `dtfit_hardware.compare_real`. The module docstring documents `python -c "import compare_real as C; print(C.report('data/your_run.csv'))"` as a supported entry point, so that call must keep working.

- [ ] **Step 1: Record the current public surface**

```bash
.venv/Scripts/python.exe -c "import dtfit_hardware.compare_real as C; print('\n'.join(sorted(n for n in dir(C) if not n.startswith('_'))))" > compare_before.txt
```

- [ ] **Step 2: Extract loading and ENU projection into `_log.py`**

Carry the local-ENU-metres rationale comment across; it explains why the fit stays well-conditioned and why the firmware matches.

- [ ] **Step 3: Extract the forecast and dropout-coasting metrics into `_score.py`**

- [ ] **Step 4: Extract report formatting into `_report.py`**

- [ ] **Step 5: Re-export from `compare_real.py` and verify the surface**

```bash
.venv/Scripts/python.exe -c "import dtfit_hardware.compare_real as C; print('\n'.join(sorted(n for n in dir(C) if not n.startswith('_'))))" > compare_after.txt
diff compare_before.txt compare_after.txt && echo "SURFACE UNCHANGED"
```

Expected: `SURFACE UNCHANGED`.

- [ ] **Step 6: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest packages/dtfit-hardware/tests -q; echo "EXIT=${PIPESTATUS[0]}"`
Expected: 5 passed, `EXIT=0`.

- [ ] **Step 7: Clean up scratch and report**

```bash
rm -f compare_before.txt compare_after.txt
```

Then report the new module list, the `SURFACE UNCHANGED` confirmation, and the
suite output. Do not stage or commit.

### Task 21: Split `experiments/domains/parameter_estimation/backend.py`

**Files:**
- Modify: `packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/parameter_estimation/backend.py` (830 lines)
- Create: sibling modules named for the study groups.
- Test: `packages/dtfit-experimental/tests/test_domain_backends_import.py`

**Interfaces:**
- Consumes: nothing from Tasks 17-20.
- Produces: `backend.py` keeps its full public surface; `parameter_estimation.ipynb` imports it by module path.

- [ ] **Step 1: Record the current public surface**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.domains.parameter_estimation.backend as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > paramest_before.txt
```

- [ ] **Step 2: Identify the study groups**

```bash
grep -n "^def \|^class \|^# ---" packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/parameter_estimation/backend.py
```

- [ ] **Step 3: Move each group to its own module, re-exporting from `backend.py`**

- [ ] **Step 4: Verify the public surface is unchanged**

```bash
.venv/Scripts/python.exe -c "import dtfit_experimental.experiments.domains.parameter_estimation.backend as B; print('\n'.join(sorted(n for n in dir(B) if not n.startswith('_'))))" > paramest_after.txt
diff paramest_before.txt paramest_after.txt && echo "SURFACE UNCHANGED"
```

Expected: `SURFACE UNCHANGED`.

- [ ] **Step 5: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"`
Expected: 72 passed, `EXIT=0`.

- [ ] **Step 6: Clean up scratch and report**

```bash
rm -f paramest_before.txt paramest_after.txt
```

Then report the new module list, the `SURFACE UNCHANGED` confirmation, and the
suite output. Do not stage or commit.

---

## Final verification

- [ ] **All three suites green at their baselines**

```bash
.venv/Scripts/python.exe -m pytest packages/dtfit/tests -q; echo "EXIT=${PIPESTATUS[0]}"
.venv/Scripts/python.exe -m pytest packages/dtfit-experimental/tests -q; echo "EXIT=${PIPESTATUS[0]}"
.venv/Scripts/python.exe -m pytest packages/dtfit-hardware/tests -q; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: 906 / 72 / 5 passed, `EXIT=0` each.

- [ ] **Lint, types, and docs clean**

```bash
.venv/Scripts/python.exe -m ruff check packages/
cd packages/dtfit
../../.venv/Scripts/python.exe -m mypy; echo "MYPY_EXIT=$?"
../../.venv/Scripts/python.exe -m mkdocs build --strict; echo "MKDOCS_EXIT=$?"
cd ../..
```

Expected: `All checks passed!`, `Success: no issues found in 45 source files`,
`MYPY_EXIT=0`, `MKDOCS_EXIT=0`.

- [ ] **Prose reduced**

```bash
.venv/Scripts/python.exe - <<'EOF'
import ast, os
roots = ["packages/dtfit/src", "packages/dtfit/tests",
         "packages/dtfit-experimental/src", "packages/dtfit-experimental/tests",
         "packages/dtfit-hardware/src"]
tot = doc = 0
for r in roots:
    for dp, _, fn in os.walk(r):
        if "__pycache__" in dp:
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            src = open(os.path.join(dp, f), encoding="utf-8").read()
            tot += src.count("\n") + 1
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                    d = ast.get_docstring(n, clean=False)
                    if d:
                        doc += d.count("\n") + 1
print(f"{doc} docstring lines / {tot} total = {doc/tot:.1%} (was 20.1%)")
EOF
```

- [ ] **Register tells reduced**

```bash
SCOPE="packages/dtfit/src packages/dtfit/tests packages/dtfit-experimental packages/dtfit-hardware/src"
echo "-- as em-dash: $(grep -rhoE ' -- ' --include='*.py' $SCOPE | wc -l)  (was 911)"
echo "**bold**:      $(grep -rhoE '\*\*[A-Za-z][^*]{2,}\*\*' --include='*.py' $SCOPE | wc -l)  (was 465)"
echo "banner rules:  $(grep -rhE '^# -{20,}' --include='*.py' $SCOPE | wc -l)  (was 255)"
```

- [ ] **No changelog narration left**

```bash
grep -rnE "#.*\b(the old|previously|used to|no longer|historical|legacy)\b" \
  --include='*.py' packages/dtfit/src packages/dtfit/tests \
  packages/dtfit-experimental packages/dtfit-hardware/src
```

Expected: no output, or only hits where the phrase describes current behaviour rather than a past edit.

- [ ] **API reference still resolves** -- confirm no symbol referenced by `packages/dtfit/docs/api/*.md` was renamed.

```bash
.venv/Scripts/python.exe - <<'EOF'
import glob, importlib, re
missing = []
for f in glob.glob("packages/dtfit/docs/api/*.md"):
    for m in re.finditer(r'^:::\s*(dtfit\.[A-Za-z_.]+)', open(f, encoding="utf-8").read(), re.M):
        path = m.group(1).split(".")
        obj = None
        for i in range(len(path), 0, -1):
            try:
                obj = importlib.import_module(".".join(path[:i]))
                rest = path[i:]
                break
            except ImportError:
                continue
        if obj is None:
            missing.append(m.group(1)); continue
        try:
            for a in rest:
                obj = getattr(obj, a)
        except AttributeError:
            missing.append(m.group(1))
print("MISSING:", missing or "none")
EOF
```

Expected: `MISSING: none`.
