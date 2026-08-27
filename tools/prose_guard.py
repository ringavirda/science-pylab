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
            # A scope whose only statement was the docstring needs a
            # filler so the tree stays valid; both revisions get the
            # same one. This deliberately collides with a literal
            # `pass`: `def f(): """doc"""` and `def f(): pass` dump
            # identically, so adding or deleting a sole docstring
            # passes the guard either way. That is intended, not a
            # bug -- see test_docstring_vs_pass_collision_is_ok.
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


def _verify_ref(ref: str) -> str | None:
    """Confirm ``ref`` resolves to a commit.

    Returns None on success, or git's own error message if it does
    not. Every other check in this module assumes ``ref`` is real;
    a typo'd ref must fail loudly here, not surface later as "no
    files changed".
    """
    done = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if done.returncode == 0:
        return None
    return done.stderr.strip() or f"git rev-parse exited {done.returncode}"


def _git_show(ref: str, path: Path) -> tuple[str | None, str | None]:
    """Read ``path`` at ``ref``.

    Returns ``(content, error)``:

    - Found: ``(text, None)``.
    - ``path`` genuinely did not exist at ``ref`` (a new file):
      ``(None, None)`` -- not a failure, the caller should skip and
      count it.
    - Any other ``git show`` failure (bad path, wrong cwd, an
      absolute path git can't resolve, ...): ``(None, message)`` --
      the caller must treat this as a hard failure, never as a
      silent "new file". Distinguished via git's own stderr, since
      both cases exit non-zero identically.
    """
    done = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if done.returncode == 0:
        return done.stdout, None
    stderr = done.stderr.strip()
    if "exists on disk, but not in" in stderr:
        return None, None
    return None, stderr or f"git show exited {done.returncode}"


def main(argv: list[str]) -> int:
    """Run the CLI.

    Exit 0: every path compared prose-identical to ``ref`` (new
    files are noted, not compared). Exit 1: at least one file
    changed more than prose. Exit 2: the invocation itself could not
    be trusted -- bad ref, a path matching nothing on disk, a
    directory with no ``.py`` files, or a run that ended up
    comparing zero files. None of these ever print "OK": a guard
    that silently checks nothing must not look like success.
    """
    if len(argv) < 2:
        print(__doc__)
        return 2
    ref, targets = argv[0], argv[1:]

    ref_error = _verify_ref(ref)
    if ref_error is not None:
        print(f"ERROR: bad ref {ref!r}: {ref_error}")
        return 2

    files: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_absolute():
            # git resolves `ref:path` relative to the repo, not the
            # filesystem: an absolute path to a real, unmodified,
            # tracked file comes back from git as "exists on disk,
            # but not in <ref>" -- indistinguishable from a genuinely
            # new file, and would be silently skipped rather than
            # compared. Reject it instead of guessing.
            print(f"ERROR: absolute paths are not supported: {t}")
            return 2
        if p.is_dir():
            found = sorted(p.rglob("*.py"))
            if not found:
                print(f"ERROR: no .py files under {t}")
                return 2
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            print(f"ERROR: no such file or directory: {t}")
            return 2

    failed = skipped = 0
    for f in files:
        if "__pycache__" in f.parts:
            continue
        try:
            before, err = _git_show(ref, f)
        except UnicodeDecodeError as exc:
            print(f"GIT ERROR:     {f}: {exc}")
            failed += 1
            continue
        if err is not None:
            print(f"GIT ERROR:     {f}: {err}")
            failed += 1
            continue
        if before is None:
            skipped += 1
            continue
        try:
            after = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"READ ERROR:    {f}: {exc}")
            failed += 1
            continue
        try:
            if not check(before, after):
                print(f"LOGIC CHANGED: {f}")
                failed += 1
        except SyntaxError as exc:
            print(f"SYNTAX ERROR:  {f}: {exc}")
            failed += 1

    checked = len(files) - skipped
    if failed:
        print(f"\n{failed} of {checked} file(s) changed more than prose.")
        return 1
    if checked == 0:
        print(
            f"ERROR: compared 0 file(s) ({skipped} new of "
            f"{len(files)} total) -- nothing was verified."
        )
        return 2
    print(f"OK: {checked} file(s) prose-identical to {ref} ({skipped} new).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
