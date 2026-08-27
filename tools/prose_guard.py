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
