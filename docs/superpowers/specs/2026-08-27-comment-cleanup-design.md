# Comment cleanup and structural pass

**Date:** 2026-08-27
**Status:** approved, not yet implemented

## Problem

Roughly 28% of the Python tree is prose, and much of it documents decisions
rather than code. Comments narrate changes that git already records ("the old
`(not monotone)` veto dropped..."), defend code that was never written
("Deliberately permissive", "so the recommender never silently drops..."), or
restate the line below them. Module docstrings run to essay length -- 55 of
them exceed 25 lines, the longest being 112. The register throughout reads as
machine-written: `--` for em-dashes, `**bold**` inside source prose, and
superlatives ("the whole game", "faithfully mirroring").

Measured across the four in-scope trees (32,389 lines, 168 files):

| | count |
|---|---|
| docstring lines | 6,503 (20.1%) |
| `#` comment lines | 3,579 (8.3%) |
| docstrings >= 25 lines | 55 |
| `--` as em-dash | 911 |
| `**bold**` in prose | 465 |
| `*italic*` in prose | 298 |
| `# -----...----- #` banner rules | 255 |
| explicit changelog comments | ~35 |

## Scope

In scope:

- `packages/dtfit/src`
- `packages/dtfit/tests`
- `packages/dtfit-experimental` (src + tests)
- `packages/dtfit-hardware/src`

Out of scope: `dis/`, `papers/` (both git-ignored, `.gitignore:189-190`),
`wiki/`, Jupyter notebooks, and `packages/dtfit/docs/api/*.md`.

The public API is frozen for this work. Because it does not move, the
mkdocstrings pages under `docs/api/` remain valid and are not touched. If any
change turns out to require an API change, it stops and gets raised rather
than being carried through silently.

## Phase 1 -- prose

Applies to every file in scope. Comments and docstrings only; no logic
changes.

### Cut

- **Changelog narration.** "the old ... veto", "previously only exercised
  via...", "unchanged from the historical...". Git holds this history.
- **Justification of code that was never written.** "Kept independent of...",
  "Deliberately permissive", "so the recommender never silently drops...".
- **Marketing register.** "answers the question the domain studies say is the
  whole game", "faithfully mirroring", "the real win".
- **Restatement.** `# stderr_ is therefore real & finite` above an assertion
  on `stderr_`.
- **Banner rules.** The 255 `# ---------- #` separators, where the function or
  test name below already carries the label.

### Keep, tightened

- `Args:` / `Returns:` / `Raises:` on every docstring that has them. These stay
  everywhere, not only on published symbols -- they are how a caller knows what
  they are dealing with. Made specific and concise: `window_size: Samples per
  window.` rather than a paragraph on why the default is 50.
- Non-obvious *why*: numerical-stability tricks, the local-ENU-metres choice in
  `compare_real.py`, the `pyright: ignore` on scipy's untyped `.statistic`.
- Mathematical anchors: the Legendre orthonormal weight `R_j` proportional to
  `(2j+1)`, quadrature choices, and similar derivation notes. These are
  load-bearing in a dissertation codebase.

### Register

- The 911 `--` sites are recast to plain punctuation -- a comma, colon, or
  full stop -- in preference to substituting a Unicode em-dash. A dash-heavy
  register is itself part of what reads as machine-written, so trading one
  dash for another misses the point.
- `**bold**` and `*italic*` are dropped from prose (763 sites); they belong in
  prose documents, not source.
- Plain declarative sentences.
- Ruff's 79-column limit holds throughout.

### Gate

For each file, parse before and after, strip every docstring, and assert
`ast.dump()` is byte-identical. This proves the edit changed nothing but prose,
and specifically catches a `--` or `**` rewrite that lands inside a string
literal or regex -- the failure mode that makes a naive codemod unsafe.

Then per batch: `ruff check`, plus the suite belonging to the package being
edited -- `packages/dtfit/tests` (906 passed, 43s), and for the other two
packages `packages/dtfit-experimental/tests` and
`packages/dtfit-hardware/tests`. A batch is not done until its package's suite
is green.

## Phase 2 -- structure

Begins only once Phase 1 is complete and green. Logic may change; the public
API may not. The AST check does not apply here; tests and review are the gate.

A file is split only where it holds genuinely separable responsibilities:

| file | lines | rationale |
|---|---|---|
| `experiments/domains/forecasting/backend.py` | 1277 | several unrelated studies in one module |
| `experiments/domains/stochastic_series/backend.py` | 1268 | same |
| `experiments/common/baselines.py` | 886 | one class per baseline family |
| `dtfit_hardware/compare_real.py` | 872 | load / project / score / report |
| `experiments/domains/parameter_estimation/backend.py` | 830 | several unrelated studies |

Explicitly left alone: `streaming/_lsi.py` (617) and `stochastic/_model.py`
(645). Both are cohesive and both are public API.

Anything else found gets reported, not acted on unilaterally.

## Organization

Investigated and found sound -- no action:

- `experiments/cases/` vs `experiments/domains/` is a deliberate two-tier
  split (per-adaptation isolation studies vs end-to-end domain studies). Both
  are live and both drive notebooks.
- The five experiment entry points (`accuracy_explore`, `benchmark`,
  `validate_methods`, `streaming_lsi_benchmark`, `download_data`) each do a
  distinct job.
- `diagnostics/_plot.py` and `models/_stochastic.py` appear orphaned to a
  module-name search but are re-exported through their package `__init__`.

Link audit: 494 apparent broken markdown links reduce to 74 once wiki-style
extension-less links resolve, and to 16 once `dis/`'s 58 are confirmed to
resolve from the repo root via `build_dis.py`.

Actions:

1. Repoint the 16 stale links in
   `experiments/cases/analysis/*.md`, which still reference the pre-monorepo
   `src/dtfit/adaptations/` layout that the promotion refactor dissolved.
2. Delete `img/` -- 7 tracked files, five of them named
   `ezgif.com-animated-gif-maker (N).gif`, referenced by no markdown in the
   repo. Approved for deletion.
3. Delete the orphan bytecode `__pycache__/_diag_variants.cpython-314.pyc`,
   left by a scratch file that was never committed.

## Sequencing

One package at a time, roughly ten files per batch, gated after each batch:

1. `packages/dtfit/src`
2. `packages/dtfit/tests`
3. `packages/dtfit-experimental`
4. `packages/dtfit-hardware/src`

Phase 2 follows once Phase 1 is green across all four. Commits are separated by
phase and by package, so any single batch can be reverted on its own.

## Baseline

Recorded 2026-08-27, all three green. This is the reference both phases must
preserve.

| suite | result | time |
|---|---|---|
| `packages/dtfit/tests` | 906 passed, 2 skipped, 17 xfailed | 43.30s |
| `packages/dtfit-experimental/tests` | 72 passed | 8.72s |
| `packages/dtfit-hardware/tests` | 5 passed | 1.27s |

Roughly 53s to run all three, so gating every batch on the affected suite is
cheap.

Read the exit status from `PIPESTATUS`, not from a pipeline ending in `tail`.
An early baseline attempt reported exit code 0 while pytest had in fact aborted
on an unrecognised `--timeout` flag (pytest-timeout is not installed).
