"""dtfit experiment suite: the study tier of ``dtfit-experimental``.

This is the validation research tree, not part of the importable library
contract (see :mod:`dtfit_experimental`): exempt from the mypy gate,
ruff-relaxed, and run notebook by notebook rather than imported. The library
tier of experimental adaptations never imports from here.

Two families of experiments, each a folder of self-contained Jupyter notebooks:

* ``cases/`` holds the per-adaptation studies, one optimization or structural
  idea per folder, scored on the promotion matrix;
* ``domains/`` holds the per-application-domain studies, where the validated
  levers are merged and run against the methods a practitioner in that domain
  actually uses.

Every experiment folder carries a ``backend.py`` (the single source of truth
for its simulation, estimation and data infra: pure compute, no plotting), the
notebook that imports it and produces the report of tables, figures and
narrative, and a ``figures/`` directory of what was saved. Open and run a
notebook directly, e.g. ``jupyter lab
cases/01_control_systems/01_control_systems.ipynb``, or headless::

    jupyter nbconvert --to notebook --execute --inplace \\
        dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb

``cases/REPORTS.md`` and ``domains/DOMAINS.md`` index the notebooks. The shared
``common`` package holds the pure-compute helpers the backends import: metrics,
baselines, datasets, plotting.
"""
