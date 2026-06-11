# Contributing to glide

Thanks for your interest. A few conventions keep this package clean and publishable.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[kinetics,test]"     # core + deeptime/matplotlib + pytest
pytest                                 # run the suite
pytest --cov=glide --cov-report=term-missing   # with coverage
```

> **macOS note.** Use a *fully isolated* venv. A `--system-site-packages` venv that
> mixes a conda **MKL** numpy with pip torch's libomp can segfault from duplicate
> OpenMP runtimes; a clean venv pulls a wheel-based numpy (Accelerate/OpenBLAS).

## Provenance boundary (please respect it)

This package owns its code. The rule:

- **Reimplement from the published method** — the flow, the entropy coders, the path
  bound, IGFS, the spline flow, the temporal/predictive models. Don't paste in
  someone else's source.
- **Import, never vendor** — `numpy/scipy/scikit-learn/torch/mdtraj` and `deeptime`
  (the kinetics engine) are pip dependencies.
- **Run external compressors as subprocesses** — MDZip / SZ3 / ZFP are baselines
  located via env vars; they are never copied into the tree.

This keeps the repo license-clean: we **cite** prior work, we do not **relicense** it.

## Tests

Every functional path has a test (~99% line coverage). torch/deeptime-dependent tests
use `pytest.importorskip` so the suite still runs without those extras. If you add a
feature, add a test that exercises it (and, for any lossless coder, assert an **exact**
round-trip).

## Honesty constraints

The package keeps claims scoped (see the README "Honesty constraints"): the ensemble
bound does **not** cover kinetics (only the path bound does); compression ratio is not
the headline; the ML components are cited prior art, not claimed as invented; rate
gains (e.g. the T9 predictive coder vs the T8 temporal coder) are **empirical**, never
assumed. Keep new claims at that bar, and verify any new citation before adding it.
