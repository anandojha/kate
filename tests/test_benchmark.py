"""T3 contrast harness. The deliverable is the CONTRAST: GLIDE's transition term ~0
(it retains the MSM) while an ensemble-only method ('shuffle') has a large transition
term with a small ensemble term -- 'ensemble preserved, kinetics not'. Pure
numpy/sklearn/scipy (no torch, no deeptime); matplotlib only for the figure."""
import os

import numpy as np
import pytest

from glide.benchmark import run_benchmark
from _synth import metastable_coords


def test_contrast_glide_zero_shuffle_large():
    coords = metastable_coords(n_steps=4000, n_atoms=8, seed=0)
    res = run_benchmark([coords], methods=["glide", "shuffle", "quantize"],
                        lag=10, nstates=40, out=None, verbose=False)
    by = {r["method"]: r for r in res if r.get("available")}
    # GLIDE retains the MSM -> transition term identically ~0
    assert by["glide"]["transition_kl"] < 1e-9
    # an ensemble-only resample destroys kinetics: large transition term ...
    assert by["shuffle"]["transition_kl"] > 1e-2
    assert by["shuffle"]["transition_kl"] > 1e6 * by["glide"]["transition_kl"] + 1e-3
    # ... while the ENSEMBLE is preserved (the static bound would 'certify' it)
    assert by["shuffle"]["ensemble_kl"] < 0.2


def test_benchmark_writes_contrast_png(tmp_path):
    pytest.importorskip("matplotlib")
    coords = metastable_coords(n_steps=3000, n_atoms=6, seed=1)
    out = str(tmp_path / "bench")
    run_benchmark([coords], methods=["glide", "shuffle"], lag=10, nstates=30,
                  out=out, verbose=False)
    assert os.path.exists(out + ".png")


def test_external_baseline_unavailable_is_skipped():
    coords = metastable_coords(n_steps=1500, n_atoms=6, seed=2)
    res = run_benchmark([coords], methods=["glide", "sz3"], lag=10, nstates=20,
                        verbose=False)
    by = {r["method"]: r for r in res}
    assert by["glide"].get("available") is True
    assert by["sz3"].get("available") is False   # SZ3 not configured locally
