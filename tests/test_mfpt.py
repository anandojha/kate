"""PCCA+ metastable coarse-graining + mean first-passage times (kinetics_deeptime.
metastable_mfpt) -- the rate observables (k_on/k_off ~ 1/MFPT) the path bound is meant to
cover, now actually computed from the reversible-MLE MSM. On a 2-state chain the MFPT must
be finite, positive, scale with the switching rate, and the metastable populations must sum
to 1."""
import numpy as np
import pytest

pytest.importorskip("deeptime")

from glide import kinetics_deeptime as kd
from _synth import two_state_dtraj


def test_metastable_mfpt_two_state():
    dt = [two_state_dtraj(n=40000, a=0.02, seed=0)]
    rep = kd.metastable_mfpt(dt, lag=1, dt_ns=1.0, n_meta=2)
    assert rep["n_meta"] == 2
    assert abs(float(rep["meta_pop"].sum()) - 1.0) < 1e-6     # populations are a distribution
    m = rep["mfpt_ns"]
    assert np.isfinite(m[0, 1]) and np.isfinite(m[1, 0]) and m[0, 1] > 0 and m[1, 0] > 0
    assert np.isnan(m[0, 0]) and np.isnan(m[1, 1])            # no self-MFPT
    # relaxation ~ 1/(2a) ~ 25 frames; MFPT is of that order, comfortably in [10, 2000]
    assert 10 < m[0, 1] < 2000


def test_metastable_mfpt_scales_with_rate():
    # slower switching (smaller a) -> longer MFPT
    fast = kd.metastable_mfpt([two_state_dtraj(n=60000, a=0.04, seed=2)], 1, 1.0, 2)
    slow = kd.metastable_mfpt([two_state_dtraj(n=60000, a=0.008, seed=3)], 1, 1.0, 2)
    assert np.nanmax(slow["mfpt_ns"]) > np.nanmax(fast["mfpt_ns"])


def test_format_mfpt_string():
    rep = kd.metastable_mfpt([two_state_dtraj(n=20000, a=0.03, seed=1)], 1, 0.5, 2)
    s = kd.format_mfpt(rep)
    assert "MFPT" in s and "metastable populations" in s and "S0" in s
