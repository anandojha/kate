"""MSM-community validation tooling required for the paper: the Chapman-Kolmogorov test
(Prinz et al. 2011) and block-bootstrap timescale confidence intervals. A genuinely
Markovian discrete trajectory must pass the CK test (small deviation); the bootstrap must
return a finite, ordered confidence interval bracketing the point estimate."""
import numpy as np
import pytest

pytest.importorskip("deeptime")

from kate import kinetics_deeptime as kd
from _synth import two_state_dtraj


def test_ck_test_passes_for_markovian_chain():
    # a reversible 2-state Markov chain is Markovian by construction -> CK deviation small
    dt = [two_state_dtraj(n=60000, a=0.03, seed=0)]
    ck = kd.ck_test(dt, lag=1, n_metastable=2, factors=(1, 2, 3, 5), n_samples=20)
    assert ck["estimates"].shape == ck["predictions"].shape
    assert ck["estimates"].shape[1:] == (2, 2)          # (n_lag, n_set, n_set)
    assert np.isfinite(ck["max_deviation"])
    assert ck["max_deviation"] < 0.15                    # predictions track estimates


def test_bootstrap_timescales_returns_ordered_ci():
    dt = [two_state_dtraj(n=40000, a=0.04, seed=1)]
    res = kd.bootstrap_timescales(dt, lag=1, k=1, n_boot=40, n_blocks=20, seed=0)
    assert res is not None
    mean, lo, hi = res
    assert mean.shape == lo.shape == hi.shape == (1,)
    assert np.all(np.isfinite(mean)) and mean[0] > 0
    assert lo[0] <= mean[0] <= hi[0]                      # CI brackets the estimate
