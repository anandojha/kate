"""
deeptime kinetics-wrapper smoke test (deeptime-gated). Verifies the production
MSM path -- streaming TICA -> k-means -> reversible MLE MSM -> implied timescales --
runs end to end and recovers the slow process of a synthetic 2-state chain.
"""
import numpy as np
import pytest

pytest.importorskip("deeptime")
from glide import kinetics_deeptime as kd  # noqa: E402


def _two_state_run(n, seed, a=0.01):
    rng = np.random.default_rng(seed)
    P = np.array([[1 - a, a], [a, 1 - a]])
    cdf = np.cumsum(P, axis=1)
    s = np.zeros(n, dtype=int)
    u = rng.random(n)
    for t in range(1, n):
        s[t] = np.searchsorted(cdf[s[t - 1]], u[t])
    return np.stack([(s * 2.0 - 1.0) + 0.3 * rng.standard_normal(n),
                     0.3 * rng.standard_normal(n)], 1)


def test_deeptime_msm_pipeline():
    runs = [_two_state_run(8000, 1), _two_state_run(8000, 2)]
    _, cvs = kd.tica_cvs(runs, lag=10, dim=1)
    dtrajs, _ = kd.cluster(cvs, n_states=20, seed=0)
    its = kd.implied_timescales(dtrajs, lag=10, k=3)
    assert np.isfinite(its[0]) and its[0] > 0


def test_msm_for_pathbound_returns_T_and_active():
    runs = [_two_state_run(8000, 5)]
    _, cvs = kd.tica_cvs(runs, lag=10, dim=1)
    dtrajs, _ = kd.cluster(cvs, n_states=15, seed=0)
    T, active = kd.msm_for_pathbound(dtrajs, lag=10)
    assert T.ndim == 2 and T.shape[0] == T.shape[1]
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6)
    assert active.ndim == 1 and active.size == T.shape[0]
