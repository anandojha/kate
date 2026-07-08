"""
Classical analysis-native codec tests -- numpy/scipy/sklearn only, no torch.
Covers the Witten-Neal-Cleary range coder (exact, reversible), the end-to-end
KineticCodec round trip, and that kinetics are recoverable from the stored MSM.
"""
import numpy as np
import pytest

from kate.kinetic_codec import (
    encode_markov, decode_markov, KineticCodec, kabsch_align,
    count_matrix, transition_matrix, implied_timescales, entropy_rate,
)


def _markov_sequence(n, T, pi, seed=0):
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(T, axis=1)
    s = np.empty(n, dtype=np.int64)
    s[0] = int(np.searchsorted(np.cumsum(pi), rng.random()))
    u = rng.random(n)
    for t in range(1, n):
        s[t] = int(np.searchsorted(cdf[s[t - 1]], u[t]))
    return s


def _simulate(n_steps, n_atoms, a=0.01, intra=0.25, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[1 - a, a, 0.0], [a, 1 - 2 * a, a], [0.0, a, 1 - a]])
    cdf = np.cumsum(P, axis=1)
    m = np.zeros(n_steps, dtype=int)
    u = rng.random(n_steps)
    for t in range(1, n_steps):
        m[t] = np.searchsorted(cdf[m[t - 1]], u[t])
    wells = np.array([-2.0, 0.0, 2.0])
    xi = wells[m] + intra * rng.standard_normal(n_steps)
    ref = rng.standard_normal((n_atoms, 3)) * 2.0
    mode = rng.standard_normal((n_atoms, 3)); mode /= np.linalg.norm(mode)
    xyz = (ref[None] + xi[:, None, None] * mode[None]
           + noise * rng.standard_normal((n_steps, n_atoms, 3)))
    return xyz.astype(np.float64)


def test_markov_range_coder_is_exact():
    rng = np.random.default_rng(3)
    K = 8
    T = rng.random((K, K)) + 0.05
    T /= T.sum(axis=1, keepdims=True)
    pi = rng.random(K); pi /= pi.sum()
    states = _markov_sequence(4000, T, pi, seed=1)
    blob = encode_markov(states, T, pi)
    decoded = decode_markov(blob, len(states), T, pi)
    assert np.array_equal(decoded, states)


def test_range_coder_approaches_entropy_rate():
    # A metastable chain has a small entropy rate; the coder should land near it.
    a = 0.02
    T = np.array([[1 - a, a], [a, 1 - a]])
    pi = np.array([0.5, 0.5])
    states = _markov_sequence(20000, T, pi, seed=2)
    blob = encode_markov(states, T, pi)
    bits_per_step = 8 * len(blob) / len(states)
    H = entropy_rate(T, pi)
    # within ~15% of the Ekroot-Cover floor (finite-length + flush overhead)
    assert bits_per_step >= H - 1e-6
    assert bits_per_step <= H * 1.15 + 0.05


def test_kinetic_codec_roundtrip_and_kinetics():
    runs = [_simulate(6000, 8, seed=11)]
    ref = None; aligned = []
    for r in runs:
        a, ref = kabsch_align(r, ref); aligned.append(a)
    codec = KineticCodec(tica_lag=10, tica_dim=2, n_states=40,
                         msm_lag=10, n_bits=4, reversible=True, seed=0)
    ct = codec.fit_encode(aligned)
    rec = codec.decode(ct)
    assert len(rec) == 1
    assert rec[0].shape == runs[0].shape
    rep = codec.report(ct)
    assert rep["ratio_vs_float32_stream_only"] > 1.0
    assert rep["frames"] == 6000
    kin = ct.kinetics(k=4)
    its = kin["implied_timescales"]
    assert np.all(np.isfinite(its[:2]))
    assert its[0] > its[1] > 0  # two slow processes, ordered


def test_count_matrix_is_run_aware():
    # Two short runs: no transition should be tallied across the seam.
    seqs = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])]
    C = count_matrix(seqs, n_states=2, lag=1)
    assert C.sum() == 6  # 3 transitions per run, no cross-seam pair
    Tm, pi = transition_matrix(C, reversible=True)
    assert np.allclose(Tm.sum(axis=1), 1.0)
