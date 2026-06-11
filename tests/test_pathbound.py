"""
Pure-numpy tests of the kinetic (path-distribution) bound -- always runnable, no
torch, no deeptime. Encodes the thesis as assertions: preserving the ensemble does
NOT preserve kinetics, and the transition term is what exposes the gap.
"""
import numpy as np
import pytest

from glide import pathbound as pb


def reversible_chain(mu, s_off):
    """Reversible row-stochastic P with stationary distribution mu, from a
    symmetric off-diagonal flow s_off (same construction as the money demo)."""
    mu = np.asarray(mu, float)
    S = np.array(s_off, float)
    np.fill_diagonal(S, 0.0)
    S = 0.5 * (S + S.T)
    diag = mu - S.sum(axis=1)
    np.fill_diagonal(S, np.clip(diag, 0.0, None))
    P = S / mu[:, None]
    return P / P.sum(axis=1, keepdims=True)


def scale_kinetics(mu, s_off, c):
    return reversible_chain(mu, c * np.asarray(s_off, float))


MU = np.array([0.5, 0.3, 0.2])
S = np.array([[0.0, 0.04, 0.01],
              [0.04, 0.0, 0.03],
              [0.01, 0.03, 0.0]])


def test_stationary_distribution_matches_construction():
    P = reversible_chain(MU, S)
    pi = pb.stationary_distribution(P)
    assert np.allclose(pi, MU, atol=1e-6)
    assert abs(pi.sum() - 1.0) < 1e-12


def test_ensemble_kl_zero_for_identical():
    assert pb.ensemble_kl(MU, MU) == pytest.approx(0.0, abs=1e-12)


def test_pinsker_is_sqrt_half_kl_and_monotone():
    assert pb.pinsker(0.0) == 0.0
    assert pb.pinsker(2.0) == pytest.approx(1.0)
    assert pb.pinsker(8.0) == pytest.approx(2.0)
    assert pb.pinsker(0.5) < pb.pinsker(0.6)


def test_ensemble_preserved_kinetics_not():
    """The central claim: an ensemble-only 'compressor' (Q_slow) has ensemble term
    ~0 but a large transition term; the static bound would wrongly certify it."""
    P = reversible_chain(MU, S)
    Q_slow = scale_kinetics(MU, S, 0.3)        # identical mu, slower rates
    r = pb.report_kinetic_fidelity(P, Q_slow, lag=1, L=10000, k=2)
    # stationary distributions identical -> ensemble term ~ 0
    assert abs(r["ensemble_kl_nats"]) < 1e-10
    # but the transition term is large -> kinetics corrupted
    assert r["transition_kl_rate_nats_per_step"] > 1e-2
    # the static (ensemble) Pinsker bound is ~0 (would 'certify' Q_slow)
    assert r["pinsker_ensemble_bound"] < 1e-4
    # the pair bound (which includes the transition term) is not ~0
    assert r["pinsker_pair_bound"] > r["pinsker_ensemble_bound"]
    # the slowest implied timescale is wrong by ~1/c
    assert r["its_cmp"][0] > 2.0 * r["its_ref"][0]


def test_retained_msm_keeps_both_terms_small():
    """Q_good ~ GLIDE's retained MSM: both terms near zero (kinetics preserved)."""
    P = reversible_chain(MU, S)
    rng = np.random.default_rng(0)
    Q_good = P + 1e-3 * rng.random(P.shape)
    Q_good = Q_good / Q_good.sum(axis=1, keepdims=True)
    r = pb.report_kinetic_fidelity(P, Q_good, lag=1, L=10000, k=2)
    assert r["ensemble_kl_nats"] < 1e-4
    assert r["transition_kl_rate_nats_per_step"] < 1e-3
    assert r["its_cmp"][0] == pytest.approx(r["its_ref"][0], rel=0.1)


def test_two_slice_kl_is_ensemble_plus_transition():
    P = reversible_chain(MU, S)
    Q = scale_kinetics(MU, S, 0.5)
    total, ens, tran = pb.two_slice_kl(P, Q)
    assert total == pytest.approx(ens + tran, rel=1e-9, abs=1e-12)


def test_path_kl_grows_with_trajectory_length():
    """Kinetic error accumulates over a trajectory; ensemble error does not."""
    P = reversible_chain(MU, S)
    Q = scale_kinetics(MU, S, 0.4)
    short = pb.path_kl(P, Q, L=10)
    long = pb.path_kl(P, Q, L=10000)
    assert long > short
    # grows linearly in (L-1) * transition term
    _, ens, tran = pb.two_slice_kl(P, Q)
    assert pb.path_kl(P, Q, L=10) == pytest.approx(ens + 9 * tran, rel=1e-9)


def test_support_ok_detects_structural_zeros():
    P = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    Q_full = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])
    Q_hole = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    assert pb.support_ok(P, Q_full)
    # P has a transition (0->1) where Q_hole also has mass, so support holds here;
    # craft a real violation: Q with a zero where P is positive.
    Q_bad = P.copy()
    Q_bad[0, 1] = 0.0
    Q_bad[0, 0] = 1.0
    assert not pb.support_ok(P, Q_bad)
