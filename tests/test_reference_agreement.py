"""Reference-agreement tests: KATE's reimplemented methods vs canonical sources.

KATE reimplements several published methods from scratch (TICA, the MSM estimators,
the VAMPnet-style soft-state machinery, the RealNVP flow, the Witten-Neal-Cleary
arithmetic coder, Kabsch alignment). These tests pin the physics by cross-checking
each reimplementation against a trusted reference -- deeptime for the kinetics, scipy
for the alignment, an analytic case for the Markov chain, and autograd for the flow
Jacobian -- and asserting agreement to numerical tolerance. They are the evidence
behind the manuscript statement that the estimators reproduce the reference
implementations, so a reviewer need not take the reimplementation on faith.

Each test importorskips only what it needs, so the pure-numpy checks (coder, bound)
run even without torch or deeptime installed.
"""
import numpy as np
import pytest


def _two_state_dtraj(a, b, T, seed):
    """Sample a 2-state Markov chain with P(0->1)=a, P(1->0)=b. Its non-trivial
    eigenvalue is 1-a-b, so the slowest implied timescale is -lag/ln(1-a-b) and the
    stationary probability of state 0 is b/(a+b)."""
    rng = np.random.default_rng(seed)
    d = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        if d[t - 1] == 0:
            d[t] = 1 if rng.random() < a else 0
        else:
            d[t] = 0 if rng.random() < b else 1
    return d


def test_tica_matches_deeptime():
    """KATE TICA solves the same generalized eigenproblem as deeptime: the slowest
    implied timescale and the leading collective variable must agree."""
    pytest.importorskip("deeptime")
    from kate.kinetic_codec import TICA as KTICA
    from deeptime.decomposition import TICA as DTICA
    rng = np.random.default_rng(0)
    T = 20000
    s = np.ones(T)
    for t in range(1, T):
        s[t] = -s[t - 1] if rng.random() < 0.005 else s[t - 1]
    slow = s + 0.3 * rng.standard_normal(T)
    Y = np.stack([0.4 * slow] + [rng.standard_normal(T) for _ in range(3)], 1)
    Y = (Y @ rng.standard_normal((4, 4))).astype(np.float64)          # entangle the slow mode
    lag, dim = 20, 2
    kt = KTICA(lag=lag, n_components=dim).fit([Y])
    dt = DTICA(lagtime=lag, dim=dim).fit_fetch([Y])
    ts_rel = abs(kt.timescales_[0] - dt.timescales()[0]) / dt.timescales()[0]
    corr = abs(np.corrcoef(kt.transform(Y)[:, 0], dt.transform(Y)[:, 0])[0, 1])
    assert ts_rel < 0.05, "slowest TICA timescale differs from deeptime by %.2e" % ts_rel
    assert corr > 0.99, "leading TICA CV correlation with deeptime is only %.4f" % corr


def test_msm_reversible_mle_matches_analytic_and_deeptime():
    """KATE's reported estimator (deeptime reversible MLE via estimate_reversible_T)
    recovers the analytic slowest timescale, agrees with deeptime, and obeys detailed
    balance."""
    pytest.importorskip("deeptime")
    from kate.kinetic_codec import count_matrix, estimate_reversible_T, implied_timescales
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM
    a, b, lag = 0.02, 0.06, 1
    t_true = -lag / np.log(1 - a - b)
    d = _two_state_dtraj(a, b, 120000, seed=1)
    C = count_matrix([d], 2, lag)
    Tk, tag = estimate_reversible_T(C)
    tk = implied_timescales(Tk, lag, 1)[0]
    cm = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch([d]).submodel_largest()
    td = MaximumLikelihoodMSM(reversible=True).fit_fetch(cm).timescales(k=1)[0]
    assert tag == "deeptime-mle"
    assert abs(tk - t_true) / t_true < 0.06, "KATE ITS %.2f vs analytic %.2f" % (tk, t_true)
    assert abs(tk - td) / td < 0.03, "KATE ITS %.2f vs deeptime %.2f" % (tk, td)
    ev, evec = np.linalg.eig(Tk.T)
    pi = np.abs(evec[:, np.argmin(abs(ev - 1.0))].real); pi /= pi.sum()
    db = np.max(np.abs(pi[:, None] * Tk - (pi[:, None] * Tk).T))
    assert db < 1e-6, "detailed balance residual %.2e" % db


def test_symmetrized_estimator_is_more_biased_on_skewed_populations():
    """The (C+C^T)/2 fallback pulls the stationary distribution toward uniform on a
    skewed-population chain, which is exactly why the deeptime MLE is preferred for
    reported numbers. This documents (does not certify) that known bias."""
    from kate.kinetic_codec import count_matrix, transition_matrix, estimate_reversible_T
    a, b = 0.002, 0.05                        # stationary pi0 = b/(a+b) ~ 0.96 (strongly skewed)
    true_pi0 = b / (a + b)
    d = _two_state_dtraj(a, b, 120000, seed=2)
    C = count_matrix([d], 2, lag=1)
    Tcc, picc = transition_matrix(C, reversible=True)
    Tmle, _ = estimate_reversible_T(C)
    ev, evec = np.linalg.eig(Tmle.T)
    pimle = np.abs(evec[:, np.argmin(abs(ev - 1.0))].real); pimle /= pimle.sum()
    assert abs(picc[0] - true_pi0) >= abs(pimle[0] - true_pi0) - 1e-9


def test_vamp2_score_matches_hand_derived():
    """KATE's vamp2_score equals the definition ||C00^-1/2 C01 C11^-1/2||_F^2."""
    torch = pytest.importorskip("torch")
    from kate.bound_loss import vamp2_score, _lagged
    rng = np.random.default_rng(3)
    F = torch.softmax(torch.tensor(rng.standard_normal((5000, 4)), dtype=torch.float64), -1)
    f0, f1 = _lagged(F, 5)
    kv = float(vamp2_score(f0, f1))
    C00 = (f0.T @ f0 / f0.shape[0]).numpy()
    C01 = (f0.T @ f1 / f0.shape[0]).numpy()
    C11 = (f1.T @ f1 / f1.shape[0]).numpy()

    def isqrt(A):
        w, V = np.linalg.eigh(A + 1e-6 * np.eye(len(A))); w = np.clip(w, 1e-6, None)
        return (V * w ** -0.5) @ V.T
    ref = float(((isqrt(C00) @ C01 @ isqrt(C11)) ** 2).sum())
    assert abs(kv - ref) / ref < 1e-3, "VAMP-2 %.6f vs reference %.6f" % (kv, ref)


def test_soft_transition_matrix_reduces_to_hard_msm():
    """With one-hot (hard) assignments the soft transition matrix T=C00^-1 C01 equals
    the ordinary count-based MSM transition matrix, and is row-stochastic."""
    torch = pytest.importorskip("torch")
    from kate.bound_loss import soft_transition_matrix, _lagged
    rng = np.random.default_rng(4)
    P = np.array([[0.9, 0.08, 0.02], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
    d = np.zeros(40000, int)
    for t in range(1, len(d)):
        d[t] = rng.choice(3, p=P[d[t - 1]])
    chi = torch.tensor(np.eye(3)[d], dtype=torch.float64)
    Tsoft = soft_transition_matrix(*_lagged(chi, 1)).numpy()
    Ch = np.zeros((3, 3)); np.add.at(Ch, (d[:-1], d[1:]), 1.0)
    Thard = Ch / Ch.sum(1, keepdims=True)
    assert np.max(np.abs(Tsoft - Thard)) < 1e-6
    assert np.max(np.abs(Tsoft.sum(1) - 1)) < 1e-8


def test_flow_change_of_variables_is_exact():
    """The RealNVP change-of-variables is implemented exactly: forward is invertible,
    the returned log|det J| matches the autograd Jacobian determinant, and the density
    integrates to one."""
    torch = pytest.importorskip("torch")
    from kate.flow import RealNVP
    torch.manual_seed(0); rng = np.random.default_rng(5)
    X = np.stack([rng.standard_normal(4000) * 0.5 + (rng.integers(0, 2, 4000) * 2 - 1),
                  0.6 * rng.standard_normal(4000)], 1).astype(np.float32)
    flow = RealNVP(2, hidden=32, n_layers=8).fit(X, epochs=25, verbose=False)
    xt = torch.tensor(X[:128])
    z, _ = flow.forward(xt)
    assert (flow.inverse(z) - xt).abs().max().item() < 1e-5
    fl = flow.double()
    diffs = []
    for i in range(4):
        xi = xt[i:i + 1].double()
        J = torch.autograd.functional.jacobian(lambda u: fl.forward(u.unsqueeze(0))[0].squeeze(0), xi[0])
        diffs.append(abs(torch.log(torch.abs(torch.det(J))).item() - fl.forward(xi)[1][0].item()))
    assert max(diffs) < 1e-4, "log|det J| disagrees with autograd by %.2e" % max(diffs)
    gx = np.linspace(-4, 4, 160); gy = np.linspace(-3, 3, 120)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        mass = float(torch.exp(flow.log_prob(grid)).sum()) * (gx[1] - gx[0]) * (gy[1] - gy[0])
    assert abs(mass - 1.0) < 0.05, "density integrates to %.4f" % mass


def test_arithmetic_coder_is_lossless_and_near_entropy():
    """The Witten-Neal-Cleary range coder round-trips exactly and its code length
    approaches the Shannon entropy of the coded symbols."""
    from kate.codec import encode_iid, decode_iid, gaussian_cumfreq
    rng = np.random.default_rng(6)
    L, zmax = 256, 6.0
    cum = gaussian_cumfreq(L, zmax)
    lev = np.clip(np.floor((np.clip(rng.standard_normal(15000), -zmax, zmax) + zmax) / (2 * zmax) * L),
                  0, L - 1).astype(np.int64)
    coded = encode_iid(lev, cum)
    assert np.array_equal(lev, decode_iid(coded, len(lev), cum)), "coder round-trip is not lossless"
    p = np.diff(cum).astype(np.float64); p /= p.sum()
    entropy = float(np.mean(-np.log2(p[lev])))
    bits = len(coded) * 8 / len(lev)
    assert bits < entropy + 0.5, "achieved %.3f b/sym vs entropy %.3f" % (bits, entropy)


def test_kabsch_recovers_rotation():
    """Kabsch alignment removes an applied rotation and translation exactly, so the
    aligned RMSD between a structure and a rigid-body-moved copy of itself is zero."""
    from scipy.spatial.transform import Rotation
    from kate.kinetic_codec import kabsch_align
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((12, 3))
    moved = ref @ Rotation.random(random_state=8).as_matrix().T + np.array([3.0, -2.0, 1.0])
    aligned, _ = kabsch_align(np.stack([ref, moved], 0).astype(np.float64))
    rmsd = float(np.sqrt(((aligned[1] - aligned[0]) ** 2).sum(1).mean()))
    assert rmsd < 1e-8, "aligned RMSD %.2e is not ~0" % rmsd


def test_pathbound_transition_term_and_pinsker():
    """The transition term equals the hand-computed relative entropy, the ensemble term
    vanishes for identical stationary distributions, and the Pinsker pair bound upper-
    bounds a genuine bounded observable's difference between two dynamics."""
    from kate.pathbound import (transition_kl_rate, ensemble_kl, pinsker, two_slice_kl,
                                stationary_distribution)
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    Q = np.array([[0.7, 0.3], [0.25, 0.75]])
    mu = stationary_distribution(P)
    hand = float((mu[:, None] * P * np.log(P / Q)).sum())
    assert abs(transition_kl_rate(P, Q) - hand) < 1e-9
    assert ensemble_kl(mu, mu) < 1e-12
    total, _, _ = two_slice_kl(P, Q)
    muq = stationary_distribution(Q)
    g = np.array([[0.0, 1.0], [1.0, 0.0]])                      # a bounded [0,1] pair observable
    diff = abs((mu[:, None] * P * g).sum() - (muq[:, None] * Q * g).sum())
    assert diff <= pinsker(total) + 1e-9, "Pinsker bound %.4f violated by |dE|=%.4f" % (pinsker(total), diff)
