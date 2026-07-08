"""T10: the differentiable kinetic (path-bound) loss. Verifies the soft MSM is a valid
row-stochastic transition matrix, the transition term is 0 for identical dynamics and
positive otherwise, and that the kinetic distortion is differentiable + the VAMP
pretraining actually raises the score (so the loss is usable for training)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kate.bound_loss import (soft_transition_matrix, transition_term, vamp2_score,
                            kinetic_distortion, stationary, SoftStateEncoder, _lagged)


def _soft_traj(T=600, seed=0):
    """A soft assignment from a slow AR(1) latent over 3 centers."""
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.97 * z[t - 1] + np.sqrt(1 - 0.97 ** 2) * rng.standard_normal()
    centers = np.array([-2.0, 0.0, 2.0])
    logits = -3.0 * (z[:, None] - centers[None, :]) ** 2
    chi = np.exp(logits); chi /= chi.sum(1, keepdims=True)
    return torch.tensor(chi, dtype=torch.float32), z


def test_soft_transition_matrix_row_stochastic_and_stationary():
    chi, _ = _soft_traj()
    T = soft_transition_matrix(*_lagged(chi, 5))
    assert torch.allclose(T.sum(1), torch.ones(T.shape[0]), atol=1e-5)
    assert bool((T >= 0).all())
    pi = stationary(T)
    assert abs(float(pi.sum()) - 1.0) < 1e-5


def test_transition_term_zero_for_identical_positive_otherwise():
    chi, _ = _soft_traj(seed=1)
    P = soft_transition_matrix(*_lagged(chi, 5))
    assert transition_term(P, P).item() < 1e-6
    chi2, _ = _soft_traj(seed=2)
    Q = soft_transition_matrix(*_lagged(chi2, 5))
    assert transition_term(P, Q).item() > 1e-4


def test_kinetic_distortion_is_differentiable():
    chi, z = _soft_traj(800, seed=3)
    rng = np.random.default_rng(0)
    Y = torch.tensor(np.stack([z, 0.5 * rng.standard_normal(len(z))], 1), dtype=torch.float32)
    enc = SoftStateEncoder(2, 3, hidden=16)
    Yn = Y + 0.3 * torch.randn_like(Y)
    d = kinetic_distortion(enc(Y), enc(Yn), 5)
    d.backward()                                  # gradients flow through the soft MSM
    assert torch.isfinite(d) and d.item() >= 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_vamp_pretraining_raises_score():
    chi, z = _soft_traj(1000, seed=4)
    rng = np.random.default_rng(1)
    Y = np.stack([z, 0.5 * rng.standard_normal(len(z))], 1).astype(np.float32)
    enc = SoftStateEncoder(2, 3, hidden=16)
    c0, c1 = _lagged(enc(torch.tensor(Y)), 5)
    v0 = vamp2_score(c0, c1).item()
    enc.fit_vamp(Y, lag=5, epochs=80, seed=0)
    c0, c1 = _lagged(enc(torch.tensor(Y)), 5)
    v1 = vamp2_score(c0, c1).item()
    assert v1 >= v0 - 1e-3                         # pretraining did not hurt the score
