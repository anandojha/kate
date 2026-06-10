"""
bound_loss.py
=============
T10 -- the kinetic (path-distribution) bound made DIFFERENTIABLE, so it can be used as
a TRAINING LOSS, not just an after-the-fact scorer. This is the one place where the ML
and the novel idea become the same thing: a neural compressor trained to preserve
KINETIC observables (the transition term of the path bound), not coordinate MSE.

The machinery: a VAMPnet-style SOFT state assignment chi(x) in the probability simplex
makes the microstate populations -- and therefore the transition matrix -- a smooth,
differentiable function of the network (Mardt & Noe 2018; Wu & Noe 2020). From soft
assignments at time t and t+tau:

    C00 = <chi_t chi_t^T>,  C01 = <chi_t chi_{t+tau}^T>,  C11 = <chi_{t+tau} chi_{t+tau}^T>
    T   = C00^{-1} C01      (the soft transition matrix, row-normalized)
    VAMP2 = || C00^{-1/2} C01 C11^{-1/2} ||_F^2   (the slow-dynamics score to pretrain chi)

The path-bound transition term between a reference dynamics P and a compressed dynamics
Q, h(P||Q) = sum_i pi_i sum_j P_ij log(P_ij/Q_ij), is then differentiable end-to-end --
so `loss = rate + lambda * h(P||Q)` trains a compressor to spend bits where they matter
for KINETICS. Everything here is pure torch (autograd); deeptime is not needed.

Honest scope: this is the differentiable SURROGATE of the bound (soft states + a
regression transition matrix), used as a loss. The reported, certified kinetics still
come from the deeptime reversible-MLE MSM + the path bound on hard states (epc.pathbound
/ epc.analyze). Whether training on the bound actually beats training on MSE is an
EMPIRICAL question -- measured, not assumed (see examples / the T10 experiment).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _lagged(chi, lag):
    """Split a (T, n) soft-assignment sequence into instantaneous/time-lagged pairs."""
    return chi[:-lag], chi[lag:]


def soft_covariances(chi0, chi1):
    n = chi0.shape[0]
    c00 = chi0.t() @ chi0 / n
    c01 = chi0.t() @ chi1 / n
    c11 = chi1.t() @ chi1 / n
    return c00, c01, c11


def _inv_sqrt(C, eps=1e-6):
    w, V = torch.linalg.eigh(C)
    w = torch.clamp(w, min=eps)
    return (V * w.rsqrt()) @ V.t()


def vamp2_score(chi0, chi1, eps=1e-6):
    """VAMP-2 score of a soft assignment (higher = captures more slow dynamics). Used to
    PRETRAIN the soft-state encoder so the readout is kinetically meaningful."""
    c00, c01, c11 = soft_covariances(chi0, chi1)
    n = c00.shape[0]
    eye = torch.eye(n, dtype=c00.dtype, device=c00.device)
    K = _inv_sqrt(c00 + eps * eye) @ c01 @ _inv_sqrt(c11 + eps * eye)
    return (K ** 2).sum()


def soft_transition_matrix(chi0, chi1, eps=1e-6):
    """Differentiable soft MSM T = C00^{-1} C01, clamped + row-normalized to a proper
    row-stochastic transition matrix."""
    c00, c01, _ = soft_covariances(chi0, chi1)
    n = c00.shape[0]
    eye = torch.eye(n, dtype=c00.dtype, device=c00.device)
    T = torch.linalg.solve(c00 + eps * eye, c01)
    T = torch.clamp(T, min=eps)
    return T / T.sum(dim=1, keepdim=True)


def stationary(T, n_iter=100):
    """Stationary distribution by power iteration on T (differentiable)."""
    n = T.shape[0]
    pi = torch.full((n,), 1.0 / n, dtype=T.dtype, device=T.device)
    for _ in range(n_iter):
        pi = pi @ T
        pi = pi / pi.sum()
    return pi


def transition_term(P, Q, pi=None, eps=1e-8):
    """The path-bound TRANSITION term h(P||Q) = sum_i pi_i sum_j P_ij log(P_ij/Q_ij), in
    nats/step -- the differentiable KINETIC distortion. P, Q are row-stochastic; pi
    defaults to P's stationary distribution."""
    if pi is None:
        pi = stationary(P).detach()         # weight by P's populations (stop-grad)
    Pc = torch.clamp(P, eps, 1.0)
    Qc = torch.clamp(Q, eps, 1.0)
    row = (Pc * torch.log(Pc / Qc)).sum(dim=1)
    return (pi * row).sum()


def kinetic_distortion(chi_ref, chi_cmp, lag):
    """Differentiable kinetic distortion between a REFERENCE soft trajectory chi_ref and
    a COMPRESSED one chi_cmp (same soft-state encoder, applied to original vs compressed
    inputs): the transition term between their soft MSMs at `lag`."""
    P = soft_transition_matrix(*_lagged(chi_ref, lag))
    Q = soft_transition_matrix(*_lagged(chi_cmp, lag))
    return transition_term(P, Q)


class SoftStateEncoder(nn.Module):
    """A small MLP -> softmax over n_states: a VAMPnet lobe that outputs a SOFT state
    assignment (a probability vector per frame). Pretrain by maximizing vamp2_score."""

    def __init__(self, in_dim, n_states, hidden=32, n_layers=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ELU()]
            d = hidden
        layers += [nn.Linear(d, n_states)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def fit_vamp(self, Y, lag, epochs=200, lr=1e-3, seed=0, verbose=False):
        """Pretrain to maximize the VAMP-2 score (capture the slow dynamics)."""
        torch.manual_seed(seed)
        Y = torch.as_tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(epochs):
            chi = self.forward(Y)
            c0, c1 = _lagged(chi, lag)
            loss = -vamp2_score(c0, c1)
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print("  vamp pretrain epoch %4d  VAMP2 = %.4f" % (ep, -loss.item()))
        return self
