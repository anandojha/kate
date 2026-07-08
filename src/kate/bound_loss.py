"""
Differentiable Kinetic Path-Bound Loss
======================================

Background
----------
This module makes the kinetic path-distribution bound differentiable, so that it may
serve as a training loss rather than only as a post hoc scorer. It enables a neural
compressor to be trained to preserve kinetic observables, specifically the transition
term of the path bound, in place of coordinate mean-squared error.

A VAMPnet-style soft state assignment chi(x) in the probability simplex renders the
microstate populations, and therefore the transition matrix, a smooth, differentiable
function of the network (VAMPnets: Mardt et al., Nat. Commun. 9, 5 (2018); deep
generalized MSMs: Wu and Noe, J. Nonlinear Sci. 30, 23 (2020)). From soft assignments
at times t and t + tau,

    C00 = <chi_t chi_t^T>,  C01 = <chi_t chi_{t+tau}^T>,  C11 = <chi_{t+tau} chi_{t+tau}^T>
    T     = C00^{-1} C01                              (row-normalized soft transition matrix)
    VAMP2 = || C00^{-1/2} C01 C11^{-1/2} ||_F^2       (slow-dynamics score for pretraining chi)

The path-bound transition term between a reference dynamics P and a compressed dynamics
Q, h(P||Q) = sum_i pi_i sum_j P_ij log(P_ij / Q_ij), is then differentiable end-to-end,
so that loss = rate + lambda * h(P||Q) trains a compressor to allocate bits where they
affect the kinetics. All computation here uses pure torch autograd; deeptime is not
required.

Scope
-----
The quantity defined here is a differentiable surrogate of the bound, combining soft
states with a regression transition matrix, used as a loss. The reported, certified
kinetics are obtained separately from the deeptime reversible-maximum-likelihood MSM
together with the path bound evaluated on hard states (kate.pathbound, kate.analyze).
Whether training on the bound improves upon training on mean-squared error is an
empirical question, measured rather than assumed (see the examples and the T10
experiment).
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
    """Compute the VAMP-2 score of a soft assignment.

    A higher score indicates that more of the slow dynamics is captured. The score is
    used to pretrain the soft-state encoder so that its readout is kinetically
    meaningful (VAMPnets: Mardt et al., Nat. Commun. 9, 5 (2018)).
    """
    c00, c01, c11 = soft_covariances(chi0, chi1)
    n = c00.shape[0]
    eye = torch.eye(n, dtype=c00.dtype, device=c00.device)
    K = _inv_sqrt(c00 + eps * eye) @ c01 @ _inv_sqrt(c11 + eps * eye)
    return (K ** 2).sum()


def soft_transition_matrix(chi0, chi1, eps=1e-6):
    """Compute the differentiable soft MSM transition matrix T = C00^{-1} C01.

    The result is clamped and row-normalized to yield a proper row-stochastic
    transition matrix.
    """
    c00, c01, _ = soft_covariances(chi0, chi1)
    n = c00.shape[0]
    eye = torch.eye(n, dtype=c00.dtype, device=c00.device)
    T = torch.linalg.solve(c00 + eps * eye, c01)
    T = torch.clamp(T, min=eps)
    return T / T.sum(dim=1, keepdim=True)


def stationary(T, n_iter=100):
    """Compute the stationary distribution of T by differentiable power iteration."""
    n = T.shape[0]
    pi = torch.full((n,), 1.0 / n, dtype=T.dtype, device=T.device)
    for _ in range(n_iter):
        pi = pi @ T
        pi = pi / pi.sum()
    return pi


def transition_term(P, Q, pi=None, eps=1e-8):
    """Compute the path-bound transition term h(P||Q).

    The term is h(P||Q) = sum_i pi_i sum_j P_ij log(P_ij / Q_ij), expressed in nats per
    step, and constitutes the differentiable kinetic distortion. P and Q are
    row-stochastic; pi defaults to the stationary distribution of P.
    """
    if pi is None:
        pi = stationary(P).detach()         # weight by the populations of P (stop-gradient)
    Pc = torch.clamp(P, eps, 1.0)
    Qc = torch.clamp(Q, eps, 1.0)
    row = (Pc * torch.log(Pc / Qc)).sum(dim=1)
    return (pi * row).sum()


def kinetic_distortion(chi_ref, chi_cmp, lag):
    """Compute the differentiable kinetic distortion between two soft trajectories.

    The reference soft trajectory chi_ref and the compressed soft trajectory chi_cmp are
    produced by the same soft-state encoder applied to the original and compressed inputs
    respectively. The distortion is the transition term between their soft MSMs at the
    given lag.
    """
    P = soft_transition_matrix(*_lagged(chi_ref, lag))
    Q = soft_transition_matrix(*_lagged(chi_cmp, lag))
    return transition_term(P, Q)


class SoftStateEncoder(nn.Module):
    """Soft-state encoder: a small MLP with a softmax over n_states.

    The module is a VAMPnet lobe that outputs a soft state assignment, namely a
    probability vector per frame (VAMPnets: Mardt et al., Nat. Commun. 9, 5 (2018)). It
    is pretrained by maximizing the VAMP-2 score.
    """

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
        """Pretrain the encoder by maximizing the VAMP-2 score to capture the slow dynamics."""
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
