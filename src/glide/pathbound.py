"""
Kinetic Path-Distribution Bound for the GLIDE Guarantee
=======================================================
Background
----------
This module provides the kinetic component of the GLIDE guarantee: a divergence
defined on the trajectory (path) distribution rather than only on the static
ensemble. It extends the analytic guarantee to kinetic observables, which the
static ensemble Pinsker bound does not cover.

Motivation
----------
The ensemble bound implemented in glide.py controls observables of single
configurations,

    |E_P[f] - E_Q[f]| <= sqrt( D(mu_P || mu_Q) / 2 )    for f in [0, 1],

but it provides no control over kinetics. Two ensembles with identical stationary
distributions can have arbitrarily different transition rates, so preserving the
stationary distribution mu (as an ensemble-matching autoencoder does when it
matches a collective-variable histogram or a TICA-projection distribution) does
not preserve k_on, k_off, mean first-passage times, or implied timescales.

Path-KL factorization
---------------------
For a Markov model at lag tau, the path-distribution KL divergence factorizes
exactly into an ensemble term and a transition (dynamics) term. With the lag-tau
joint distributions

    rho_P(i,j) = mu_P(i) P(i,j),    rho_Q(i,j) = mu_Q(i) Q(i,j),

the divergence is

    D(rho_P || rho_Q) = D(mu_P || mu_Q)                       (ensemble term)
                      + sum_i mu_P(i) D( P(i,.) || Q(i,.) )    (transition term).

Over a full trajectory of L consecutive frames under stationary Markov dynamics,

    D(path_P || path_Q) = D(mu_P || mu_Q) + (L - 1) h(P||Q),
    h(P||Q) = sum_i mu_P(i) sum_j P(i,j) log( P(i,j) / Q(i,j) )    [nats/step].

The Pinsker inequality applied to the joint then bounds any bounded observable of
consecutive pairs (x_t, x_{t+tau}),

    |E_P[g] - E_Q[g]| <= sqrt( D(rho_P || rho_Q) / 2 )    for g in [0, 1].

Transition fluxes and counts, which determine rates, are exactly such pairwise
observables, so this constitutes a kinetic guarantee; the ensemble term alone does
not.

Role within GLIDE
----------------
GLIDE retains the MSM transition matrix, so for the GLIDE artifact itself Q = P on
the retained dynamics and the transition term is approximately zero by
construction, i.e. GLIDE preserves kinetics. The principal role of this module is
to serve as the reference measure for the contrast experiment: given any
compressor's reconstruction, its MSM Q is re-estimated at the same discretization,
and the ensemble term D(mu_P || mu_Q) is reported against the transition term. An
ensemble-only method shows an ensemble term near zero but a positive transition
term, indicating corrupted kinetics that the static bound would have incorrectly
certified as faithful. This contrast is the central result of the accompanying
work.

Scope and conventions
--------------------
The factorization assumes a Markov model at the chosen lag (the MSM assumption
already made for kinetics) together with stationary statistics. The result should
be reported alongside an implied-timescale lag scan rather than as a
lag-independent certificate. All KL divergences are expressed in nats. The
implementation uses only numpy; torch and deeptime are not required.
"""

from __future__ import annotations
import numpy as np


def _row_normalize(C):
    C = np.asarray(C, dtype=np.float64)
    rs = C.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return C / rs


def stationary_distribution(P):
    """Compute the stationary vector of a row-stochastic matrix P.

    The stationary vector is the left eigenvector of P associated with eigenvalue
    one."""
    P = np.asarray(P, dtype=np.float64)
    evals, evecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.abs(np.real(evecs[:, i]))
    s = pi.sum()
    return pi / s if s > 0 else np.ones(P.shape[0]) / P.shape[0]


def ensemble_kl(mu_p, mu_q, eps=1e-12):
    """Compute the ensemble term D(mu_P || mu_Q) in nats.

    This is the static ensemble contribution to the path divergence."""
    p = np.clip(np.asarray(mu_p, float), eps, None); p = p / p.sum()
    q = np.clip(np.asarray(mu_q, float), eps, None); q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def transition_kl_rate(P, Q, mu_p=None, eps=1e-12):
    """Compute the transition (dynamics) term h(P||Q) in nats per step.

    The transition term is

        h(P||Q) = sum_i mu_P(i) sum_j P_ij log( P_ij / Q_ij )    [nats/step].

    By default mu_P is the stationary distribution of P. Absolute continuity
    requires Q to place mass wherever P does; Q is clipped to keep the result
    finite. If Q has structural zeros where P does not, the true divergence is
    infinite and the clipped value is a large lower bound; the `support_ok`
    predicate should be used to detect this case."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if mu_p is None:
        mu_p = stationary_distribution(P)
    Pc = np.clip(P, eps, None)
    Qc = np.clip(Q, eps, None)
    row = (Pc * np.log(Pc / Qc)).sum(axis=1)          # per-state D(P(i,.)||Q(i,.))
    return float((np.asarray(mu_p, float) * row).sum())


def support_ok(P, Q, eps=1e-12):
    """Return True if Q is positive wherever P is, so the path KL is finite."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    return bool(np.all(Q[P > eps] > eps))


def two_slice_kl(P, Q, mu_p=None, mu_q=None):
    """Compute D(rho_P || rho_Q) for the lag-tau joint rho(i,j) = mu(i) P(i,j).

    This is the divergence whose Pinsker bound covers observables of
    (x_t, x_{t+tau}) pairs, that is, transition fluxes and rates.

    Returns
    -------
    tuple of float
        The total divergence, the ensemble term, and the transition term, all in
        nats.
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if mu_p is None: mu_p = stationary_distribution(P)
    if mu_q is None: mu_q = stationary_distribution(Q)
    ens = ensemble_kl(mu_p, mu_q)
    tran = transition_kl_rate(P, Q, mu_p)
    return ens + tran, ens, tran


def path_kl(P, Q, L, mu_p=None, mu_q=None):
    """Compute the KL divergence between path measures of L consecutive frames.

    Under stationary Markov dynamics,

        D(path) = D(mu_P || mu_Q) + (L - 1) h(P||Q)    [nats].

    The divergence grows with trajectory length, reflecting that kinetic error
    accumulates over a trajectory whereas ensemble error does not."""
    if mu_p is None: mu_p = stationary_distribution(P)
    if mu_q is None: mu_q = stationary_distribution(Q)
    ens = ensemble_kl(mu_p, mu_q)
    tran = transition_kl_rate(P, Q, mu_p)
    return float(ens + max(int(L) - 1, 0) * tran)


def pinsker(kl_nats):
    """Apply the Pinsker bounded-observable bound.

    For g in [0, 1] and a KL divergence in nats, the bound is

        |E_P[g] - E_Q[g]| <= sqrt( KL / 2 ).
    """
    return float(np.sqrt(max(float(kl_nats), 0.0) / 2.0))


def implied_timescales(P, lag=1, k=5):
    ev = np.sort(np.real(np.linalg.eigvals(np.asarray(P, float))))[::-1]
    ev = np.clip(ev[1:k + 1], 1e-12, 0.999999)
    return -lag / np.log(ev)


def report_kinetic_fidelity(P_ref, Q_cmp, lag=1, L=None,
                            mu_ref=None, mu_cmp=None, k=4):
    """Compare a reference dynamics against a reconstructed dynamics at a lag.

    The reference dynamics P_ref and the compressed or reconstructed dynamics Q_cmp
    may be supplied as transition matrices or as raw count matrices, which are
    row-normalized here.

    Returns
    -------
    dict
        A report containing the ensemble term, the transition term, the two-slice
        KL with its Pinsker pair bound, an optional path KL, the support check, and
        the implied timescales of both dynamics, the latter being the kinetic
        observable that must match.
    """
    P_ref = _row_normalize(P_ref); Q_cmp = _row_normalize(Q_cmp)
    if mu_ref is None: mu_ref = stationary_distribution(P_ref)
    if mu_cmp is None: mu_cmp = stationary_distribution(Q_cmp)
    total, ens, tran = two_slice_kl(P_ref, Q_cmp, mu_ref, mu_cmp)
    ok = support_ok(P_ref, Q_cmp)
    out = {
        "lag": lag,
        "support_ok": ok,
        # When the support check fails, Q has a structural zero where P is positive
        # (for example a transition the compressor never reproduces), so the true
        # path divergence is infinite. The clipped transition, pair, and path values
        # below are then only lower bounds and the Pinsker bounds do not hold. The
        # field `kinetic_bound_valid` indicates whether these values may be trusted.
        "kinetic_bound_valid": ok,
        "ensemble_kl_nats": ens,
        "transition_kl_rate_nats_per_step": tran,
        "transition_kl_is_lower_bound": not ok,
        "two_slice_kl_nats": total,
        "pinsker_pair_bound": pinsker(total) if ok else float("inf"),
        "pinsker_ensemble_bound": pinsker(ens),   # ensemble term has no support issues
        "its_ref": implied_timescales(P_ref, lag, k),
        "its_cmp": implied_timescales(Q_cmp, lag, k),
    }
    if L is not None:
        out["path_kl_nats"] = path_kl(P_ref, Q_cmp, L, mu_ref, mu_cmp)
        out["pinsker_path_bound"] = pinsker(out["path_kl_nats"])
    return out
