"""
glide_pathbound.py
================
The kinetic half of the GLIDE guarantee: a divergence on the TRAJECTORY (path)
distribution, not just the static ensemble. This is the module that turns the
abstract's "analytic guarantee" into something that also covers KINETIC
observables -- the part the ensemble (static) Pinsker bound does NOT cover.

Why this module exists
----------------------
The ensemble bound in glide.py controls observables of single configurations:
    |E_P[f] - E_Q[f]| <= sqrt( D(mu_P || mu_Q) / 2 )      for  f in [0,1].
It says NOTHING about kinetics. Two ensembles with IDENTICAL stationary
distributions can have arbitrarily different transition rates -- so preserving
mu (what an ensemble-matching autoencoder does, e.g. matching a CV histogram or
TICA-projection distribution) does NOT preserve k_on/k_off, MFPTs, or implied
timescales.

For a Markov model at lag tau the path-distribution KL factorizes EXACTLY into
an ENSEMBLE term + a TRANSITION (dynamics) term. With the lag-tau joint
    rho_P(i,j) = mu_P(i) P(i,j),     rho_Q(i,j) = mu_Q(i) Q(i,j),
    D(rho_P || rho_Q) = D(mu_P || mu_Q)                       (ensemble term)
                      + sum_i mu_P(i) D( P(i,.) || Q(i,.) )    (transition term).
Over a full trajectory of L consecutive frames (stationary Markov),
    D(path_P || path_Q) = D(mu_P || mu_Q) + (L-1) * h(P||Q),
    h(P||Q) = sum_i mu_P(i) sum_j P(i,j) log( P(i,j) / Q(i,j) )   (nats/step).

Pinsker on the joint then bounds ANY bounded observable of consecutive pairs
(x_t, x_{t+tau}):
    |E_P[g] - E_Q[g]| <= sqrt( D(rho_P || rho_Q) / 2 ),   g in [0,1].
Transition fluxes / counts -- which determine rates -- are exactly such pairwise
observables, so this IS a kinetic guarantee. The ensemble term alone is not.

How GLIDE uses it
---------------
GLIDE retains the MSM transition matrix, so for the GLIDE artifact itself Q = P on
the kept dynamics and the transition term is ~0 by construction -- i.e. GLIDE
preserves kinetics. The module's real job is to be the MEASURING STICK for the
contrast experiment: take ANY compressor's reconstruction, re-estimate its MSM
Q at the same discretization, and report D(mu_P||mu_Q) (ensemble) vs the
transition term (kinetics). An ensemble-only method shows ensemble ~ 0 but
transition > 0 -- kinetics corrupted -- which the static bound would have
WRONGLY certified as faithful. That contrast is the paper's central point.

Honest scope
------------
The factorization assumes a Markov model at the chosen lag (the MSM assumption
you already make for kinetics) and stationary statistics. Report it ALONGSIDE an
implied-timescale lag scan, not as a lag-independent certificate. All KLs are in
NATS. Pure numpy; no torch, no deeptime.
"""

from __future__ import annotations
import numpy as np


def _row_normalize(C):
    C = np.asarray(C, dtype=np.float64)
    rs = C.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return C / rs


def stationary_distribution(P):
    """Stationary vector of a row-stochastic P (left eigenvector, eigenvalue 1)."""
    P = np.asarray(P, dtype=np.float64)
    evals, evecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.abs(np.real(evecs[:, i]))
    s = pi.sum()
    return pi / s if s > 0 else np.ones(P.shape[0]) / P.shape[0]


def ensemble_kl(mu_p, mu_q, eps=1e-12):
    """D(mu_P || mu_Q) in NATS -- the static / ensemble term."""
    p = np.clip(np.asarray(mu_p, float), eps, None); p = p / p.sum()
    q = np.clip(np.asarray(mu_q, float), eps, None); q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def transition_kl_rate(P, Q, mu_p=None, eps=1e-12):
    """Transition (dynamics) term  h(P||Q) = sum_i mu_P(i) sum_j P_ij log(P_ij/Q_ij),
    in NATS PER STEP. mu_P defaults to P's stationary distribution. Q must place
    mass wherever P does (absolute continuity); we clip Q to keep the result
    finite. If Q has structural zeros where P does not, the true divergence is
    +inf and the clipped value is a (large) lower bound -- check `support_ok`."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if mu_p is None:
        mu_p = stationary_distribution(P)
    Pc = np.clip(P, eps, None)
    Qc = np.clip(Q, eps, None)
    row = (Pc * np.log(Pc / Qc)).sum(axis=1)          # D(P(i,.)||Q(i,.)) per state
    return float((np.asarray(mu_p, float) * row).sum())


def support_ok(P, Q, eps=1e-12):
    """True if Q is positive wherever P is (so the path KL is finite)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    return bool(np.all(Q[P > eps] > eps))


def two_slice_kl(P, Q, mu_p=None, mu_q=None):
    """D( rho_P || rho_Q ) for the lag-tau joint rho(i,j) = mu(i) P(i,j).
    Returns (total, ensemble_term, transition_term), all in nats. This is the
    divergence whose Pinsker bound covers observables of (x_t, x_{t+tau}) pairs,
    i.e. transition fluxes / rates."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if mu_p is None: mu_p = stationary_distribution(P)
    if mu_q is None: mu_q = stationary_distribution(Q)
    ens = ensemble_kl(mu_p, mu_q)
    tran = transition_kl_rate(P, Q, mu_p)
    return ens + tran, ens, tran


def path_kl(P, Q, L, mu_p=None, mu_q=None):
    """KL between path measures of L consecutive frames (stationary Markov):
    D(path) = D(mu_P||mu_Q) + (L-1) h(P||Q), in nats. It GROWS with trajectory
    length -- kinetic error accumulates over a trajectory, while ensemble error
    does not."""
    if mu_p is None: mu_p = stationary_distribution(P)
    if mu_q is None: mu_q = stationary_distribution(Q)
    ens = ensemble_kl(mu_p, mu_q)
    tran = transition_kl_rate(P, Q, mu_p)
    return float(ens + max(int(L) - 1, 0) * tran)


def pinsker(kl_nats):
    """Bounded-observable / total-variation bound: |E_P[g]-E_Q[g]| <= sqrt(KL/2)
    for g in [0,1], KL in nats."""
    return float(np.sqrt(max(float(kl_nats), 0.0) / 2.0))


def implied_timescales(P, lag=1, k=5):
    ev = np.sort(np.real(np.linalg.eigvals(np.asarray(P, float))))[::-1]
    ev = np.clip(ev[1:k + 1], 1e-12, 0.999999)
    return -lag / np.log(ev)


def report_kinetic_fidelity(P_ref, Q_cmp, lag=1, L=None,
                            mu_ref=None, mu_cmp=None, k=4):
    """Compare a reference dynamics P_ref against a compressed/reconstructed
    dynamics Q_cmp at a given lag. P_ref, Q_cmp may be transition matrices or raw
    count matrices (row-normalized here). Returns the ensemble term, transition
    term, two-slice KL + Pinsker pair bound, optional path KL, the support check,
    and the implied timescales of both (the kinetic observable that must match).
    """
    P_ref = _row_normalize(P_ref); Q_cmp = _row_normalize(Q_cmp)
    if mu_ref is None: mu_ref = stationary_distribution(P_ref)
    if mu_cmp is None: mu_cmp = stationary_distribution(Q_cmp)
    total, ens, tran = two_slice_kl(P_ref, Q_cmp, mu_ref, mu_cmp)
    out = {
        "lag": lag,
        "support_ok": support_ok(P_ref, Q_cmp),
        "ensemble_kl_nats": ens,
        "transition_kl_rate_nats_per_step": tran,
        "two_slice_kl_nats": total,
        "pinsker_pair_bound": pinsker(total),
        "pinsker_ensemble_bound": pinsker(ens),
        "its_ref": implied_timescales(P_ref, lag, k),
        "its_cmp": implied_timescales(Q_cmp, lag, k),
    }
    if L is not None:
        out["path_kl_nats"] = path_kl(P_ref, Q_cmp, L, mu_ref, mu_cmp)
        out["pinsker_path_bound"] = pinsker(out["path_kl_nats"])
    return out
