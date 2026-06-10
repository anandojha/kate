"""
demo_pathbound.py
=================
The central thesis, made measurable: preserving the ENSEMBLE does not preserve
KINETICS, and the path-distribution decomposition (epc_pathbound) detects the
difference the static ensemble bound misses.

Setup: a 3-state reversible chain P_ref (the "truth"). We build two "compressed"
dynamics that share the SAME stationary distribution mu:
  Q_slow : off-diagonal flow scaled by c < 1  (mu identical, kinetics slower) --
           what an ensemble-only method can produce: right populations, wrong
           rates.
  Q_good : a tiny perturbation of P_ref -- what EPC's retained MSM gives.

It prints:
  - ensemble term D(mu||mu) ~ 0 for BOTH -> the static bound would certify both;
  - the TRANSITION term, large for Q_slow and ~0 for Q_good -> only the path
    decomposition flags that Q_slow's kinetics are wrong;
  - implied timescales: Q_slow off by ~1/c, Q_good matching.

Exact (matrices, not sampling), so the numbers are the mechanism, not an
estimate.
"""
import numpy as np
from epc.pathbound import (report_kinetic_fidelity, stationary_distribution)


def _stochastic(P):
    P = np.clip(np.asarray(P, float), 0, None)
    return P / P.sum(axis=1, keepdims=True)


def reversible_chain(mu, s_off):
    """Reversible row-stochastic P with stationary mu, from a symmetric
    off-diagonal 'flow' matrix s_off (s_off[i,j]=s_off[j,i]>=0, i!=j):
    S symmetric with S_ii = mu_i - sum_{j!=i} S_ij, then P_ij = S_ij/mu_i.
    Requires the diagonal to stay >= 0 (s_off small enough)."""
    mu = np.asarray(mu, float)
    S = np.array(s_off, float); np.fill_diagonal(S, 0.0)
    S = 0.5 * (S + S.T)
    diag = mu - S.sum(axis=1)
    assert (diag >= -1e-12).all(), "off-diagonal flow too large; diagonal < 0"
    np.fill_diagonal(S, np.clip(diag, 0.0, None))
    return _stochastic(S / mu[:, None])


def scale_kinetics(mu, s_off, c):
    """Same mu, off-diagonal flow scaled by c -> slower (c<1) / faster (c>1)
    kinetics with an identical stationary distribution."""
    return reversible_chain(mu, c * np.asarray(s_off, float))


def main():
    np.set_printoptions(precision=4, suppress=True)
    mu = np.array([0.5, 0.3, 0.2])
    s = np.array([[0.00, 0.04, 0.01],
                  [0.04, 0.00, 0.03],
                  [0.01, 0.03, 0.00]])
    P_ref = reversible_chain(mu, s)
    c = 0.3
    Q_slow = scale_kinetics(mu, s, c)              # ensemble-faithful, kinetics wrong
    rng = np.random.default_rng(0)
    Q_good = _stochastic(P_ref + 1e-3 * rng.random(P_ref.shape))   # ~ EPC retained MSM

    print("=" * 72)
    print("ENSEMBLE PRESERVED, KINETICS NOT -- detecting it with the path bound")
    print("=" * 72)
    print("shared stationary mu:", mu)
    print("  mu(P_ref) :", np.round(stationary_distribution(P_ref), 4))
    print("  mu(Q_slow):", np.round(stationary_distribution(Q_slow), 4),
          "  <- identical to mu")
    print("  mu(Q_good):", np.round(stationary_distribution(Q_good), 4))

    for name, Q in [("Q_slow  (ensemble-only; rates x%.1f)" % c, Q_slow),
                    ("Q_good  (retained MSM ~ EPC)", Q_good)]:
        r = report_kinetic_fidelity(P_ref, Q, lag=1, L=10000, k=2)
        print("-" * 72)
        print(name)
        print("  ensemble term  D(mu_P||mu_Q)   : %.3e nats   (static bound sees ONLY this)"
              % r["ensemble_kl_nats"])
        print("  transition term h(P||Q)        : %.4e nats/step   (the kinetic signal)"
              % r["transition_kl_rate_nats_per_step"])
        print("  two-slice KL (pairs)           : %.4e nats  ->  Pinsker pair bound %.4f"
              % (r["two_slice_kl_nats"], r["pinsker_pair_bound"]))
        print("  path KL over 10,000 frames     : %.2f nats" % r["path_kl_nats"])
        print("  implied timescale ref / cmp    : %.2f / %.2f frames"
              % (r["its_ref"][0], r["its_cmp"][0]))
        print("  static (ensemble) Pinsker      : %.4e  -> would certify as 'faithful'"
              % r["pinsker_ensemble_bound"])
    print("=" * 72)
    print("Takeaway: the ensemble term is ~0 for BOTH, so the STATIC guarantee")
    print("certifies Q_slow as faithful -- yet its slowest timescale is wrong by")
    print("~1/c = %.1fx. The transition term exposes exactly that gap; EPC retains" % (1.0 / c))
    print("the MSM to keep it near zero, which is the kinetic half of the bound.")
    print("=" * 72)


if __name__ == "__main__":
    main()
