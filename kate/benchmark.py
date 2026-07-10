"""
The T3 kinetic-fidelity contrast, KATE's central benchmark figure.

Each compression method is scored by the path-space divergence between its
recovered Markov dynamics and the original MSM. For a Markov model at lag tau the
lag-tau joint rho(i,j) = mu(i) P(i,j) splits the KL divergence into a static and a
dynamic part,

    D(rho_P || rho_Q) = D(mu_P || mu_Q) + sum_i mu_P(i) D( P(i,.) || Q(i,.) ),

the ensemble term D(mu_P || mu_Q) in nats and the transition term
h(P||Q) = sum_i mu_P(i) sum_j P_ij log(P_ij / Q_ij) in nats/step (kate.pathbound).
Here P is the reference transition matrix, Q the method's re-estimated one, and mu
the matching stationary vector. Rates, fluxes, and implied timescales are
observables of consecutive pairs (x_t, x_{t+tau}), so they are controlled by the
transition term and left free by the ensemble term alone.

The pipeline holds the featurization fixed so every method is judged on the same
footing. The reference trajectory is Kabsch-aligned, projected onto a common TICA,
and discretized against one set of k-means centers, and a reversible MSM is
estimated on it. Each baseline then reconstructs coordinates, is pushed through the
same TICA and centers, and gets its own reversible MSM. KATE carries its MSM
through the artifact, so for KATE Q = P and the transition term vanishes by
construction.

Because the path bound compares P and Q entrywise, both must share one state
indexing. A single active set is fixed as the largest connected set of the
reference counts, and every method's MSM is estimated on exactly those states via
estimate_reversible_T, the reversible maximum-likelihood estimator (deeptime) when
installed and the (C + C^T)/2 fallback otherwise, so the contrast runs with no
extra dependencies.

The reading: an ensemble-only or coordinate-bounded method keeps a small ensemble
term but drifts to a large transition term, its implied timescales wandering off
the reference, whereas KATE's transition term is ~0. A static bound that saw only
the ensemble term would wrongly certify the drifting methods as faithful.
"""
from __future__ import annotations

import numpy as np

from .kinetic_codec import (kabsch_align, TICA, discretize, count_matrix,
                            estimate_reversible_T, largest_connected_set,
                            implied_timescales)
from .pathbound import report_kinetic_fidelity
from . import baselines


def _assign(CV, centers):
    """Nearest-center assignment of each frame, projecting a reconstruction onto
    the common k-means discretization."""
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(centers)).query(np.asarray(CV))[1].astype(np.int64)


def run_benchmark(coords_runs, methods=("kate", "shuffle", "quantize"), *, lag=10,
                  nstates=50, cv_dim=2, dt_strided_ns=0.1, out=None, seed=0,
                  verbose=True):
    """Score each method's kinetic fidelity against the original MSM.

    A contrast plot is written to ``out`` when that argument is provided.

    Returns
    -------
    list of dict
        One result entry per method.
    """
    # Reference featurization on the original trajectory.
    ref = None
    aligned = []
    for r in coords_runs:
        a, ref = kabsch_align(np.asarray(r, float), ref)
        aligned.append(a.reshape(a.shape[0], -1))
    Ttot = sum(a.shape[0] for a in aligned)
    tica = TICA(lag=lag, n_components=cv_dim).fit(aligned)
    CV_ref = [tica.transform(a) for a in aligned]

    # Common discretization; centers are fit once on the original trajectory.
    _, centers = discretize(CV_ref, nstates, seed)
    ref_labels = [_assign(c, centers) for c in CV_ref]
    C_ref = count_matrix(ref_labels, nstates, lag)
    active = largest_connected_set(C_ref)
    P, _ = estimate_reversible_T(C_ref[np.ix_(active, active)])   # deeptime MLE if present
    its_ref = implied_timescales(P, lag, 4)

    results = []
    for m in methods:
        m = m.lower()
        if m == "kate":
            # KATE retains the MSM, so its kinetics are the reference dynamics.
            Q = P
            note = "retained MSM (kinetics not re-estimated)"
            available = True
        else:
            try:
                labels_m = []
                for orig in coords_runs:
                    rec = baselines.reconstruct(m, np.asarray(orig, float), seed=seed)
                    ra, _ = kabsch_align(rec, ref)
                    cv = tica.transform(ra.reshape(ra.shape[0], -1))
                    labels_m.append(_assign(cv, centers))
                C_m = count_matrix(labels_m, nstates, lag)
                Q, _ = estimate_reversible_T(C_m[np.ix_(active, active)])
                note = "re-estimated from reconstruction"
                available = True
            except baselines.BaselineUnavailable as e:
                if verbose:
                    print("  [skip] %-8s : %s" % (m, e))
                results.append({"method": m, "available": False, "reason": str(e)})
                continue

        rep = report_kinetic_fidelity(P, Q, lag=lag, L=Ttot, k=4)
        results.append({
            "method": m, "available": available, "note": note,
            "ensemble_kl": rep["ensemble_kl_nats"],
            "transition_kl": rep["transition_kl_rate_nats_per_step"],
            "pinsker_pair": rep["pinsker_pair_bound"],
            "pinsker_ensemble": rep["pinsker_ensemble_bound"],
            "its_ref_ns": rep["its_ref"] * dt_strided_ns,
            "its_cmp_ns": rep["its_cmp"] * dt_strided_ns,
        })

    if verbose:
        _print_table(results, its_ref * dt_strided_ns)
    if out is not None:
        _plot(results, its_ref * dt_strided_ns, out)
    return results


def _print_table(results, its_ref_ns):
    print("=" * 84)
    print("BENCHMARK -- kinetic fidelity vs the original MSM (path bound; same features"
          "/centers/lag)")
    print("  reference slow timescales : %s ns" % np.round(its_ref_ns[:3], 1))
    print("-" * 84)
    print("  %-9s %12s %14s %14s   %s"
          % ("method", "ensemble", "transition", "Pinsker pair", "t2 (ns)"))
    print("  %-9s %12s %14s %14s   %s"
          % ("", "(nats)", "(nats/step)", "(kinetic)", ""))
    print("-" * 84)
    for r in results:
        if not r.get("available", False):
            print("  %-9s   unavailable (runs on the cluster)" % r["method"])
            continue
        t2 = r["its_cmp_ns"][0] if len(r["its_cmp_ns"]) else float("nan")
        print("  %-9s %12.2e %14.2e %14.2e   %7.1f"
              % (r["method"], r["ensemble_kl"], r["transition_kl"],
                 r["pinsker_pair"], t2))
    print("-" * 84)
    print("Reading: a small ENSEMBLE term with a large TRANSITION term = 'ensemble")
    print("preserved, kinetics not' -- the static bound would wrongly certify it. KATE's")
    print("transition term is ~0 because it retains the MSM. (Real MDZip/SZ3/ZFP numbers")
    print("come from their cluster runs; 'shuffle'/'quantize' are local stand-ins.)")
    print("=" * 84)


def _plot(results, its_ref_ns, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("  (matplotlib unavailable: %s -- skipping plot; pip install kate[kinetics])" % e)
        return
    avail = [r for r in results if r.get("available")]
    if not avail:
        return
    names = [r["method"] for r in avail]
    trans = [r["transition_kl"] for r in avail]
    t2 = [r["its_cmp_ns"][0] if len(r["its_cmp_ns"]) else np.nan for r in avail]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, trans, color=["#2a7" if n == "kate" else "#c44" for n in names])
    ax1.set_yscale("symlog", linthresh=1e-6)
    ax1.set_ylabel("transition term  h(P||Q)  (nats/step)")
    ax1.set_title("Kinetic distortion (lower = better)")
    ax1.axhline(0, color="k", lw=0.5)

    ax2.axhline(its_ref_ns[0], color="k", ls="--", lw=1, label="reference t2")
    ax2.bar(names, t2, color=["#2a7" if n == "kate" else "#c44" for n in names])
    ax2.set_ylabel("slowest implied timescale  t2  (ns)")
    ax2.set_title("Slowest timescale per method")
    ax2.legend()
    fig.tight_layout()
    png = out if str(out).endswith(".png") else f"{out}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("  contrast figure written : %s" % png)
    return png
