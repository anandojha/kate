"""
benchmark.py
============
The T3 contrast -- the paper's central figure. ONE pipeline:

  load trajectory
    -> {EPC, baselines}  reconstruct  (EPC RETAINS its MSM; baselines reconstruct coords)
    -> featurize with a COMMON TICA and discretize against COMMON k-means centers
    -> reversible MSM per method on the SAME active-state support  (matched indexing)
    -> epc_pathbound score vs the ORIGINAL MSM  (ensemble term + transition term)
    -> table + plot (implied timescales / transition-term per method).

Why matched support, not deeptime's per-method largest-connected-set: the path bound
compares P and Q ENTRYWISE, so every method's transition matrix must live on the SAME
state indexing. We therefore fix ONE active set (largest connected set of the
reference counts) and estimate every method's reversible MSM on exactly those states.
This is the fair-comparison discipline RECIPE T3 asks for (same features, same centers,
same lag) carried through to the estimator. (The headline single-method kinetics use
deeptime's reversible MLE via `epc analyze`; here the CONTRAST is the deliverable.)

Expected result (the claim): ensemble-only / coordinate-bounded methods show a large
TRANSITION term (their implied timescales drift); EPC's is ~0 because it retains the
MSM. The ensemble term is small for all -- which is exactly why the static bound would
WRONGLY certify the others as faithful.
"""
from __future__ import annotations

import numpy as np

from .kinetic_codec import (kabsch_align, TICA, discretize, count_matrix,
                            transition_matrix, largest_connected_set,
                            implied_timescales)
from .pathbound import report_kinetic_fidelity
from . import baselines


def _assign(CV, centers):
    """Nearest-center microstate assignment (project a reconstruction onto the COMMON
    discretization)."""
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(centers)).query(np.asarray(CV))[1].astype(np.int64)


def run_benchmark(coords_runs, methods=("epc", "shuffle", "quantize"), *, lag=10,
                  nstates=50, cv_dim=2, dt_strided_ns=0.1, out=None, seed=0,
                  verbose=True):
    """Score each method's kinetic fidelity against the original MSM. Returns a list
    of per-method result dicts; writes a contrast plot to ``out`` if given."""
    # --- reference featurization on the ORIGINAL trajectory ---
    ref = None
    aligned = []
    for r in coords_runs:
        a, ref = kabsch_align(np.asarray(r, float), ref)
        aligned.append(a.reshape(a.shape[0], -1))
    Ttot = sum(a.shape[0] for a in aligned)
    tica = TICA(lag=lag, n_components=cv_dim).fit(aligned)
    CV_ref = [tica.transform(a) for a in aligned]

    # --- COMMON discretization (centers fit once, on the original) ---
    _, centers = discretize(CV_ref, nstates, seed)
    ref_labels = [_assign(c, centers) for c in CV_ref]
    C_ref = count_matrix(ref_labels, nstates, lag)
    active = largest_connected_set(C_ref)
    P, _ = transition_matrix(C_ref[np.ix_(active, active)], reversible=True)
    its_ref = implied_timescales(P, lag, 4)

    results = []
    for m in methods:
        m = m.lower()
        if m == "epc":
            # EPC RETAINS the MSM -> its kinetics ARE the reference dynamics.
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
                Q, _ = transition_matrix(C_m[np.ix_(active, active)], reversible=True)
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
    print("preserved, kinetics not' -- the static bound would wrongly certify it. EPC's")
    print("transition term is ~0 because it retains the MSM. (Real MDZip/SZ3/ZFP numbers")
    print("come from their cluster runs; 'shuffle'/'quantize' are local stand-ins.)")
    print("=" * 84)


def _plot(results, its_ref_ns, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("  (matplotlib unavailable: %s -- skipping plot; pip install epc[kinetics])" % e)
        return
    avail = [r for r in results if r.get("available")]
    if not avail:
        return
    names = [r["method"] for r in avail]
    trans = [r["transition_kl"] for r in avail]
    t2 = [r["its_cmp_ns"][0] if len(r["its_cmp_ns"]) else np.nan for r in avail]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, trans, color=["#2a7" if n == "epc" else "#c44" for n in names])
    ax1.set_yscale("symlog", linthresh=1e-6)
    ax1.set_ylabel("transition term  h(P||Q)  (nats/step)")
    ax1.set_title("Kinetic distortion (lower = better)")
    ax1.axhline(0, color="k", lw=0.5)

    ax2.axhline(its_ref_ns[0], color="k", ls="--", lw=1, label="reference t2")
    ax2.bar(names, t2, color=["#2a7" if n == "epc" else "#c44" for n in names])
    ax2.set_ylabel("slowest implied timescale  t2  (ns)")
    ax2.set_title("Slowest timescale per method")
    ax2.legend()
    fig.tight_layout()
    png = out if str(out).endswith(".png") else f"{out}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("  contrast figure written : %s" % png)
    return png
