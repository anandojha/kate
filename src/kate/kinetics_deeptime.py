"""
Markov State Model Kinetics via deeptime
========================================
Background
----------
This module provides the reference Markov state model (MSM) kinetics estimators
used to report timescales and rate observables. The reversible maximum-likelihood
MSM and BayesianMSM are obtained from deeptime (Hoffmann et al., J. Chem. Phys.
2021), in place of the symmetrized (C + C^T)/2 estimator in kinetic_codec. The
symmetrized estimator is adequate as a coder illustration but is not a
detailed-balance maximum-likelihood estimator and is therefore not used for
reported kinetic quantities.

Capabilities
------------
The deeptime backend supplies a reversible maximum-likelihood MSM, an
implied-timescale convergence diagnostic via a lag scan, a BayesianMSM for
sampled uncertainties on timescales, and a streaming TICA estimator whose
covariance is accumulated through partial_fit so the collective-variable step
scales beyond available memory. VAMPnets for nonlinear slow collective variables
are available in deeptime but are not wrapped here.

Run handling
------------
Every estimator accepts a list of trajectories, one per run; deeptime tallies
transitions within each trajectory and never across run boundaries, matching the
convention in kinetic_codec.count_matrix.

Implementation notes
--------------------
The pipeline was verified against deeptime 0.4.5 with the following call
sequence:
    TransitionCountEstimator(lagtime, count_mode).fit_fetch(dtrajs).submodel_largest()
    MaximumLikelihoodMSM(reversible=True).fit_fetch(count_model)        -> msm
    BayesianMSM(reversible=True, n_samples=N).fit_fetch(count_model)    -> posterior.samples
    KMeans(n_clusters, fixed_seed=seed).fit_fetch(X); model.transform(X)
    TICA(lagtime, dim).partial_fit(run); model = est.fetch_model(); model.transform(run)
    msm.timescales(k=...), msm.transition_matrix, msm.stationary_distribution
These signatures have shifted across deeptime releases and should be re-checked
after an upgrade. Imports are guarded: if deeptime is absent, calling any
function raises an ImportError with installation instructions, so importing the
package remains safe.
"""

from __future__ import annotations
import numpy as np

try:
    from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
    from deeptime.markov import TransitionCountEstimator, TransitionCountModel
    from deeptime.clustering import KMeans
    from deeptime.decomposition import TICA
    _HAVE_DEEPTIME = True
    _IMPORT_ERR = None
except Exception as _e:                              # pragma: no cover
    _HAVE_DEEPTIME = False
    _IMPORT_ERR = _e


def reversible_mle_from_counts(C, reversible=True):
    """Estimate a reversible maximum-likelihood transition matrix from a count matrix.

    The estimate is restricted to the largest connected set of the count matrix.
    This is the detailed-balance maximum-likelihood estimator (deeptime), rather
    than the (C + C^T)/2 symmetrization, and backs the reported timescales and
    path bound via kinetic_codec.estimate_reversible_T when deeptime is available.

    Returns
    -------
    tuple
        (T_active, active_state_indices): the transition matrix on the active set
        and the indices of those states in the original state space.
    """
    _require()
    import numpy as _np
    tcm = TransitionCountModel(_np.asarray(C, dtype=_np.float64)).submodel_largest()
    msm = MaximumLikelihoodMSM(reversible=reversible).fit_fetch(tcm)
    return msm.transition_matrix, _np.asarray(tcm.state_symbols, dtype=_np.int64)


def _require():
    if not _HAVE_DEEPTIME:
        raise ImportError(
            "kinetics_deeptime requires deeptime. Install with "
            "`pip install deeptime`. Original import error: %r" % (_IMPORT_ERR,))


def tica_cvs(runs_feat, lag, dim):
    """Fit streaming TICA on a list of per-run feature arrays.

    Features may be aligned heavy-atom coordinates reshaped to (T, 3N) or contact
    features. The covariance is accumulated run-by-run through partial_fit so the
    full dataset need not reside in memory.

    Returns
    -------
    tuple
        (model, list_of_CV_trajectories): the fitted TICA model and the
        transformed collective-variable trajectory for each run.
    """
    _require()
    est = TICA(lagtime=int(lag), dim=int(dim))
    for r in runs_feat:
        est.partial_fit(np.asarray(r, dtype=np.float64))
    model = est.fetch_model()
    cvs = [model.transform(np.asarray(r, dtype=np.float64)) for r in runs_feat]
    return model, cvs


def cluster(cv_runs, n_states, seed=0):
    """Discretize the pooled collective variables into k-means microstates.

    Returns
    -------
    tuple
        (dtrajs, model): the per-run discrete trajectories and the fitted
        k-means model.
    """
    _require()
    X = np.concatenate([np.asarray(c, dtype=np.float64) for c in cv_runs], axis=0)
    model = KMeans(n_clusters=int(n_states), fixed_seed=int(seed),
                   progress=None).fit_fetch(X)
    dtrajs = [model.transform(np.asarray(c, dtype=np.float64)).astype(np.int64)
              for c in cv_runs]
    return dtrajs, model


def mlmsm(dtrajs, lag, reversible=True, count_mode="sliding"):
    """Estimate a reversible maximum-likelihood MSM from discrete trajectories.

    The estimate is restricted to the largest connected set. The returned
    deeptime MSM model exposes .timescales(k), .transition_matrix, and
    .stationary_distribution.
    """
    _require()
    counts = TransitionCountEstimator(lagtime=int(lag),
                                      count_mode=count_mode).fit_fetch(dtrajs)
    counts = counts.submodel_largest()
    return MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)


def implied_timescales(dtrajs, lag, k=5, reversible=True):
    """Return the k slowest implied timescales (in frames) at a single lag."""
    return mlmsm(dtrajs, lag, reversible).timescales(k=k)


def its_lag_scan(dtrajs, lags, k=5, reversible=True):
    """Compute implied timescales across a range of lag times.

    This is the convergence diagnostic that justifies a lag choice: the implied
    timescales plateau once the lag exceeds the discretization error, and the lag
    is selected at the onset of that plateau. Lags at which the reversible
    maximum-likelihood estimate fails to converge (which can occur on real data
    when the discretization is too fine or poorly connected at a given lag) yield
    a row of NaN rather than an exception, so the remaining lags are still
    reported.

    Returns
    -------
    numpy.ndarray
        Array of shape (len(lags), k) of implied timescales in frames.
    """
    _require()
    out = []
    for lag in lags:
        try:
            out.append(mlmsm(dtrajs, int(lag), reversible).timescales(k=k))
        except Exception:                       # deeptime MLE non-convergence, etc.
            out.append(np.full(k, np.nan))
    return np.asarray(out)


def bayes_timescales(dtrajs, lag, k=5, n_samples=100, reversible=True):
    """Compute implied timescales with Bayesian (sampled) uncertainties.

    The 'effective' count mode is used, following the deeptime recommendation for
    statistical-error estimation.

    Returns
    -------
    tuple
        (mean[k], std[k]) in frames, providing the kinetic uncertainty estimates.
    """
    _require()
    counts = TransitionCountEstimator(lagtime=int(lag),
                                      count_mode="effective").fit_fetch(dtrajs)
    counts = counts.submodel_largest()
    posterior = BayesianMSM(reversible=reversible,
                            n_samples=int(n_samples)).fit_fetch(counts)
    samples = np.array([m.timescales(k=k) for m in posterior.samples])
    return samples.mean(axis=0), samples.std(axis=0)


def kinetic_resolution(dtrajs, lag, dt_ns, k=5, n_samples=100,
                       rel_err_max=0.25, min_events=10, reversible=True):
    """Report which dynamical processes the trajectory can statistically resolve.

    For each implied-timescale process the report gives the Bayesian timescale,
    its 95% confidence interval, the relative uncertainty, and the number of
    independent events the trajectory contains for that process (approximately
    T_total / t_i, the number of round trips). A process is flagged ``resolved``
    only when its Bayesian relative error is below ``rel_err_max`` and it has at
    least ``min_events`` independent events; a compressor cannot be claimed to
    preserve a kinetic observable that the source trajectory itself never sampled.
    The argument ``dt_ns`` converts (strided) frames to nanoseconds. Reporting the
    statistical resolution of the reference before comparing methods on it is a
    discipline frequently omitted in the MD-compression literature.

    Returns
    -------
    list of dict
        One entry per process, ordered slowest first.
    """
    _require()
    mean, std = bayes_timescales(dtrajs, lag, k=k, n_samples=int(n_samples),
                                 reversible=reversible)
    T_ns = sum(len(np.asarray(d)) for d in dtrajs) * float(dt_ns)
    report = []
    for i, (m, s) in enumerate(zip(mean, std)):
        t = float(m) * float(dt_ns)
        sd = float(s) * float(dt_ns)
        rel = sd / t if t > 0 else float("inf")
        nev = T_ns / t if t > 0 else 0.0
        report.append({
            "process": i + 1,
            "timescale_ns": t,
            "ci_lo_ns": max(0.0, t - 1.96 * sd),    # timescales are non-negative
            "ci_hi_ns": t + 1.96 * sd,
            "rel_err": rel,
            "n_events": nev,
            "resolved": bool(rel < rel_err_max and nev >= min_events),
        })
    return report


def format_resolution(report, total_us=None):
    """Format a kinetic_resolution() report as a table, returned as a string."""
    lines = []
    if total_us is not None:
        lines.append("trajectory length: %.1f us" % total_us)
    lines.append(" proc  timescale(ns)   Bayes 95% CI (ns)       rel.err   indep.events   resolved?")
    for r in report:
        lines.append("  t%-2d   %9.0f    [%8.0f,%8.0f]    %5.0f%%      %7.1f       %s"
                     % (r["process"], r["timescale_ns"], r["ci_lo_ns"], r["ci_hi_ns"],
                        100 * r["rel_err"], r["n_events"], "YES" if r["resolved"] else "no"))
    nres = sum(r["resolved"] for r in report)
    fastest_unres = next((r["timescale_ns"] for r in report if not r["resolved"]), None)
    if nres == 0:
        lines.append("=> NONE of the listed processes are statistically resolved here.")
    else:
        res_ts = [r["timescale_ns"] for r in report if r["resolved"]]
        lines.append("=> resolved: processes <= %.0f ns (%d of %d). Slower processes are "
                     "sampling-limited;" % (max(res_ts), nres, len(report)))
        lines.append("   kinetic claims must target the resolved band, not the slow tail.")
    return "\n".join(lines)


def metastable_mfpt(dtrajs, lag, dt_ns, n_meta=2, count_mode="sliding", reversible=True):
    """Compute PCCA+ metastable sets and inter-set mean first-passage times.

    The mean first-passage times yield the rate observables (k_on, k_off
    approximately 1/MFPT) addressed by the path bound, computed directly from the
    reversible maximum-likelihood MSM rather than only bounded. This expresses the
    transition-term guarantee in the rate language used in the field. The deeptime
    MFPT is in trajectory-step units and is converted to nanoseconds via ``dt_ns``.

    Returns
    -------
    dict
        Metastable populations, the inter-set MFPT matrix in nanoseconds, and the
        leading implied timescales in nanoseconds.
    """
    _require()
    counts = TransitionCountEstimator(lagtime=int(lag),
                                      count_mode=count_mode).fit_fetch(dtrajs)
    counts = counts.submodel_largest()
    msm = MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)
    n_meta = int(max(2, min(n_meta, msm.n_states)))
    pcca = msm.pcca(n_meta)
    assign = np.asarray(pcca.assignments)
    sets = [np.where(assign == m)[0] for m in range(n_meta)]
    pi = msm.stationary_distribution
    meta_pop = np.array([float(pi[s].sum()) for s in sets])
    mfpt = np.full((n_meta, n_meta), np.nan)
    for i in range(n_meta):
        for j in range(n_meta):
            if i != j and len(sets[i]) and len(sets[j]):
                mfpt[i, j] = float(msm.mfpt(sets[i], sets[j])) * float(dt_ns)   # frames->ns
    k = int(min(n_meta - 1, msm.n_states - 1))
    its = (msm.timescales(k=k) * float(dt_ns)) if k >= 1 else np.array([])
    return {"n_meta": n_meta, "meta_pop": meta_pop, "mfpt_ns": mfpt, "timescales_ns": its}


def format_mfpt(report):
    """Format a metastable_mfpt() report as a table, returned as a string."""
    n = report["n_meta"]
    lines = ["PCCA+ metastable kinetics  (%d metastable sets, reversible-MLE MSM)" % n,
             "  metastable populations : %s"
             % np.array2string(report["meta_pop"], precision=3, suppress_small=True)]
    if report["timescales_ns"].size:
        lines.append("  leading timescales (ns): %s"
                     % np.array2string(report["timescales_ns"], precision=1))
    lines.append("  mean first-passage time MFPT(i->j) in ns  (rate k_ij ~ 1/MFPT):")
    lines.append("         " + "".join("   ->S%-8d" % j for j in range(n)))
    for i in range(n):
        row = "    S%-3d" % i
        for j in range(n):
            v = report["mfpt_ns"][i, j]
            row += "  %10s" % ("--" if (i == j or not np.isfinite(v)) else "%.1f" % v)
        lines.append(row)
    return "\n".join(lines)


def msm_for_pathbound(dtrajs, lag, reversible=True):
    """Return (transition_matrix, active_state_indices) for the path bound.

    The result is intended for kate_pathbound.report_kinetic_fidelity. A fair
    comparison of two compressors requires discretizing both against the same
    k-means centers, estimating each MSM here, and mapping both transition
    matrices onto a common active-state index set before calling the path bound
    (see baselines.py and the project instructions).
    """
    _require()
    counts = TransitionCountEstimator(lagtime=int(lag),
                                      count_mode="sliding").fit_fetch(dtrajs)
    counts = counts.submodel_largest()
    msm = MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)
    # deeptime retains the mapping from full state ids to the active submodel.
    active = np.asarray(counts.state_symbols, dtype=np.int64)
    return msm.transition_matrix, active


if __name__ == "__main__":
    # Smoke test: two synthetic two-state runs exercise the pipeline end to end.
    if not _HAVE_DEEPTIME:
        print("deeptime not installed; `pip install deeptime` to use this module.")
        raise SystemExit(0)
    np.set_printoptions(precision=3, suppress=True)

    def _run(n, seed, a=0.01):
        r = np.random.default_rng(seed)
        P = np.array([[1 - a, a], [a, 1 - a]]); cdf = np.cumsum(P, 1)
        s = np.zeros(n, int); u = r.random(n)
        for t in range(1, n):
            s[t] = np.searchsorted(cdf[s[t - 1]], u[t])
        return np.stack([(s * 2.0 - 1.0) + 0.3 * r.standard_normal(n),
                         0.3 * r.standard_normal(n)], 1)

    runs = [_run(8000, 1), _run(8000, 2)]
    _, cvs = tica_cvs(runs, lag=10, dim=1)
    dtrajs, _ = cluster(cvs, n_states=20, seed=0)
    print("implied timescales (frames):", np.round(implied_timescales(dtrajs, 10, k=3), 1))
    m, sd = bayes_timescales(dtrajs, 10, k=2, n_samples=50)
    print("Bayesian t2: %.1f +/- %.1f frames" % (m[0], sd[0]))
    scan = its_lag_scan(dtrajs, lags=[5, 10, 20, 40], k=2)
    print("lag scan (t2 vs lag 5/10/20/40):", np.round(scan[:, 0], 1))
