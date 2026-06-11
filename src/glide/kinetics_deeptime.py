"""
kinetics_deeptime.py
====================
Production MSM kinetics via deeptime (deeptime-ml/deeptime), to replace the
crude symmetrized estimator in kinetic_codec for the numbers that go in the
paper. The hand-rolled estimator in kinetic_codec.transition_matrix uses
(C+C^T)/2; that is fine as a coder/illustration but not what you publish.

What deeptime buys you (the reason the recon said "reuse, don't hand-roll"):
  - reversible MAXIMUM-LIKELIHOOD MSM (proper estimator);
  - implied-timescale convergence (lag scan) to choose the lag honestly;
  - BayesianMSM for sampled uncertainties / error bars on timescales;
  - streaming TICA/VAMP (online covariance via partial_fit) so the CV step
    scales past what fits in RAM -- directly relevant to the 419k -> 1M frame
    trajectory;
  - VAMPnets later, if you want nonlinear slow CVs (not wrapped here).

Run-aware by construction: every function takes a LIST of trajectories (one per
run); deeptime tallies transitions within each, never across seams -- the same
discipline as kinetic_codec.count_matrix.

VERIFIED against deeptime 0.4.5 (signatures introspected and pipeline tested):
    TransitionCountEstimator(lagtime, count_mode).fit_fetch(dtrajs).submodel_largest()
    MaximumLikelihoodMSM(reversible=True).fit_fetch(count_model)        -> msm
    BayesianMSM(reversible=True, n_samples=N).fit_fetch(count_model)    -> posterior.samples
    KMeans(n_clusters, fixed_seed=seed).fit_fetch(X); model.transform(X)
    TICA(lagtime, dim).partial_fit(run); model = est.fetch_model(); model.transform(run)
    msm.timescales(k=...), msm.transition_matrix, msm.stationary_distribution
If you upgrade deeptime, re-check these -- the API has shifted across releases.

Import-guarded: if deeptime is absent, calling any function raises a clear
ImportError with install instructions, so importing the package never breaks.
"""

from __future__ import annotations
import numpy as np

try:
    from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
    from deeptime.markov import TransitionCountEstimator
    from deeptime.clustering import KMeans
    from deeptime.decomposition import TICA
    _HAVE_DEEPTIME = True
    _IMPORT_ERR = None
except Exception as _e:                              # pragma: no cover
    _HAVE_DEEPTIME = False
    _IMPORT_ERR = _e


def _require():
    if not _HAVE_DEEPTIME:
        raise ImportError(
            "kinetics_deeptime requires deeptime. Install with "
            "`pip install deeptime`. Original import error: %r" % (_IMPORT_ERR,))


def tica_cvs(runs_feat, lag, dim):
    """Streaming TICA on a list of per-run feature arrays (e.g. aligned heavy
    coords reshaped to (T,3N), or contact features). Uses partial_fit so the
    covariance is accumulated run-by-run without holding everything in RAM.
    Returns (model, list_of_CV_trajectories)."""
    _require()
    est = TICA(lagtime=int(lag), dim=int(dim))
    for r in runs_feat:
        est.partial_fit(np.asarray(r, dtype=np.float64))
    model = est.fetch_model()
    cvs = [model.transform(np.asarray(r, dtype=np.float64)) for r in runs_feat]
    return model, cvs


def cluster(cv_runs, n_states, seed=0):
    """k-means microstates on the pooled CVs. Returns (dtrajs list, model)."""
    _require()
    X = np.concatenate([np.asarray(c, dtype=np.float64) for c in cv_runs], axis=0)
    model = KMeans(n_clusters=int(n_states), fixed_seed=int(seed),
                   progress=None).fit_fetch(X)
    dtrajs = [model.transform(np.asarray(c, dtype=np.float64)).astype(np.int64)
              for c in cv_runs]
    return dtrajs, model


def mlmsm(dtrajs, lag, reversible=True, count_mode="sliding"):
    """Reversible maximum-likelihood MSM on a list of discrete trajectories,
    restricted to the largest connected set. Returns the deeptime MSM model
    (use .timescales(k), .transition_matrix, .stationary_distribution)."""
    _require()
    counts = TransitionCountEstimator(lagtime=int(lag),
                                      count_mode=count_mode).fit_fetch(dtrajs)
    counts = counts.submodel_largest()
    return MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)


def implied_timescales(dtrajs, lag, k=5, reversible=True):
    """Convenience: the k slowest implied timescales (in frames) at one lag."""
    return mlmsm(dtrajs, lag, reversible).timescales(k=k)


def its_lag_scan(dtrajs, lags, k=5, reversible=True):
    """Implied timescales vs lag -- the convergence check that justifies a lag
    choice. Returns array (len(lags), k) in frames. Timescales should plateau
    once the lag exceeds the discretization error; pick the lag where they do.

    Robust to the reversible-MLE NOT CONVERGING (which happens on real data when the
    discretization is too fine / poorly connected at a given lag): the failing lag's
    row is filled with NaN rather than raising, so the rest of the scan still reports."""
    _require()
    out = []
    for lag in lags:
        try:
            out.append(mlmsm(dtrajs, int(lag), reversible).timescales(k=k))
        except Exception:                       # deeptime MLE non-convergence, etc.
            out.append(np.full(k, np.nan))
    return np.asarray(out)


def bayes_timescales(dtrajs, lag, k=5, n_samples=100, reversible=True):
    """Implied timescales with Bayesian (sampled) uncertainties. Uses the
    'effective' count mode (deeptime's recommendation for statistical error).
    Returns (mean[k], std[k]) in frames -- your kinetic error bars."""
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
    """Honest kinetic-RESOLUTION report -- which dynamical processes this trajectory
    can actually validate. For each implied-timescale process it returns the Bayesian
    timescale, its 95% confidence interval, the relative uncertainty, and the number of
    INDEPENDENT events the trajectory contains for it (~ T_total / t_i, the number of
    round trips). A process is flagged `resolved` only if its Bayesian relative error is
    below `rel_err_max` AND it has at least `min_events` independent events: you cannot
    claim to preserve a kinetic observable the SOURCE trajectory never sampled, no matter
    how good the compressor. `dt_ns` converts (strided) frames to nanoseconds. Returns a
    list of dicts, slowest process first.

    This is the discipline the MD-compression literature usually skips: report the
    statistical resolution of the reference before comparing methods on it."""
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
    """Pretty-print a kinetic_resolution() report as a table (returned as a string)."""
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


def msm_for_pathbound(dtrajs, lag, reversible=True):
    """Return (transition_matrix, active_state_indices) for handing to
    glide_pathbound.report_kinetic_fidelity. To compare two compressors fairly,
    discretize BOTH against the SAME k-means centers, then estimate each MSM
    here, then map both transition matrices onto a common active-state index
    set before calling the path bound (see baselines.py / INSTRUCTIONS)."""
    _require()
    counts = TransitionCountEstimator(lagtime=int(lag),
                                      count_mode="sliding").fit_fetch(dtrajs)
    counts = counts.submodel_largest()
    msm = MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)
    # deeptime keeps the mapping from full state ids to the active submodel:
    active = np.asarray(counts.state_symbols, dtype=np.int64)
    return msm.transition_matrix, active


if __name__ == "__main__":
    # Smoke test: two synthetic 2-state runs -> the pipeline runs end to end.
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
