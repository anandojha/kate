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
    once the lag exceeds the discretization error; pick the lag where they do."""
    _require()
    out = []
    for lag in lags:
        out.append(mlmsm(dtrajs, int(lag), reversible).timescales(k=k))
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


def msm_for_pathbound(dtrajs, lag, reversible=True):
    """Return (transition_matrix, active_state_indices) for handing to
    epc_pathbound.report_kinetic_fidelity. To compare two compressors fairly,
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
