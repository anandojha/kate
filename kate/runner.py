#!/usr/bin/env python
"""
Flow-Based KATE Trajectory Compression Runner
==============================================
This module provides the execution backend for ``kate compress``, applying
flow-based KATE to molecular-dynamics trajectories supplied either as in-memory
arrays or as DCD files. Dimensionality reduction to a small set of TICA collective
variables precedes flow training, which keeps the procedure tractable for systems
of size 3N of order 5000.

Entry points
------------
Three entry points share a common post-collective-variable core, ``_assemble_artifact``:

  * ``compress_trajectory(coords_runs)`` operates in memory and is run-aware. It is
    the default path.
  * ``compress_streaming(chunk_factory)`` performs out-of-core compression using
    streaming TICA (chunked ``partial_fit``), so the collective-variable step scales
    beyond available memory. The full trajectory is never retained; only the
    low-dimensional collective variables and the sparse set of kept frames are held.
    The procedure is multi-pass: a streaming TICA fit, a transform pass producing
    collective variables followed by flow training, frame selection, and MSM
    estimation, and a final pass that reads only the kept frames for the residual.
    Streaming TICA reproduces batch TICA exactly, since cross-chunk lagged pairs are
    retained.
  * ``run_kate(top, dcd[, streaming=True])`` provides the DCD loader for both paths.

Kinetic bound
-------------
The path-distribution bound is incorporated directly. For KATE the retained MSM Q
equals the full-data reference P, so the transition term vanishes by construction.
The full-atom residual stage recovers the fast modes discarded by TICA.
"""
from __future__ import annotations

import argparse
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch

from .flow import RealNVP
from .codec import igfs_select, encode_iid, decode_iid, gaussian_cumfreq
from .kinetic_codec import (kabsch_align, TICA, discretize, count_matrix,
                            transition_matrix, estimate_reversible_T, implied_timescales,
                            largest_connected_set, DitheredResidualCodec)
from .inspect_traj import heavy_indices
from .pathbound import report_kinetic_fidelity
from .artifact import Artifact


def free_energy_1d(v, bins):
    h, _ = np.histogram(v, bins=bins, density=True)
    return -np.log(np.clip(h, 1e-8, None))


def kl_1d(p, q, bins):
    a, _ = np.histogram(p, bins=bins); b, _ = np.histogram(q, bins=bins)
    a = np.clip(a / a.sum(), 1e-8, None); b = np.clip(b / b.sum(), 1e-8, None)
    a /= a.sum(); b /= b.sum()
    return float((a * np.log(a / b)).sum())


def _assemble_artifact(CV_runs, fetch_kept, cv_meta, ref, *, cv_dim, keep_frac, epochs,
                       nstates, lag, stride, dt_ps, lat_bits, n_bits, seed, verbose,
                       entropy="gaussian", flow_kind="realnvp", predictive_kind="gru"):
    """Assemble a compression artifact from precomputed collective variables.

    This shared post-collective-variable core executes the full pipeline: flow density
    estimation, information-gain frame selection, entropy coding, retained-MSM
    estimation, path-bound evaluation, full-atom residual coding, and artifact
    construction. ``fetch_kept(global_indices)`` returns the aligned flattened
    coordinates of shape (n_keep, 3N) for the kept frames; this is an in-memory slice
    on the batch path and a streamed read on the out-of-core path. ``cv_meta`` carries
    the collective-variable metadata (TICA parameters, VAMP score, and method tag). The
    reconstruction is independent of the collective-variable method, relying on a fitted
    linear decoder, and therefore applies equally to TICA and VAMPnet variables."""
    dt_strided_ns = stride * dt_ps / 1000.0
    CV_all = np.concatenate(CV_runs, axis=0).astype(np.float32)
    run_lengths = [c.shape[0] for c in CV_runs]
    T_total = CV_all.shape[0]

    # Flow density estimation on the collective variables. RealNVP is the default;
    # the spline-coupling flow is selected by flow_kind == "spline".
    if flow_kind == "spline":
        from .spline_flow import SplineFlow
        flow = SplineFlow(cv_dim, hidden=64, n_layers=10, n_bins=8)
        flow_arch = {"dim": cv_dim, "hidden": 64, "n_layers": 10, "n_bins": 8}
    else:
        flow = RealNVP(cv_dim, hidden=64, n_layers=10)
        flow_arch = {"dim": cv_dim, "hidden": 64, "n_layers": 10}
    if verbose:
        print("  training %s flow on %d-D CVs ..." % (flow_kind, cv_dim))
    flow = flow.fit(CV_all, epochs=epochs, batch=1024, verbose=verbose, seed=seed)
    with torch.no_grad():
        z_all = flow.forward(torch.as_tensor(CV_all))[0].numpy()

    # Information-gain frame selection followed by lossless coding of the kept
    # latents against the base density N(0, I).
    n_keep = max(2, int(keep_frac * T_total))
    kept = igfs_select(z_all, n_keep, seed=seed)
    L = 1 << lat_bits
    zmax = max(6.0, float(np.abs(z_all[kept]).max()) * 1.02)
    cum = gaussian_cumfreq(L, zmax)
    lev = np.clip(np.floor((np.clip(z_all[kept], -zmax, zmax) + zmax) /
                           (2 * zmax) * L).astype(np.int64), 0, L - 1).ravel()
    coded = encode_iid(lev, cum)
    dlev = decode_iid(coded, len(kept) * cv_dim, cum).reshape(len(kept), cv_dim)
    zrec = -zmax + (dlev + 0.5) * (2 * zmax / L)
    with torch.no_grad():
        cv_rec = flow.inverse(torch.as_tensor(zrec, dtype=torch.float32)).numpy()
    cv_err = float(np.abs(cv_rec - CV_all[kept]).max())

    # Ensemble fidelity of the flow against the data, with the Pinsker inequality
    # applied to the leading collective variable.
    with torch.no_grad():
        samp = flow.sample(40000).numpy()
    lo, hi = np.percentile(CV_all[:, 0], [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    KL = kl_1d(CV_all[:, 0], samp[:, 0], bins)
    mid = 0.5 * (lo + hi)
    obs = abs(float((CV_all[:, 0] > mid).mean()) - float((samp[:, 0] > mid).mean()))

    # Dynamics term: a run-aware MSM estimated on the collective variables.
    # Reported kinetics, comprising the implied timescales and the path bound, use the
    # reversible maximum-likelihood estimator from deeptime when it is available, and
    # otherwise fall back to the symmetrized (C + C^T)/2 estimator; the field
    # `msm_estimator` records which was used. The full-size T_msm retained for
    # convenience remains the pure-numpy estimate.
    labels, centers = discretize(CV_runs, nstates, seed)
    C = count_matrix(labels, nstates, lag)
    T_msm, _ = transition_matrix(C, reversible=True)
    act = largest_connected_set(C)
    Tm_act, msm_estimator = estimate_reversible_T(C[np.ix_(act, act)])
    its = implied_timescales(Tm_act, lag, 5)
    its_ns = its * dt_strided_ns
    kin = report_kinetic_fidelity(Tm_act, Tm_act, lag=lag, L=T_total, k=4)  # KATE: Q == P

    # Full-atom residual stage: a method-agnostic linear decoder combined with a
    # dithered residual. A linear collective-variable-to-coordinate decoder, fit on the
    # kept frames, maps the decoded collective variables back to 3N dimensions; the
    # residual recovers the component the linear map omits. The construction applies to
    # any collective variables (TICA or VAMPnet) and requires only the kept frames,
    # which makes it compatible with streaming.
    labels_all = np.concatenate(labels)
    X_kept = np.asarray(fetch_kept(kept), dtype=np.float64)          # (n_keep, 3N) aligned
    xmean = X_kept.mean(axis=0)
    cmean = cv_rec.mean(axis=0)
    Bdec, *_ = np.linalg.lstsq(cv_rec - cmean, X_kept - xmean, rcond=None)   # (cv_dim, 3N)
    X_approx_kept = (cv_rec - cmean) @ Bdec + xmean
    R_kept = X_kept - X_approx_kept
    labels_kept = labels_all[kept]
    D = X_kept.shape[1]
    state_mean_R = np.zeros((nstates, D))
    for s in np.unique(labels_kept):
        state_mean_R[s] = R_kept[labels_kept == s].mean(axis=0)
    resid = R_kept - state_mean_R[labels_kept]
    rstep = float((resid.std() + 1e-12) * (2.0 ** (1 - n_bits)) * 3.0)
    rcodec = DitheredResidualCodec(n_bits=n_bits, seed=seed)
    rq = rcodec.quantize(resid, rstep)
    residual = {"state_mean_R": state_mean_R, "q": rq, "step": rstep,
                "n_bits": n_bits, "seed": seed, "labels_kept": labels_kept,
                "n_atoms": D // 3, "B": Bdec, "cmean": cmean, "xmean": xmean}
    full_rec = X_approx_kept + rcodec.dequantize(rq, rstep) + state_mean_R[labels_kept]
    fullatom_rmsd = float(np.sqrt(((full_rec - X_kept) ** 2)
                                  .reshape(len(kept), -1, 3).sum(2).mean()))
    # Steric validity. This force-field-free geometry check tests whether the
    # reconstruction introduces atomic overlaps absent from the original. The metric is
    # the smallest inter-atomic distance per frame, whose natural floor is the bonded
    # distance. A reconstruction floor well below the original floor indicates that
    # decoding created clashes that the path bound does not detect.
    nat = D // 3
    orig_min = _min_interatomic(X_kept.reshape(len(kept), nat, 3), seed)
    rec_min = _min_interatomic(full_rec.reshape(len(kept), nat, 3), seed)
    steric = {"orig_min_nm": float(np.percentile(orig_min, 1)),
              "rec_min_nm": float(np.percentile(rec_min, 1))}
    steric["ok"] = bool(steric["rec_min_nm"] >= 0.9 * steric["orig_min_nm"])

    # Temporal learned-entropy model. The coding rate is evaluated with and without the
    # temporal prior over all frames.
    temporal_arch = temporal_state = None
    predictive_arch = predictive_state = None
    rate_gaussian_bpv = rate_temporal_bpv = None
    rd_curve = pred_cond_nll = pred_static_nll = None
    if entropy == "temporal":
        from .temporal_prior import (TemporalPrior, gaussian_rate_bits_per_value,
                                     temporal_rate_bits_per_value)
        if verbose:
            print("  training temporal prior on the %d-frame latent sequence ..." % T_total)
        temporal_arch = {"dim": cv_dim, "hidden": 64, "n_layers": 3}
        tmodel = TemporalPrior(**temporal_arch).fit(
            z_all, epochs=max(100, epochs // 2), verbose=False, seed=seed)
        rate_gaussian_bpv = gaussian_rate_bits_per_value(z_all, L, zmax)
        rate_temporal_bpv = temporal_rate_bits_per_value(z_all, tmodel, L, zmax)
        temporal_state = {k: v.detach().cpu() for k, v in tmodel.state_dict().items()}
    elif entropy == "predictive":
        # Lossy learned predictive coding. The causal predictor is trained with the
        # bound-as-loss conditional negative log-likelihood, and the rate-distortion
        # curve is reported together with the no-predictor floor. The rate-versus-
        # observable-error comparison against the temporal model is evaluated on the
        # trypsin set.
        from .predictive_coder import (make_predictor, rate_distortion_curve,
                                       conditional_nll, static_gaussian_nll)
        from .temporal_prior import gaussian_rate_bits_per_value
        if verbose:
            print("  training %s predictor on the %d-frame latent sequence ..."
                  % (predictive_kind, T_total))
        pmodel = make_predictor(cv_dim, kind=predictive_kind, hidden=64)
        pmodel.fit(z_all, epochs=max(100, epochs // 2), verbose=False, seed=seed)
        predictive_arch = {"dim": cv_dim, "kind": predictive_kind, "hidden": 64}
        predictive_state = {k: v.detach().cpu() for k, v in pmodel.state_dict().items()}
        rd_curve = rate_distortion_curve(z_all, pmodel, [4, 6, 8, 10, 12], seed=seed)
        pred_cond_nll = conditional_nll(pmodel, z_all)
        pred_static_nll = static_gaussian_nll(z_all)
        rate_gaussian_bpv = gaussian_rate_bits_per_value(z_all, L, zmax)

    artifact = Artifact(
        cv_dim=cv_dim, L=L, zmax=zmax, n_keep=len(kept),
        coded_latents=coded, kept_idx=kept,
        run_lengths=run_lengths, dtraj=labels, centers=centers, counts=C,
        T_msm=T_msm, msm_estimator=msm_estimator, n_states=nstates, lag=lag,
        stride=stride, dt_ps=dt_ps, dt_strided_ns=dt_strided_ns,
        flow_arch=flow_arch,
        cv=cv_meta["cv"], flow_kind=flow_kind, entropy=entropy,
        tica_mean=cv_meta.get("tica_mean"), tica_eigvecs=cv_meta.get("tica_eigvecs"),
        tica_timescales=cv_meta.get("tica_timescales"),
        align_ref=ref, x_mean=cv_meta.get("x_mean"), residual=residual,
        temporal_arch=temporal_arch, temporal_state=temporal_state,
        predictive_arch=predictive_arch, predictive_state=predictive_state,
        flow_state={k: v.detach().cpu() for k, v in flow.state_dict().items()},
    )
    flow_bytes = sum(p.numel() for p in flow.parameters()) * 4
    report = {
        "frames": T_total, "run_lengths": run_lengths,
        "kl_cv1_nats": KL, "pinsker_cv1": float(np.sqrt(KL / 2)),
        "bounded_obs_diff": obs, "pinsker_ok": bool(obs <= np.sqrt(KL / 2) + 1e-9),
        "n_keep": len(kept), "keep_frac": len(kept) / T_total, "cv_recon_err": cv_err,
        "fullatom_rmsd": fullatom_rmsd, "steric": steric, "n_bits": n_bits,
        "residual_bits": int(rq.size) * n_bits, "state_mean_bits": int(state_mean_R.size) * 32,
        "flow_bytes": flow_bytes, "coded_bytes": len(coded),
        "implied_timescales_ns": its_ns, "lag": lag, "dt_strided_ns": dt_strided_ns,
        "msm_estimator": msm_estimator,
        "kinetic_bound": kin, "entropy": entropy, "cv": cv_meta["cv"],
        "features": cv_meta.get("features", "cartesian"),
        "vamp_score": cv_meta.get("vamp_score"),
        "rate_gaussian_bpv": rate_gaussian_bpv, "rate_temporal_bpv": rate_temporal_bpv,
        "rd_curve": rd_curve, "pred_cond_nll": pred_cond_nll,
        "pred_static_nll": pred_static_nll, "predictive_kind": predictive_kind,
    }
    return artifact, report


def _tica_cvs(aligned, lag, cv_dim, x_mean, verbose):
    tica = TICA(lag=lag, n_components=cv_dim).fit(aligned)
    CV_runs = [tica.transform(a) for a in aligned]
    if verbose:
        print("  TICA CVs              : %d-D   leading timescales (frames): %s"
              % (cv_dim, np.round(tica.timescales_[:cv_dim], 1)))
    cv_meta = {"cv": "tica", "features": "cartesian",
               "tica_mean": tica.mean_, "tica_eigvecs": tica.eigvecs_,
               "tica_timescales": tica.timescales_, "x_mean": tica.mean_,
               "vamp_score": None}
    return CV_runs, cv_meta


def _min_interatomic(X3, seed, n_sample=200):
    """Compute the per-frame minimum inter-atomic distance over a sampled set of frames.

    The distance is returned in nanometres and provides the floor used by the
    steric-validity check. Because the per-frame computation is O(N^2), a seeded
    random subset of frames is sampled rather than the full set."""
    X3 = np.asarray(X3, dtype=np.float64)
    n = len(X3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(int(n_sample), n), replace=False)
    out = np.empty(len(idx))
    for k, f in enumerate(idx):
        P = X3[f]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        out[k] = np.sqrt(d2.min())
    return out


def _contact_cvs(coords_runs, lag, cv_dim, verbose, max_atoms=64, sep=2):
    """Featurize trajectories by pairwise inter-atomic distances and apply TICA.

    Inter-atomic distances are invariant to global translation and rotation, which
    removes the spurious rigid-body slow modes that TICA on aligned Cartesian
    coordinates can introduce. This internal-coordinate featurization is the physically
    preferred choice for kinetics (Perez-Hernandez, J. Chem. Phys. 139, 015102, 2013;
    Scherer, J. Chem. Theory Comput. 11, 5525, 2015). The featurization is
    topology-free: it uses a capped, evenly spaced atom subset together with a
    sequence-separation filter, so it operates without residue information. Residue or
    contact-map featurization derived from the topology constitutes the ideal
    refinement. Reconstruction is unaffected, since the full-atom residual decoder is
    fit on coordinates rather than on the collective-variable featurization."""
    N = int(np.asarray(coords_runs[0]).shape[1])
    sub = np.unique(np.linspace(0, N - 1, min(int(max_atoms), N)).astype(int))
    pr = [(int(i), int(j)) for a, i in enumerate(sub) for j in sub[a + 1:]
          if abs(int(i) - int(j)) >= int(sep)]
    pi = np.array([p[0] for p in pr]); pj = np.array([p[1] for p in pr])
    feats = []
    for r in coords_runs:
        X = np.asarray(r, dtype=np.float64)
        feats.append(np.linalg.norm(X[:, pi, :] - X[:, pj, :], axis=2))   # (T, n_pairs)
    F = np.concatenate(feats, 0)
    mu = F.mean(0); sd = F.std(0) + 1e-9
    feats = [(f - mu) / sd for f in feats]                                 # standardize
    tica = TICA(lag=lag, n_components=cv_dim).fit(feats)
    CV_runs = [tica.transform(f) for f in feats]
    if verbose:
        print("  contact-TICA CVs      : %d-D from %d pairwise distances  timescales(frames): %s"
              % (cv_dim, F.shape[1], np.round(tica.timescales_[:cv_dim], 1)))
    cv_meta = {"cv": "tica", "features": "contacts",
               "tica_mean": None, "tica_eigvecs": None,
               "tica_timescales": tica.timescales_, "x_mean": None, "vamp_score": None}
    return CV_runs, cv_meta


def _vampnet_cvs(aligned, lag, cv_dim, x_mean, epochs, seed, verbose):
    from .vampnet_cv import vampnet_cvs
    _, CV_runs, score = vampnet_cvs(aligned, lag, cv_dim, n_epochs=max(20, epochs // 4),
                                    seed=seed, verbose=verbose)
    cv_meta = {"cv": "vampnet", "tica_mean": None, "tica_eigvecs": None,
               "tica_timescales": None, "x_mean": x_mean, "vamp_score": score}
    return CV_runs, cv_meta


def compress_trajectory(coords_runs: List[np.ndarray], *, cv="tica", features="cartesian",
                        cv_dim=6, keep_frac=0.10, epochs=300, nstates=200, lag=10, stride=1,
                        dt_ps=100.0, lat_bits=14, n_bits=4, seed=0, verbose=True,
                        entropy="gaussian", flow_kind="realnvp",
                        predictive_kind="gru") -> Tuple[Artifact, dict]:
    """Run in-memory, run-aware flow-based KATE on a list of coordinate arrays.

    Each input array has shape (T_i, N, 3) in nanometres. ``cv`` selects the
    collective-variable method: 'tica' (linear, the default) or 'vampnet'. ``features``
    selects the TICA featurization: 'cartesian' (aligned coordinates, the default) or
    'contacts' (rotation- and translation-invariant inter-atomic distances, which remove
    spurious rigid-body slow modes and are physically preferred for kinetics). The
    ``features`` argument is ignored when cv='vampnet'."""
    ref = None
    aligned = []
    for r in coords_runs:
        a, ref = kabsch_align(np.asarray(r, dtype=np.float64), ref)
        aligned.append(a.reshape(a.shape[0], -1))
    X_all = np.concatenate(aligned, axis=0)
    x_mean = X_all.mean(axis=0)
    if cv == "vampnet":
        CV_runs, cv_meta = _vampnet_cvs(aligned, lag, cv_dim, x_mean, epochs, seed, verbose)
    elif features == "contacts":
        CV_runs, cv_meta = _contact_cvs(coords_runs, lag, cv_dim, verbose)
    else:
        CV_runs, cv_meta = _tica_cvs(aligned, lag, cv_dim, x_mean, verbose)
    return _assemble_artifact(CV_runs, lambda idx: X_all[idx], cv_meta, ref,
                              cv_dim=cv_dim, keep_frac=keep_frac, epochs=epochs,
                              nstates=nstates, lag=lag, stride=stride, dt_ps=dt_ps,
                              lat_bits=lat_bits, n_bits=n_bits, seed=seed,
                              verbose=verbose, entropy=entropy, flow_kind=flow_kind,
                              predictive_kind=predictive_kind)


def compress_streaming(chunk_factory: Callable[[], Iterable[np.ndarray]], *, cv_dim=6,
                       keep_frac=0.10, epochs=300, nstates=200, lag=10, stride=1,
                       dt_ps=100.0, lat_bits=14, n_bits=4, seed=0, verbose=True,
                       entropy="gaussian", flow_kind="realnvp",
                       predictive_kind="gru") -> Tuple[Artifact, dict]:
    """Run out-of-core flow-based KATE over a stream of coordinate chunks.

    ``chunk_factory()`` returns a fresh iterator of coordinate chunks of shape
    (T_c, N, 3), for example from ``md.iterload``. The trajectory is treated as a single
    continuous run and is never held in memory in full; only the low-dimensional
    collective variables and the sparse kept frames are retained. Streaming TICA
    accumulates the same covariances as the batch path, so the TICA model, the seeded
    discretization, the retained MSM, and the kinetics are identical to those produced
    by ``compress_trajectory``. The flow and frame-selection exemplars may vary between
    runs because CPU multi-threading renders flow training non-deterministic, but the
    kinetics, the ensemble, and the bound are unaffected."""
    # The reference is the centred first frame of the first chunk, matching the batch
    # path.
    first_chunk = next(iter(chunk_factory()))
    ref = np.asarray(first_chunk[0], dtype=np.float64)
    ref = ref - ref.mean(axis=0, keepdims=True)

    # Pass 1: streaming TICA via chunked partial_fit.
    if verbose:
        print("  [pass 1] streaming TICA partial_fit ...")
    tica = TICA(lag=lag, n_components=cv_dim)
    for i, chunk in enumerate(chunk_factory()):
        a, _ = kabsch_align(np.asarray(chunk, dtype=np.float64), ref)
        tica.partial_fit(a.reshape(a.shape[0], -1), run_start=(i == 0))
    tica.finalize()
    if verbose:
        print("  TICA CVs              : %d-D   leading timescales (frames): %s"
              % (cv_dim, np.round(tica.timescales_[:cv_dim], 1)))

    # Pass 2: transform to collective variables, which are small and collected in
    # memory.
    if verbose:
        print("  [pass 2] transform -> CVs ...")
    CV_chunks = []
    for chunk in chunk_factory():
        a, _ = kabsch_align(np.asarray(chunk, dtype=np.float64), ref)
        CV_chunks.append(tica.transform(a.reshape(a.shape[0], -1)))
    CV_all = np.concatenate(CV_chunks, axis=0)

    # Pass 3, evaluated lazily: fetch only the aligned coordinates of the kept frames
    # for the residual stage.
    def fetch_kept(kept_idx):
        kept_idx = np.asarray(kept_idx, dtype=np.int64)
        kmap = {int(g): i for i, g in enumerate(kept_idx)}
        kset = set(kmap)
        out = np.zeros((len(kept_idx), ref.shape[0] * 3))
        pos = 0
        for chunk in chunk_factory():
            a, _ = kabsch_align(np.asarray(chunk, dtype=np.float64), ref)
            af = a.reshape(a.shape[0], -1)
            for j in range(af.shape[0]):
                g = pos + j
                if g in kset:
                    out[kmap[g]] = af[j]
            pos += af.shape[0]
        return out

    if verbose:
        print("  [pass 3] residual for kept frames (streamed) ...")
    cv_meta = {"cv": "tica", "tica_mean": tica.mean_, "tica_eigvecs": tica.eigvecs_,
               "tica_timescales": tica.timescales_, "x_mean": tica.mean_,
               "vamp_score": None}
    return _assemble_artifact([CV_all], fetch_kept, cv_meta, ref,
                              cv_dim=cv_dim, keep_frac=keep_frac, epochs=epochs,
                              nstates=nstates, lag=lag, stride=stride, dt_ps=dt_ps,
                              lat_bits=lat_bits, n_bits=n_bits, seed=seed,
                              verbose=verbose, entropy=entropy, flow_kind=flow_kind,
                              predictive_kind=predictive_kind)


def run_kate(top: str, dcd: str, *, stride=10, cv="tica", features="cartesian", cv_dim=6,
            keep_frac=0.10, epochs=300, nstates=200, lag_ns=5.0, dt_ps=100.0, lat_bits=14,
            n_bits=4, seed=0, streaming=False, chunk=2000, entropy="gaussian",
            flow_kind="realnvp", predictive_kind="gru",
            verbose=True) -> Tuple[Artifact, dict]:
    """Load a DCD trajectory and run flow-based KATE.

    The trajectory comprises the heavy atoms of a solvent-stripped system. ``cv`` is
    'tica' (the default) or 'vampnet'. ``features`` is 'cartesian' (the default) or
    'contacts', the latter using invariant internal coordinates that are physically
    preferred for kinetics. Setting ``streaming=True`` selects the out-of-core path,
    which uses chunked ``md.iterload`` with Cartesian TICA."""
    import mdtraj as md
    if streaming and cv == "vampnet":
        raise SystemExit("--cv vampnet is not supported with --streaming (the VAMPNet "
                         "needs its features in RAM). Use the in-RAM path.")
    if streaming and features == "contacts":
        raise SystemExit("--features contacts is not supported with --streaming yet "
                         "(distance featurization needs the frames in RAM). Use in-RAM.")
    dt_strided_ns = stride * dt_ps / 1000.0
    lag = max(1, int(round(lag_ns / dt_strided_ns)))
    if verbose:
        print("=" * 70)
        print("KATE on %s  (stride %d -> %.3f ns/frame, lag %d frames = %.2f ns; cv=%s%s)"
              % (dcd, stride, dt_strided_ns, lag, lag * dt_strided_ns, cv,
                 ", STREAMING" if streaming else ""))
        print("=" * 70)
    topo = md.load_topology(top)
    sel = heavy_indices(topo)
    kw = dict(cv_dim=cv_dim, keep_frac=keep_frac, epochs=epochs, nstates=nstates,
              lag=lag, stride=stride, dt_ps=dt_ps, lat_bits=lat_bits, n_bits=n_bits,
              seed=seed, verbose=verbose, entropy=entropy, flow_kind=flow_kind,
              predictive_kind=predictive_kind)
    if not streaming:
        kw["features"] = features            # contact featurization is in-memory only
    if streaming:
        def factory():
            return (np.asarray(ch.xyz, dtype=np.float64) for ch in
                    md.iterload(dcd, top=top, chunk=chunk, atom_indices=sel, stride=stride))
        return compress_streaming(factory, **kw)
    chunks = []
    for ch in md.iterload(dcd, top=top, chunk=chunk, atom_indices=sel, stride=stride):
        chunks.append(np.asarray(ch.xyz, dtype=np.float64))
    coords = np.concatenate(chunks, 0)
    if verbose:
        print("  loaded                : %s (nm), %d heavy atoms" % (coords.shape, len(sel)))
    return compress_trajectory([coords], cv=cv, **kw)


def print_report(report: dict) -> None:
    r = report
    kin = r["kinetic_bound"]
    print("-" * 70)
    if r.get("vamp_score") is not None:
        print("COLLECTIVE VARIABLES  : VAMPnet (T6), VAMP2 score = %.3f" % r["vamp_score"])
    elif r.get("features") == "contacts":
        print("COLLECTIVE VARIABLES  : TICA on internal-coordinate contacts "
              "(rotation/translation-invariant)")
    else:
        print("COLLECTIVE VARIABLES  : TICA (linear, aligned Cartesian)")
    print("ENSEMBLE FIDELITY (flow vs data)")
    print("  KL(data||flow) on CV1 : %.4f nats" % r["kl_cv1_nats"])
    print("  bounded-obs |diff|    : %.4f   <= Pinsker sqrt(KL/2)=%.4f : %s"
          % (r["bounded_obs_diff"], r["pinsker_cv1"], r["pinsker_ok"]))
    print("KEPT-FRAME RECONSTRUCTION")
    print("  frames kept           : %d / %d (%.0f%%)"
          % (r["n_keep"], r["frames"], 100 * r["keep_frac"]))
    print("  max CV recon error    : %.2e (slow modes; quantization-limited)"
          % r["cv_recon_err"])
    if "steric" in r:
        s = r["steric"]
        print("  steric validity       : recon min-dist %.4f nm vs orig %.4f nm (1st pctile) -- %s"
              % (s["rec_min_nm"], s["orig_min_nm"],
                 "OK (no new clashes)" if s["ok"] else "WARNING: reconstruction adds atom overlaps"))
    print("  full-atom RMSD        : %.4e nm (T4 residual recovers the fast modes; %d-bit)"
          % (r["fullatom_rmsd"], r["n_bits"]))
    print("ARTIFACT (ensemble + kinetics model + full-atom residual)")
    print("  flow / coded latents  : %.3f / %.3f MB"
          % (r["flow_bytes"] / 1e6, r["coded_bytes"] / 1e6))
    print("  residual / state-means: %.3f / %.3f MB  (full-atom side, charged honestly)"
          % (r["residual_bits"] / 8e6, r["state_mean_bits"] / 8e6))
    if r.get("rate_temporal_bpv") is not None:
        rg, rt = r["rate_gaussian_bpv"], r["rate_temporal_bpv"]
        print("LEARNED-ENTROPY CODING (T8; latent rate, all frames)")
        print("  bits/value  gaussian / temporal : %.4f / %.4f  (%.1f%% %s)"
              % (rg, rt, 100 * (rg - rt) / rg,
                 "saved" if rt <= rg else "WORSE -- no inter-frame redundancy here"))
        print("  NOTE: the flow already Gaussianizes per frame; T8 exploits only INTER-")
        print("  frame redundancy -- at 100 ps spacing the real-data gain may be modest.")
    if r.get("rd_curve") is not None:
        print("PREDICTIVE LEARNED-ENTROPY CODING (T9; LOSSY, %s predictor)" % r["predictive_kind"])
        print("  conditional NLL / static N(0,1) NLL : %.3f / %.3f nats  (predictor gain %.2f)"
              % (r["pred_cond_nll"], r["pred_static_nll"],
                 r["pred_static_nll"] - r["pred_cond_nll"]))
        if r.get("rate_gaussian_bpv") is not None:
            print("  static lossless floor               : %.3f bits/value" % r["rate_gaussian_bpv"])
        print("  rate-distortion curve (bits/value @ latent MSE):")
        for c in r["rd_curve"]:
            print("     %2d-bit : %6.3f bits/value   latent MSE %.2e"
                  % (c["bits"], c["rate_bpv"], c["latent_mse"]))
        print("  NOTE: rate gain over T8 is EMPIRICAL -- the rate-vs-observable-error gate")
        print("  (CV-KL, MSM timescale error, vs the path bound) runs on the trypsin set.")
    print("KINETICS (MSM dynamics term)")
    _est = r.get("msm_estimator", "symmetrized-cc")
    print("  MSM estimator         : %s" % (
        "deeptime reversible MLE (publishable)" if _est == "deeptime-mle"
        else "(C+C^T)/2 symmetrized -- deeptime absent; install kate[kinetics] for the MLE"))
    print("  implied timescales    : %s ns" % np.round(r["implied_timescales_ns"], 1))
    print("KINETIC BOUND (path-distribution; retained Q vs reference P = full-data MSM)")
    print("  ensemble term         : %.3e nats" % kin["ensemble_kl_nats"])
    print("  transition term       : %.3e nats/step  (~0 by construction: KATE retains the MSM)"
          % kin["transition_kl_rate_nats_per_step"])
    print("  Pinsker pair bound    : %.3e   (the kinetic-observable guarantee)"
          % kin["pinsker_pair_bound"])
    print("  NOTE: the contrast vs other compressors is `kate benchmark`; here Q==P so ~0.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="flow-based KATE on a DCD (kate compress backend)")
    ap.add_argument("top"); ap.add_argument("dcd")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--cv-dim", type=int, default=6)
    ap.add_argument("--keep-frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--nstates", type=int, default=200)
    ap.add_argument("--lag-ns", type=float, default=5.0)
    ap.add_argument("--dt-ps", type=float, default=100.0)
    ap.add_argument("--lat-bits", type=int, default=14)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--streaming", action="store_true", help="out-of-core (T5)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("-o", "--out", default=None, help="write artifact to this .kate path")
    a = ap.parse_args()
    art, report = run_kate(a.top, a.dcd, stride=a.stride, cv_dim=a.cv_dim,
                          keep_frac=a.keep_frac, epochs=a.epochs, nstates=a.nstates,
                          lag_ns=a.lag_ns, dt_ps=a.dt_ps, lat_bits=a.lat_bits,
                          n_bits=a.n_bits, streaming=a.streaming, chunk=a.chunk)
    print_report(report)
    if a.out:
        from .artifact import save_artifact
        save_artifact(art, a.out)
        print("  artifact written      : %s" % a.out)


if __name__ == "__main__":
    main()
