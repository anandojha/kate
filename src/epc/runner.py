#!/usr/bin/env python
"""
runner.py -- flow-based EPC on real (or array) trajectories, the SCALABLE way, and
the backend for ``epc compress``.

Unlike examples/demo_epc.py (flow on full Cartesian, only for tiny systems), this
reduces to a few TICA collective variables FIRST and trains the flow on those -- the
abstract's stage-1 ordering -- so it is tractable at 3N ~ 5000. The CV-space ensemble
is reconstructed exactly; full-atom reconstruction of the fast modes is the residual
stage [T4].

T1 additions over the original run_epc.py script:
  * factored into ``compress_trajectory(coords_runs, ...) -> (Artifact, report)`` (the
    core, run-aware) and ``run_epc(top, dcd, ...)`` (the DCD loader);
  * the path bound is wired in: after the retained MSM (Q) we also form the reference
    MSM (P) from the full-data discretization and call
    ``pathbound.report_kinetic_fidelity(P, Q)``. For EPC P == Q (the artifact RETAINS
    the MSM), so the transition term is ~0 by construction -- EPC's kinetic distortion
    is ~0. The bound's real use as a contrast SCORER is ``epc bound`` / ``epc benchmark``.

USAGE (compute node, not a login node):
  python -m epc.runner TOP DCD [--stride 10] [--cv-dim 6] [--keep-frac 0.1]
       [--epochs 300] [--nstates 200] [--lag-ns 5] [--dt-ps 100] [-o OUT.epc]
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch

from .flow import RealNVP
from .codec import igfs_select, encode_iid, decode_iid, gaussian_cumfreq
from .kinetic_codec import (kabsch_align, TICA, discretize, count_matrix,
                            transition_matrix, implied_timescales,
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


def compress_trajectory(coords_runs: List[np.ndarray], *, cv_dim=6, keep_frac=0.10,
                        epochs=300, nstates=200, lag=10, stride=1, dt_ps=100.0,
                        lat_bits=14, n_bits=4, seed=0, verbose=True) -> Tuple[Artifact, dict]:
    """Run-aware flow-based EPC on a list of (T_i, N, 3) coordinate arrays (nm).
    Returns (Artifact, report dict). The MSM/TICA lag is in (strided) frames."""
    dt_strided_ns = stride * dt_ps / 1000.0

    # --- align (run-aware, shared reference) + flatten ---
    ref = None
    aligned = []
    for r in coords_runs:
        a, ref = kabsch_align(np.asarray(r, dtype=np.float64), ref)
        aligned.append(a.reshape(a.shape[0], -1))
    run_lengths = [a.shape[0] for a in aligned]
    X_all = np.concatenate(aligned, axis=0)
    T_total = X_all.shape[0]

    # --- stage 1: TICA -> CVs ---
    tica = TICA(lag=lag, n_components=cv_dim).fit(aligned)
    CV_runs = [tica.transform(a).astype(np.float32) for a in aligned]
    CV_all = np.concatenate(CV_runs, axis=0)
    if verbose:
        print("  TICA CVs              : %s   leading timescales (frames): %s"
              % (CV_all.shape, np.round(tica.timescales_[:cv_dim], 1)))

    # --- stage 2: flow density on the CVs ---
    if verbose:
        print("  training flow on %d-D CVs ..." % cv_dim)
    flow = RealNVP(cv_dim, hidden=64, n_layers=10).fit(
        CV_all, epochs=epochs, batch=1024, verbose=verbose, seed=seed)
    with torch.no_grad():
        z_all = flow.forward(torch.as_tensor(CV_all))[0].numpy()

    # --- stage 3: IGFS + lossless coding of kept latents (vs N(0,I)) ---
    n_keep = max(2, int(keep_frac * T_total))
    kept = igfs_select(z_all, n_keep, seed=seed)
    L = 1 << lat_bits
    zmax = max(6.0, float(np.abs(z_all[kept]).max()) * 1.02)
    cum = gaussian_cumfreq(L, zmax)
    lev = np.clip(np.floor((np.clip(z_all[kept], -zmax, zmax) + zmax) /
                           (2 * zmax) * L).astype(np.int64), 0, L - 1).ravel()
    coded = encode_iid(lev, cum)

    # CV-space reconstruction error (exact up to quantization)
    dlev = decode_iid(coded, len(kept) * cv_dim, cum).reshape(len(kept), cv_dim)
    zrec = -zmax + (dlev + 0.5) * (2 * zmax / L)
    with torch.no_grad():
        cv_rec = flow.inverse(torch.as_tensor(zrec, dtype=torch.float32)).numpy()
    cv_err = float(np.abs(cv_rec - CV_all[kept]).max())

    # --- ensemble fidelity (flow vs data) + Pinsker on CV1 ---
    with torch.no_grad():
        samp = flow.sample(40000).numpy()
    lo, hi = np.percentile(CV_all[:, 0], [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    KL = kl_1d(CV_all[:, 0], samp[:, 0], bins)
    mid = 0.5 * (lo + hi)
    obs = abs(float((CV_all[:, 0] > mid).mean()) - float((samp[:, 0] > mid).mean()))

    # --- dynamics term: run-aware MSM on the CVs ---
    labels, centers = discretize(CV_runs, nstates, seed)
    C = count_matrix(labels, nstates, lag)
    T_msm, _ = transition_matrix(C, reversible=True)

    # --- the path bound, wired in (T1): reference P (full data) vs retained Q ---
    act = largest_connected_set(C)
    Tm_act, _ = transition_matrix(C[np.ix_(act, act)], reversible=True)
    its = implied_timescales(Tm_act, lag, 5)
    its_ns = its * dt_strided_ns
    # EPC RETAINS the MSM -> Q == P on the kept dynamics -> transition term ~ 0.
    kin = report_kinetic_fidelity(Tm_act, Tm_act, lag=lag, L=T_total, k=4)

    # --- T4: full-atom residual stage (the fast modes TICA discards) ---
    # The kept frames' CV reconstructs the SLOW modes (flow inverse of the decoded
    # latents); the residual recovers the rest of the 3N coordinates. It is defined
    # against the DECODED CV (cv_rec), so it absorbs the latent quantization exactly,
    # then a per-state mean + a subtractive-dithered uniform quantizer code it cheaply.
    labels_all = np.concatenate(labels)
    Vp = np.linalg.pinv(tica.eigvecs_)                       # (cv_dim, 3N)
    X_approx_kept = tica.mean_ + cv_rec @ Vp                 # CV-subspace recon (decoded)
    R_kept = X_all[kept] - X_approx_kept                     # discarded fast modes (3N)
    labels_kept = labels_all[kept]
    D = X_all.shape[1]
    state_mean_R = np.zeros((nstates, D))
    for s in np.unique(labels_kept):
        state_mean_R[s] = R_kept[labels_kept == s].mean(axis=0)
    resid = R_kept - state_mean_R[labels_kept]
    rstep = float((resid.std() + 1e-12) * (2.0 ** (1 - n_bits)) * 3.0)
    rcodec = DitheredResidualCodec(n_bits=n_bits, seed=seed)
    rq = rcodec.quantize(resid, rstep)
    residual = {"state_mean_R": state_mean_R, "q": rq, "step": rstep,
                "n_bits": n_bits, "seed": seed, "labels_kept": labels_kept,
                "n_atoms": D // 3}
    # honest self-check of the residual round-trip (full-atom recon error)
    full_rec = X_approx_kept + rcodec.dequantize(rq, rstep) + state_mean_R[labels_kept]
    fullatom_rmsd = float(np.sqrt(((full_rec - X_all[kept]) ** 2)
                                  .reshape(len(kept), -1, 3).sum(2).mean()))

    artifact = Artifact(
        cv_dim=cv_dim, L=L, zmax=zmax, n_keep=len(kept),
        coded_latents=coded, kept_idx=kept,
        run_lengths=run_lengths, dtraj=labels, centers=centers, counts=C,
        T_msm=T_msm, n_states=nstates, lag=lag,
        stride=stride, dt_ps=dt_ps, dt_strided_ns=dt_strided_ns,
        flow_arch={"dim": cv_dim, "hidden": 64, "n_layers": 10},
        cv="tica", flow_kind="realnvp", entropy="gaussian",
        tica_mean=tica.mean_, tica_eigvecs=tica.eigvecs_,
        tica_timescales=tica.timescales_,
        align_ref=ref, x_mean=X_all.mean(axis=0),
        residual=residual,
        flow_state={k: v.detach().cpu() for k, v in flow.state_dict().items()},
    )

    flow_bytes = sum(p.numel() for p in flow.parameters()) * 4
    residual_bits = int(rq.size) * n_bits
    state_mean_bits = int(state_mean_R.size) * 32
    report = {
        "frames": T_total, "run_lengths": run_lengths,
        "kl_cv1_nats": KL, "pinsker_cv1": float(np.sqrt(KL / 2)),
        "bounded_obs_diff": obs, "pinsker_ok": bool(obs <= np.sqrt(KL / 2) + 1e-9),
        "n_keep": len(kept), "keep_frac": len(kept) / T_total, "cv_recon_err": cv_err,
        "fullatom_rmsd": fullatom_rmsd, "n_bits": n_bits,
        "residual_bits": residual_bits, "state_mean_bits": state_mean_bits,
        "flow_bytes": flow_bytes, "coded_bytes": len(coded),
        "implied_timescales_ns": its_ns, "lag": lag, "dt_strided_ns": dt_strided_ns,
        "kinetic_bound": kin,
    }
    return artifact, report


def run_epc(top: str, dcd: str, *, stride=10, cv_dim=6, keep_frac=0.10, epochs=300,
            nstates=200, lag_ns=5.0, dt_ps=100.0, lat_bits=14, seed=0,
            verbose=True) -> Tuple[Artifact, dict]:
    """Load a DCD (heavy atoms of a solvent-stripped system) and run flow-based EPC.
    The DCD backend for ``epc compress``."""
    import mdtraj as md
    dt_strided_ns = stride * dt_ps / 1000.0
    lag = max(1, int(round(lag_ns / dt_strided_ns)))
    if verbose:
        print("=" * 70)
        print("EPC on %s  (stride %d -> %.3f ns/frame, lag %d frames = %.2f ns)"
              % (dcd, stride, dt_strided_ns, lag, lag * dt_strided_ns))
        print("=" * 70)
    topo = md.load_topology(top)
    sel = heavy_indices(topo)
    chunks = []
    for ch in md.iterload(dcd, top=top, chunk=2000, atom_indices=sel, stride=stride):
        chunks.append(np.asarray(ch.xyz, dtype=np.float64))
    coords = np.concatenate(chunks, 0)
    if verbose:
        print("  loaded                : %s (nm), %d heavy atoms" % (coords.shape, len(sel)))
    return compress_trajectory([coords], cv_dim=cv_dim, keep_frac=keep_frac,
                               epochs=epochs, nstates=nstates, lag=lag, stride=stride,
                               dt_ps=dt_ps, lat_bits=lat_bits, seed=seed, verbose=verbose)


def print_report(report: dict) -> None:
    r = report
    kin = r["kinetic_bound"]
    print("-" * 70)
    print("ENSEMBLE FIDELITY (flow vs data)")
    print("  KL(data||flow) on CV1 : %.4f nats" % r["kl_cv1_nats"])
    print("  bounded-obs |diff|    : %.4f   <= Pinsker sqrt(KL/2)=%.4f : %s"
          % (r["bounded_obs_diff"], r["pinsker_cv1"], r["pinsker_ok"]))
    print("KEPT-FRAME RECONSTRUCTION")
    print("  frames kept           : %d / %d (%.0f%%)"
          % (r["n_keep"], r["frames"], 100 * r["keep_frac"]))
    print("  max CV recon error    : %.2e (slow modes; quantization-limited)"
          % r["cv_recon_err"])
    print("  full-atom RMSD        : %.4e nm (T4 residual recovers the fast modes; %d-bit)"
          % (r["fullatom_rmsd"], r["n_bits"]))
    print("ARTIFACT (ensemble + kinetics model + full-atom residual)")
    print("  flow / coded latents  : %.3f / %.3f MB"
          % (r["flow_bytes"] / 1e6, r["coded_bytes"] / 1e6))
    print("  residual / state-means: %.3f / %.3f MB  (full-atom side, charged honestly)"
          % (r["residual_bits"] / 8e6, r["state_mean_bits"] / 8e6))
    print("KINETICS (MSM dynamics term)")
    print("  implied timescales    : %s ns" % np.round(r["implied_timescales_ns"], 1))
    print("KINETIC BOUND (path-distribution; retained Q vs reference P = full-data MSM)")
    print("  ensemble term         : %.3e nats" % kin["ensemble_kl_nats"])
    print("  transition term       : %.3e nats/step  (~0 by construction: EPC retains the MSM)"
          % kin["transition_kl_rate_nats_per_step"])
    print("  Pinsker pair bound    : %.3e   (the kinetic-observable guarantee)"
          % kin["pinsker_pair_bound"])
    print("  NOTE: the contrast vs other compressors is `epc benchmark`; here Q==P so ~0.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="flow-based EPC on a DCD (epc compress backend)")
    ap.add_argument("top"); ap.add_argument("dcd")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--cv-dim", type=int, default=6)
    ap.add_argument("--keep-frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--nstates", type=int, default=200)
    ap.add_argument("--lag-ns", type=float, default=5.0)
    ap.add_argument("--dt-ps", type=float, default=100.0)
    ap.add_argument("--lat-bits", type=int, default=14)
    ap.add_argument("-o", "--out", default=None, help="write artifact to this .epc path")
    a = ap.parse_args()
    art, report = run_epc(a.top, a.dcd, stride=a.stride, cv_dim=a.cv_dim,
                          keep_frac=a.keep_frac, epochs=a.epochs, nstates=a.nstates,
                          lag_ns=a.lag_ns, dt_ps=a.dt_ps, lat_bits=a.lat_bits)
    print_report(report)
    if a.out:
        from .artifact import save_artifact
        save_artifact(art, a.out)
        print("  artifact written      : %s" % a.out)


if __name__ == "__main__":
    main()
