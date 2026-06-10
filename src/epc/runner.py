#!/usr/bin/env python
"""
run_epc.py -- run flow-based EPC on a real DCD, the SCALABLE way.

Unlike demo_epc.py (flow on full Cartesian, only for tiny systems), this reduces
to a few TICA collective variables FIRST and trains the flow on those -- the
abstract's stage-1 ordering -- so it is tractable at 3N ~ 5000. The CV-space
ensemble is reconstructed exactly; full-atom reconstruction of the fast modes is
the residual-coding extension (not in this script).

USAGE (on a COMPUTE node, not a login node):
  python run_epc.py TOP DCD [--stride 10] [--cv-dim 6] [--keep-frac 0.1]
                            [--epochs 300] [--nstates 200] [--lag-ns 5] [--dt-ps 100]

dt-ps is YOUR DCDReporter interval (TRAJ_INTERVAL * timestep). For your run that
is 50000 * 0.002 = 100 ps. The MSM/TICA lag in frames is derived from lag-ns.
Assumes a solvent-stripped trajectory (selects all heavy atoms = protein+ligand).
"""

import argparse
import numpy as np
import torch

from .flow import RealNVP
from .codec import igfs_select, encode_iid, decode_iid, gaussian_cumfreq
from .kinetic_codec import (kabsch_align, TICA, discretize, count_matrix,
                           transition_matrix, implied_timescales,
                           largest_connected_set)
from .inspect_traj import heavy_indices


def free_energy_1d(v, bins):
    h, _ = np.histogram(v, bins=bins, density=True)
    return -np.log(np.clip(h, 1e-8, None))


def kl_1d(p, q, bins):
    a, _ = np.histogram(p, bins=bins); b, _ = np.histogram(q, bins=bins)
    a = np.clip(a / a.sum(), 1e-8, None); b = np.clip(b / b.sum(), 1e-8, None)
    a /= a.sum(); b /= b.sum()
    return float((a * np.log(a / b)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("top"); ap.add_argument("dcd")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--cv-dim", type=int, default=6)
    ap.add_argument("--keep-frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--nstates", type=int, default=200)
    ap.add_argument("--lag-ns", type=float, default=5.0)
    ap.add_argument("--dt-ps", type=float, default=100.0)
    ap.add_argument("--lat-bits", type=int, default=14)
    a = ap.parse_args()

    import mdtraj as md
    # frame interval after striding, in ns; MSM/TICA lag in (strided) frames
    dt_strided_ns = a.stride * a.dt_ps / 1000.0
    lag = max(1, int(round(a.lag_ns / dt_strided_ns)))

    print("=" * 70)
    print("EPC on %s  (stride %d -> %.3f ns/frame, lag %d frames = %.2f ns)"
          % (a.dcd, a.stride, dt_strided_ns, lag, lag * dt_strided_ns))
    print("=" * 70)

    topo = md.load_topology(a.top)
    sel = heavy_indices(topo)                      # all heavy atoms (= prot+lig if stripped)
    chunks = []
    for ch in md.iterload(a.dcd, top=a.top, chunk=2000, atom_indices=sel, stride=a.stride):
        chunks.append(np.asarray(ch.xyz, dtype=np.float64))
    coords = np.concatenate(chunks, 0)
    T = coords.shape[0]
    print("  loaded                : %s (nm), %d heavy atoms" % (coords.shape, len(sel)))

    # align + flatten
    aligned, _ = kabsch_align(coords, None)
    X = aligned.reshape(T, -1)

    # ---- stage 1: TICA -> CVs ----
    tica = TICA(lag=lag, n_components=a.cv_dim).fit([X])
    CV = tica.transform(X).astype(np.float32)
    print("  TICA CVs              : %s   leading timescales (frames): %s"
          % (CV.shape, np.round(tica.timescales_[:a.cv_dim], 1)))

    # ---- stage 2: flow density on the CVs ----
    print("  training flow on %d-D CVs ..." % a.cv_dim)
    flow = RealNVP(a.cv_dim, hidden=64, n_layers=10).fit(
        CV, epochs=a.epochs, batch=1024, verbose=True)
    with torch.no_grad():
        z = flow.forward(torch.as_tensor(CV))[0].numpy()

    # ---- stage 3: IGFS + lossless coding of kept latents ----
    n_keep = max(2, int(a.keep_frac * T))
    kept = igfs_select(z, n_keep, seed=0)
    L = 1 << a.lat_bits
    zmax = max(6.0, float(np.abs(z[kept]).max()) * 1.02)
    cum = gaussian_cumfreq(L, zmax)
    lev = np.clip(np.floor((np.clip(z[kept], -zmax, zmax) + zmax) /
                           (2 * zmax) * L).astype(np.int64), 0, L - 1).ravel()
    coded = encode_iid(lev, cum)

    # CV-space reconstruction (exact up to quantization)
    dlev = decode_iid(coded, len(kept) * a.cv_dim, cum).reshape(len(kept), a.cv_dim)
    zrec = -zmax + (dlev + 0.5) * (2 * zmax / L)
    with torch.no_grad():
        cv_rec = flow.inverse(torch.as_tensor(zrec, dtype=torch.float32)).numpy()
    cv_err = np.abs(cv_rec - CV[kept]).max()

    # ---- ensemble fidelity (flow vs data) + Pinsker ----
    with torch.no_grad():
        samp = flow.sample(40000).numpy()
    lo, hi = np.percentile(CV[:, 0], [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    Ff = free_energy_1d(CV[:, 0], bins); Fs = free_energy_1d(samp[:, 0], bins)
    Ff -= Ff.min(); Fs -= Fs.min()
    counts, _ = np.histogram(CV[:, 0], bins=bins)
    popm = counts >= max(20, int(0.002 * counts.sum()))
    dF = np.abs(Ff - Fs)[popm]
    KL = kl_1d(CV[:, 0], samp[:, 0], bins)
    mid = 0.5 * (lo + hi)
    obs = abs(float((CV[:, 0] > mid).mean()) - float((samp[:, 0] > mid).mean()))

    # ---- kinetics from the MSM (dynamics term) ----
    labels, centers = discretize([CV], a.nstates, 0)
    C = count_matrix([labels[0]], a.nstates, lag)
    act = largest_connected_set(C)
    Tm, _ = transition_matrix(C[np.ix_(act, act)], reversible=True)
    its = implied_timescales(Tm, lag, 5)
    its_ns = its * dt_strided_ns

    # ---- report ----
    flow_bytes = sum(p.numel() for p in flow.parameters()) * 4
    artifact = flow_bytes + len(coded) + (C.size * 2 + centers.size * 4 + tica.eigvecs_.size * 4)
    print("-" * 70)
    print("ENSEMBLE FIDELITY (flow vs data)")
    print("  KL(data||flow) on CV1 : %.4f nats" % KL)
    print("  max |dF(CV1)|         : %.2f kT (populated)" % dF.max())
    print("  bounded-obs |diff|    : %.4f   <= Pinsker sqrt(KL/2)=%.4f : %s"
          % (obs, np.sqrt(KL / 2), obs <= np.sqrt(KL / 2) + 1e-9))
    print("KEPT-FRAME (CV-space) RECONSTRUCTION")
    print("  frames kept           : %d / %d (%.0f%%)" % (len(kept), T, 100 * len(kept) / T))
    print("  max CV recon error    : %.2e (quantization-limited; full-atom = residual ext.)"
          % cv_err)
    print("ARTIFACT (ensemble + kinetics model)")
    print("  flow / coded / total  : %.3f / %.3f / %.3f MB"
          % (flow_bytes / 1e6, len(coded) / 1e6, artifact / 1e6))
    print("KINETICS (MSM dynamics term)")
    print("  implied timescales    : %s ns" % np.round(its_ns, 1))
    print("  trajectory duration   : %.2f ns (%d frames @ %.1f ps)"
          % (T * a.stride * a.dt_ps / 1000, T, a.stride * a.dt_ps))
    print("  NOTE: TICA-on-Cartesian, single lag, single run -> first-pass kinetics;")
    print("        validate with a lag scan, contact features, and more sampling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
