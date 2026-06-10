"""
demo_epc.py
===========
End-to-end flow-based EPC on synthetic metastable data with known ground truth.

What it demonstrates (the abstract, made concrete and measured):
  - the learned flow REPRODUCES THE ENSEMBLE: the free-energy profile and state
    populations from samples of the trained flow match the full trajectory, and
    the measured KL is small (this is the divergence the bound is built on);
  - bounded observables obey the Pinsker envelope |dExpectation| <= sqrt(KL/2);
  - kept frames reconstruct EXACTLY (flow is invertible), entropy-coded against
    the Gaussian base;
  - kinetics come from the separately retained MSM transition matrix (the
    dynamics term the ensemble bound does NOT cover).

Synthetic, illustrative numbers. The point is the mechanism and the bound, not
absolute rates/ratios on this toy.
"""

import numpy as np
import torch
from epc.codec import EPCCodec
from epc.kinetic_codec import kabsch_align

WELLS = np.array([-2.0, 0.0, 2.0])


def simulate_run(n, n_atoms, a=0.01, intra=0.25, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[1 - a, a, 0], [a, 1 - 2 * a, a], [0, a, 1 - a]])
    cdf = np.cumsum(P, 1); u = rng.random(n); m = np.empty(n, int); m[0] = 0
    for t in range(1, n):
        m[t] = np.searchsorted(cdf[m[t - 1]], u[t])
    xi = WELLS[m] + intra * rng.standard_normal(n)
    ref = rng.standard_normal((n_atoms, 3)) * 2.0
    mode = rng.standard_normal((n_atoms, 3)); mode /= np.linalg.norm(mode)
    xyz = (ref[None] + xi[:, None, None] * mode[None]
           + noise * rng.standard_normal((n, n_atoms, 3)))
    return xyz.astype(np.float64), m


def free_energy_1d(v, bins):
    h, _ = np.histogram(v, bins=bins, density=True)
    h = np.clip(h, 1e-8, None)
    return -np.log(h)


def kl_1d(p_samples, q_samples, bins):
    """KL(P||Q) between two 1-D empirical distributions on shared bins."""
    p, _ = np.histogram(p_samples, bins=bins, density=False)
    q, _ = np.histogram(q_samples, bins=bins, density=False)
    p = p / p.sum(); q = q / q.sum()
    p = np.clip(p, 1e-8, None); q = np.clip(q, 1e-8, None)
    p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def main():
    np.set_printoptions(precision=4, suppress=True)
    N_ATOMS, NOISE = 10, 0.10
    runs, ms = [], []
    for k in range(4):
        c, m = simulate_run(2000, N_ATOMS, noise=NOISE, seed=10 + k)
        runs.append(c); ms.append(m)
    T = sum(len(r) for r in runs)

    codec = EPCCodec(n_keep_frac=0.10, flow_layers=10, flow_hidden=64,
                     flow_epochs=400, lat_bits=14, tica_lag=10, tica_dim=2,
                     n_states=80, msm_lag=10, seed=0)
    print("=" * 68)
    print("EPC  (synthetic; %d atoms, %d frames, 4 runs)" % (N_ATOMS, T))
    print("=" * 68)
    ct = codec.fit_encode(runs, verbose=True)

    # re-align to get the full coordinate matrix in the codec's frame
    ref = None; aligned = []
    for r in runs:
        a, ref = kabsch_align(r, ref); aligned.append(a.reshape(a.shape[0], -1))
    X = np.concatenate(aligned, 0)

    # ---- ENSEMBLE FIDELITY: flow samples vs full trajectory, in CV space ----
    with torch.no_grad():
        samp = ct.flow.sample(40000).numpy()
    cv_full = ct.tica.transform(X)[:, 0]
    cv_samp = ct.tica.transform(samp)[:, 0]
    lo, hi = np.percentile(cv_full, [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    F_full = free_energy_1d(cv_full, bins)
    F_samp = free_energy_1d(cv_samp, bins)
    F_full -= F_full.min(); F_samp -= F_samp.min()
    counts, _ = np.histogram(cv_full, bins=bins)
    pop = counts >= max(20, int(0.002 * counts.sum()))   # populated bins only
    dF = np.abs(F_full - F_samp)[pop]
    KL = kl_1d(cv_full, cv_samp, bins)

    print("-" * 68)
    print("ENSEMBLE FIDELITY  (flow density vs full trajectory)")
    print("  KL(full || flow) along CV1     : %.4f nats" % KL)
    print("  max |dF(CV1)| (populated range): %.3f kT" % dF.max())

    # Pinsker check on a bounded observable: indicator CV1 > midpoint
    mid = 0.5 * (lo + hi)
    p_full = float((cv_full > mid).mean())
    p_samp = float((cv_samp > mid).mean())
    pinsker = np.sqrt(KL / 2.0)
    print("  bounded observable  P(CV1>mid) : full=%.3f  flow=%.3f  |diff|=%.4f"
          % (p_full, p_samp, abs(p_full - p_samp)))
    print("  Pinsker bound sqrt(KL/2)       : %.4f   -> satisfied: %s"
          % (pinsker, abs(p_full - p_samp) <= pinsker + 1e-9))

    # ---- EXACT RECONSTRUCTION of kept frames ----
    rec = EPCCodec.decode_ensemble(ct)
    orig_kept = X[ct.kept_idx].reshape(-1, N_ATOMS, 3)
    rmsd = np.sqrt(((rec - orig_kept) ** 2).sum(2).mean(1))
    print("-" * 68)
    print("KEPT-FRAME RECONSTRUCTION  (flow is invertible)")
    print("  frames kept                    : %d / %d (%.0f%%)"
          % (ct.n_keep, T, 100 * ct.n_keep / T))
    print("  max reconstruction RMSD        : %.5f nm (quantization-limited)"
          % rmsd.max())

    # ---- COMPRESSION ACCOUNTING ----
    flow_bits = sum(p.numel() for p in ct.flow.parameters()) * 32
    coded_bits = len(ct.coded_latents) * 8
    msm_bits = ct.counts.size * 16 + ct.centers.size * 32 + ct.tica.eigvecs_.size * 32
    orig_bits = T * ct.dim * 32
    artifact = flow_bits + coded_bits + msm_bits
    print("-" * 68)
    print("ARTIFACT SIZE")
    print("  flow (density+decoder)         : %.3f MB" % (flow_bits / 8 / 1e6))
    print("  coded kept latents             : %.3f MB" % (coded_bits / 8 / 1e6))
    print("  MSM + TICA side info           : %.3f MB" % (msm_bits / 8 / 1e6))
    print("  total artifact / original      : %.3f / %.3f MB  (%.2fx)"
          % (artifact / 8 / 1e6, orig_bits / 8 / 1e6, orig_bits / artifact))
    print("  (flow is a fixed cost; the ratio grows with trajectory length, and")
    print("   the ensemble is reproduced by the flow regardless of frames kept.)")

    # ---- KINETICS from the retained MSM ----
    P = np.array([[0.99, 0.01, 0], [0.01, 0.98, 0.01], [0, 0.01, 0.99]])
    ev = np.sort(np.real(np.linalg.eigvals(P)))[::-1]
    gt = -1.0 / np.log(ev[1:3])
    its = EPCCodec.kinetics(ct, k=4)
    print("-" * 68)
    print("KINETICS  (retained MSM transition matrix -- the dynamics term)")
    print("  ground-truth slow timescales   : %s frames" % np.round(gt, 1))
    print("  recovered slow timescales      : %s frames" % np.round(its[:2], 1))
    print("=" * 68)


if __name__ == "__main__":
    main()
