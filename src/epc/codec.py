"""
epc.py
======
Ensemble-Preserving Compression, the flow-based version from the abstract.

Pipeline (each stage is the abstract's):
  1. align (Kabsch) and learn a normalizing flow density p(x) over configurations
     -- the flow is exactly invertible, so kept frames reconstruct exactly, and a
     divergence measured in its Gaussian base z transfers to configuration space
     without a Gaussian-reference assumption.
  2. information-gain frame selection (IGFS): keep the subset of frames whose base-
     space points best COVER the density (farthest-point sampling in z) -- this
     preferentially retains rare/transition states, i.e. frames of high
     thermodynamic information. This is the lossy step; nothing is corrupted, a
     representative subset is chosen.
  3. lossless entropy coding of the kept latents against the flow's own base
     density N(0,I): the code length is -log2 p(z), the NLL in bits. The flow
     having Gaussianized the data is exactly what makes this cheap.
  4. the bound: a divergence between the compressed and original empirical
     measures bounds the error of bounded observables (Pinsker). This is a
     STATIC/ensemble guarantee. Kinetic observables are NOT covered by it -- so
     the MSM transition matrix is retained separately as the dynamics term (the
     path-distribution KL = ensemble term + transition term).

Honest scope: this demo runs the flow on the full (small) Cartesian vector so that
kept-frame reconstruction is exact. For a real protein (3N ~ 5000) you would either
train a larger flow on GPU or dimensionality-reduce first and accept loss in the
discarded fast modes. IGFS here is farthest-point sampling, one concrete instance
of an information-gain selector.
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from scipy.special import erf

from .flow import RealNVP
from .kinetic_codec import (
    kabsch_align, TICA, discretize, count_matrix, transition_matrix,
    implied_timescales, largest_connected_set,
    _BitWriter, _BitReader, _HALF, _QUARTER, _3QUARTER, _MASK, _probs_to_cumfreq,
)


# ============================================================================
# Static (i.i.d.) arithmetic coder: code integer levels against ONE fixed PMF.
# Same WNC core as the Markov coder, single cumulative table.
# ============================================================================

def encode_iid(levels: np.ndarray, cum: np.ndarray) -> bytes:
    total = int(cum[-1])
    w = _BitWriter()
    low, high, pending = 0, _MASK, 0
    for s in levels:
        c_low = int(cum[s]); c_high = int(cum[s + 1])
        rng = high - low + 1
        high = low + (rng * c_high) // total - 1
        low = low + (rng * c_low) // total
        while True:
            if high < _HALF:
                pending = w.emit(0, pending)
            elif low >= _HALF:
                pending = w.emit(1, pending); low -= _HALF; high -= _HALF
            elif low >= _QUARTER and high < _3QUARTER:
                pending += 1; low -= _QUARTER; high -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) | 1) & _MASK
    pending += 1
    w.emit(0 if low < _QUARTER else 1, pending)
    return w.to_bytes()


def decode_iid(data: bytes, n: int, cum: np.ndarray) -> np.ndarray:
    total = int(cum[-1])
    r = _BitReader(data)
    low, high, code = 0, _MASK, 0
    for _ in range(32):
        code = ((code << 1) | r.next_bit()) & _MASK
    out = np.empty(n, dtype=np.int64)
    for t in range(n):
        rng = high - low + 1
        value = (((code - low) + 1) * total - 1) // rng
        s = int(np.searchsorted(cum, value, side="right") - 1)
        s = min(max(s, 0), cum.size - 2)
        c_low = int(cum[s]); c_high = int(cum[s + 1])
        high = low + (rng * c_high) // total - 1
        low = low + (rng * c_low) // total
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                code -= _HALF; low -= _HALF; high -= _HALF
            elif low >= _QUARTER and high < _3QUARTER:
                code -= _QUARTER; low -= _QUARTER; high -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) | 1) & _MASK
            code = ((code << 1) | r.next_bit()) & _MASK
        out[t] = s
    return out


def gaussian_cumfreq(L: int, zmax: float) -> np.ndarray:
    """Integer cumulative-frequency table for N(0,1) discretized into L bins on
    [-zmax, zmax]. Encoder and decoder build the identical table."""
    edges = np.linspace(-zmax, zmax, L + 1)
    cdf = 0.5 * (1.0 + erf(edges / np.sqrt(2.0)))
    p = np.diff(cdf)
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    return _probs_to_cumfreq(p)


# ============================================================================
# Information-gain frame selection (farthest-point sampling in base space)
# ============================================================================

def igfs_select(z: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling in the flow's base space. Maximizes
    coverage of the density -> keeps diverse and rare-state frames (high
    information per frame). Returns sorted indices of kept frames."""
    T = z.shape[0]
    n_keep = min(n_keep, T)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(T))
    chosen = [start]
    d2 = ((z - z[start]) ** 2).sum(1)
    for _ in range(n_keep - 1):
        i = int(np.argmax(d2))
        chosen.append(i)
        d2 = np.minimum(d2, ((z - z[i]) ** 2).sum(1))
    return np.sort(np.array(chosen, dtype=int))


# ============================================================================
# EPC codec
# ============================================================================

@dataclass
class EPCArtifact:
    coded_latents: bytes          # entropy-coded base-space latents of kept frames
    n_keep: int
    dim: int
    L: int
    zmax: float
    flow: RealNVP                 # the invertible decoder + density (one-time model)
    kept_idx: np.ndarray          # indices of the frames retained (coverage subset)
    # dynamics term (kinetics), retained separately from the IGFS subset:
    T_msm: np.ndarray
    counts: np.ndarray
    lag: int
    tica: TICA
    centers: np.ndarray


class EPCCodec:
    def __init__(self, n_keep_frac=0.1, flow_layers=10, flow_hidden=64,
                 flow_epochs=200, lat_bits=12, zmax=5.0,
                 tica_lag=10, tica_dim=2, n_states=80, msm_lag=10, seed=0):
        self.n_keep_frac = n_keep_frac
        self.flow_layers = flow_layers
        self.flow_hidden = flow_hidden
        self.flow_epochs = flow_epochs
        self.L = 1 << lat_bits
        self.lat_bits = lat_bits
        self.zmax = zmax
        self.tica_lag = tica_lag
        self.tica_dim = tica_dim
        self.n_states = n_states
        self.msm_lag = msm_lag
        self.seed = seed

    def fit_encode(self, runs: List[np.ndarray], verbose=True) -> EPCArtifact:
        # align + flatten
        ref = None
        aligned = []
        for r in runs:
            a, ref = kabsch_align(np.asarray(r, float), ref)
            aligned.append(a.reshape(a.shape[0], -1))
        X = np.concatenate(aligned, axis=0)               # (T, 3N)
        dim = X.shape[1]

        # --- stage 1: learn the flow density p(x) ---
        if verbose:
            print("  training flow density on %d frames x %d dims ..." % X.shape)
        flow = RealNVP(dim, hidden=self.flow_hidden, n_layers=self.flow_layers)
        flow.fit(X, epochs=self.flow_epochs, batch=512, verbose=verbose, seed=self.seed)
        with torch.no_grad():
            z_all, _ = flow.forward(torch.as_tensor(X, dtype=torch.float32))
        z_all = z_all.numpy()

        # --- stage 2: information-gain frame selection ---
        n_keep = max(2, int(self.n_keep_frac * X.shape[0]))
        kept = igfs_select(z_all, n_keep, seed=self.seed)

        # --- stage 3: lossless entropy coding of kept latents vs N(0,I) ---
        # IGFS deliberately keeps tail (large-|z|) frames, so size the grid to the
        # data rather than clipping them.
        zmax = max(self.zmax, float(np.abs(z_all[kept]).max()) * 1.02)
        cum = gaussian_cumfreq(self.L, zmax)
        zc = np.clip(z_all[kept], -zmax, zmax)
        levels = np.floor((zc + zmax) / (2 * zmax) * self.L).astype(np.int64)
        levels = np.clip(levels, 0, self.L - 1).ravel()
        coded = encode_iid(levels, cum)

        # --- dynamics term: MSM on the FULL trajectory's slow CVs ---
        tica = TICA(lag=self.tica_lag, n_components=self.tica_dim)
        white_runs = []
        off = 0
        for a in aligned:
            white_runs.append(X[off:off + a.shape[0]])
            off += a.shape[0]
        tica.fit(white_runs)
        tica_runs = [tica.transform(w) for w in white_runs]
        labels, centers = discretize(tica_runs, self.n_states, self.seed)
        C = count_matrix(labels, self.n_states, self.msm_lag)
        T_msm, _ = transition_matrix(C, reversible=True)

        return EPCArtifact(coded_latents=coded, n_keep=len(kept), dim=dim,
                           L=self.L, zmax=zmax, flow=flow, kept_idx=kept,
                           T_msm=T_msm, counts=C, lag=self.msm_lag, tica=tica,
                           centers=centers)

    @staticmethod
    def decode_ensemble(ct: EPCArtifact) -> np.ndarray:
        """Reconstruct the kept representative configurations (the compressed
        ensemble). Exact up to latent quantization, because the flow inverts
        exactly."""
        cum = gaussian_cumfreq(ct.L, ct.zmax)
        levels = decode_iid(ct.coded_latents, ct.n_keep * ct.dim, cum)
        levels = levels.reshape(ct.n_keep, ct.dim)
        z = -ct.zmax + (levels + 0.5) * (2 * ct.zmax / ct.L)
        with torch.no_grad():
            x = ct.flow.inverse(torch.as_tensor(z, dtype=torch.float32)).numpy()
        N = ct.dim // 3
        return x.reshape(ct.n_keep, N, 3)

    @staticmethod
    def kinetics(ct: EPCArtifact, k=5):
        active = largest_connected_set(ct.counts)
        Tc, _ = transition_matrix(ct.counts[np.ix_(active, active)], reversible=True)
        return implied_timescales(Tc, ct.lag, k)
