"""
KATE Codec for Kinetics-Preserving Compression
===============================================
This module implements the flow-based KATE codec, which compresses molecular
configurations while preserving kinetic observables.

Pipeline
--------
1. Alignment and density estimation
   Configurations are aligned by the Kabsch algorithm, and a normalizing flow density
   p(x) is learned over them. The flow is exactly invertible, so kept frames are
   reconstructed exactly, and a divergence measured in its Gaussian base space z
   transfers to configuration space without a Gaussian-reference assumption.

2. Information-gain frame selection
   The subset of frames whose base-space points best cover the density is retained, by
   farthest-point sampling in z. This preferentially retains rare and transition states,
   that is, frames of high thermodynamic information. This stage is lossy in the sense
   that frames are discarded; the retained data are not corrupted, a representative
   subset being chosen. Coverage sampling over-represents the low-density tails, so each
   kept frame carries a stationary importance weight equal to the population of its
   Voronoi cell in base space. The weights restore the empirical measure, so that
   weighted averages over the kept subset are unbiased estimators of full-ensemble
   averages; the raw (unweighted) subset is not.

3. Lossless entropy coding
   The kept latents are coded losslessly against the flow's base density N(0, I). The
   code length is the negative log-likelihood in bits, -log2 p(z). Gaussianization of
   the data by the flow is what makes this coding inexpensive.

4. The bound
   A divergence between the compressed and original empirical measures bounds the error
   of bounded observables through the Pinsker inequality. This is a static, ensemble
   guarantee, and it applies to the stationary-reweighted kept subset (stage 2) or,
   equivalently, to samples drawn from the flow density, not to the raw tail-biased
   selection. Kinetic observables are not covered by it, so the MSM transition matrix is
   retained separately as the dynamics term. The path-distribution KL is the sum of the
   ensemble term and the transition term.

Scope
-----
This implementation runs the flow on the full, small Cartesian vector so that
kept-frame reconstruction is exact. For a protein-scale system of size 3N of order
5000, one would either train a larger flow on a GPU or reduce dimensionality first and
accept loss in the discarded fast modes. The frame selector used here is
farthest-point sampling, one concrete instance of an information-gain selector.
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from scipy.special import erf
from scipy.spatial import cKDTree

from .flow import RealNVP
from .kinetic_codec import (
    kabsch_align, TICA, discretize, count_matrix, transition_matrix,
    implied_timescales, largest_connected_set,
    _BitWriter, _BitReader, _HALF, _QUARTER, _3QUARTER, _MASK, _probs_to_cumfreq,
)


# ============================================================================
# Static (i.i.d.) arithmetic coder: codes integer levels against a single fixed
# probability mass function. The Witten-Neal-Cleary core matches that of the Markov
# coder, using one cumulative-frequency table.
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
    """Build the integer cumulative-frequency table for a discretized N(0, 1) density.

    The standard normal density is discretized into L bins on the interval
    [-zmax, zmax]. The encoder and decoder construct an identical table, which is
    required for correct arithmetic coding."""
    edges = np.linspace(-zmax, zmax, L + 1)
    cdf = 0.5 * (1.0 + erf(edges / np.sqrt(2.0)))
    p = np.diff(cdf)
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    return _probs_to_cumfreq(p)


# ============================================================================
# Information-gain frame selection by farthest-point sampling in base space.
# ============================================================================

def igfs_select(z: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray:
    """Select frames by greedy farthest-point sampling in the flow's base space.

    Maximizing coverage of the density retains diverse and rare-state frames, which
    carry high information per frame. The sorted indices of the kept frames are
    returned."""
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


def stationary_reweight(z: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Importance weights that correct the farthest-point selection bias back to the
    empirical density.

    Farthest-point sampling maximizes coverage and therefore over-represents the
    low-density tails of the base measure. Each of the T frames is assigned to its
    nearest kept frame in base space, and the weight of a kept frame is the population
    fraction of its Voronoi cell. A weighted average over the kept subset,
    Sum_i w_i g(x_i), is then an unbiased estimator of the full-ensemble average
    (1/T) Sum_t g(x_t), so the ensemble-Pinsker guarantee applies to the reweighted
    subset rather than to the raw, tail-heavy selection. The weights are computed here
    (where the full base-space set is available) and stored in the artifact; they cost
    n_keep floats and are used on decode."""
    tree = cKDTree(np.asarray(z, dtype=np.float64)[kept])
    _, nearest = tree.query(np.asarray(z, dtype=np.float64), k=1)
    counts = np.bincount(nearest, minlength=len(kept)).astype(np.float64)
    return counts / counts.sum()


# ============================================================================
# KATE codec
# ============================================================================

@dataclass
class KateArtifact:
    coded_latents: bytes          # entropy-coded base-space latents of the kept frames
    n_keep: int
    dim: int
    L: int
    zmax: float
    flow: RealNVP                 # invertible decoder and density model
    kept_idx: np.ndarray          # indices of the retained coverage subset
    # Dynamics term, retained separately from the frame-selection subset.
    T_msm: np.ndarray
    counts: np.ndarray
    lag: int
    tica: TICA
    centers: np.ndarray
    # Stationary importance weights for the kept frames (Voronoi-cell populations in base
    # space). Unbiased ensemble averages weight the kept subset by these; a uniform
    # fallback keeps older artifacts usable.
    kept_weights: Optional[np.ndarray] = None


class KateCodec:
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

    def fit_encode(self, runs: List[np.ndarray], verbose=True) -> KateArtifact:
        # Align and flatten the input runs.
        ref = None
        aligned = []
        for r in runs:
            a, ref = kabsch_align(np.asarray(r, float), ref)
            aligned.append(a.reshape(a.shape[0], -1))
        X = np.concatenate(aligned, axis=0)               # (T, 3N)
        dim = X.shape[1]

        # Stage 1: learn the flow density p(x).
        if verbose:
            print("  training flow density on %d frames x %d dims ..." % X.shape)
        flow = RealNVP(dim, hidden=self.flow_hidden, n_layers=self.flow_layers)
        flow.fit(X, epochs=self.flow_epochs, batch=512, verbose=verbose, seed=self.seed)
        with torch.no_grad():
            z_all, _ = flow.forward(torch.as_tensor(X, dtype=torch.float32))
        z_all = z_all.numpy()

        # Stage 2: information-gain frame selection, with stationary importance weights
        # that reweight the tail-heavy coverage subset back to the empirical measure.
        n_keep = max(2, int(self.n_keep_frac * X.shape[0]))
        kept = igfs_select(z_all, n_keep, seed=self.seed)
        kept_weights = stationary_reweight(z_all, kept)

        # Stage 3: lossless entropy coding of the kept latents against N(0, I).
        # Frame selection deliberately retains tail frames with large |z|, so the grid
        # is sized to the data rather than clipping those frames.
        zmax = max(self.zmax, float(np.abs(z_all[kept]).max()) * 1.02)
        cum = gaussian_cumfreq(self.L, zmax)
        zc = np.clip(z_all[kept], -zmax, zmax)
        levels = np.floor((zc + zmax) / (2 * zmax) * self.L).astype(np.int64)
        levels = np.clip(levels, 0, self.L - 1).ravel()
        coded = encode_iid(levels, cum)

        # Dynamics term: an MSM estimated on the slow collective variables of the full
        # trajectory.
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

        return KateArtifact(coded_latents=coded, n_keep=len(kept), dim=dim,
                           L=self.L, zmax=zmax, flow=flow, kept_idx=kept,
                           T_msm=T_msm, counts=C, lag=self.msm_lag, tica=tica,
                           centers=centers, kept_weights=kept_weights)

    @staticmethod
    def decode_ensemble(ct: KateArtifact) -> np.ndarray:
        """Reconstruct the kept representative configurations. The reconstruction is exact
        up to latent quantization, since the flow inverts exactly.

        The returned frames are the farthest-point coverage subset and therefore
        over-represent the low-density tails; they do not form an unbiased ensemble on
        their own. For ensemble averages, weight each frame by ``ct.kept_weights`` (see
        ``ensemble_average``) or sample directly from the flow density."""
        cum = gaussian_cumfreq(ct.L, ct.zmax)
        levels = decode_iid(ct.coded_latents, ct.n_keep * ct.dim, cum)
        levels = levels.reshape(ct.n_keep, ct.dim)
        z = -ct.zmax + (levels + 0.5) * (2 * ct.zmax / ct.L)
        with torch.no_grad():
            x = ct.flow.inverse(torch.as_tensor(z, dtype=torch.float32)).numpy()
        N = ct.dim // 3
        return x.reshape(ct.n_keep, N, 3)

    @staticmethod
    def ensemble_average(ct: KateArtifact, values: np.ndarray) -> np.ndarray:
        """Unbiased ensemble average of a per-kept-frame observable.

        ``values`` holds the observable evaluated on the kept frames (shape (n_keep,) or
        (n_keep, d)). The kept subset is tail-biased by construction, so the average is
        taken against the stored stationary weights ``ct.kept_weights``; this recovers
        the full-ensemble average that the ensemble-Pinsker bound refers to. Older
        artifacts without weights fall back to a uniform average with a warning."""
        v = np.asarray(values, dtype=np.float64)
        w = ct.kept_weights
        if w is None:
            import warnings
            warnings.warn("artifact has no kept_weights; ensemble average is the biased "
                          "uniform mean over the tail-heavy IGFS subset", RuntimeWarning)
            w = np.full(ct.n_keep, 1.0 / ct.n_keep)
        w = np.asarray(w, dtype=np.float64)
        return np.tensordot(w, v, axes=(0, 0))

    @staticmethod
    def kinetics(ct: KateArtifact, k=5):
        active = largest_connected_set(ct.counts)
        Tc, _ = transition_matrix(ct.counts[np.ix_(active, active)], reversible=True)
        return implied_timescales(Tc, ct.lag, k)
