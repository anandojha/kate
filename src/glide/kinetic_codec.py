"""
kinetic_codec.py
================
Analysis-native compression of MD trajectories.

The thesis: store a trajectory as its own validated kinetic model, so the
compressed object IS the analysis substrate, not an opaque blob you must
decompress first.

A trajectory becomes three things in one file:

  1. A linear "slow-mode" transform (TICA for kinetics; full-rank whitening for
     reconstruction). The whitening is an exactly-invertible *linear normalizing
     flow* to a Gaussian reference, so a KL bound survives it (Gaussian-reference
     version of the GLIDE guarantee). Swap in a Boltzmann-generator-style flow via
     the `Transform` interface to remove the Gaussian assumption.

  2. A discrete microstate sequence + its MSM transition matrix. The MSM is the
     entropy model: the state sequence is range-coded against the row-conditional
     transition probabilities, so its cost approaches the Markov-chain entropy
     rate  H = -sum_i pi_i sum_j T_ij log2 T_ij  (Ekroot & Cover). Metastable
     systems have a tiny entropy rate => the dynamics compress hard. The same
     transition matrix you coded against IS your kinetics (timescales, MFPT).

  3. A per-state continuous residual, quantized with a subtractive-dithered
     uniform scalar quantizer in the whitened space. Conditioning the residual
     on the microstate (per-state mean) shrinks it, so structure costs fewer bits.
     Dithering keeps reconstruction unbiased (preserves linear ensemble
     observables) and the reconstruction density continuous (so KL is finite and
     Pinsker applies).

NOVELTY / PRIOR-ART NOTE (be honest in any writeup):
  - MSM/Markov models as entropy coders are standard (textbook; ONTRAC for GPS
    trajectories; FSAR for learned lossless). NOT novel on its own.
  - Neural latent MD compression exists (MDZip, JCIM AE, pcazip). NOT novel.
  - Error-bounded MD compression exists (MDZ/SZ/ZFP; QoI-preserving variants).
    So "first error-bounded MD compressor" is FALSE.
  - The contribution here is the *unification*: one artifact that is the
    kinetic model, a near-entropy-optimal code for the dynamics, and a generative
    decoder, under a single *distributional* (KL->Pinsker) bound on arbitrary
    bounded observables (vs pointwise-coordinate or finite pre-chosen-QoI bounds).
    State even this as "to our knowledge" and check the QoI-compression papers.

Dependencies: numpy, scipy, scikit-learn. No torch required for this core.
I/O boundary: feed a list of (T_i, N, 3) float arrays (one per run). Wire your
OpenMM/DCD reader to produce that; the codec does not touch file formats.
Run-aware by construction: transition counts are tallied WITHIN runs, never
across concatenation seams.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# scikit-learn is only used for the clustering step.
from sklearn.cluster import MiniBatchKMeans


# ============================================================================
# 1. Markov range coder  (Witten-Neal-Cleary integer arithmetic coder)
# ============================================================================
# Exact, reversible. Encodes a sequence of states against per-step conditional
# distributions. For a first-order MSM there are only K distinct conditionals
# (one per previous state) plus an initial distribution, so the K integer
# cumulative tables are precomputed once and indexed by the previous state.

_PREC = 32
_WHOLE = 1 << _PREC          # 2^32
_HALF = 1 << (_PREC - 1)     # 2^31
_QUARTER = 1 << (_PREC - 2)  # 2^30
_3QUARTER = 3 << (_PREC - 2) # 3 * 2^30
_MASK = _WHOLE - 1
_FREQ_BITS = 16
_FREQ_TOTAL = 1 << _FREQ_BITS  # 65536


def _probs_to_cumfreq(probs: np.ndarray, total: int = _FREQ_TOTAL) -> np.ndarray:
    """Map a probability vector to an integer cumulative-frequency table of
    length K+1 (cum[0]=0, cum[K]=total). Every symbol gets freq >= 1 so any
    symbol is decodable (robust to transitions unseen at fit time). Deterministic,
    so encoder and decoder build identical tables from the same probs."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    K = p.size
    if s <= 0:
        freq = np.ones(K, dtype=np.int64)
    else:
        freq = np.floor(p / s * total).astype(np.int64)
        freq = np.maximum(freq, 1)
    # Fix the total to exactly `total` by adjusting the largest bins.
    diff = total - int(freq.sum())
    if diff != 0:
        order = np.argsort(-freq)  # touch largest bins first
        i = 0
        step = 1 if diff > 0 else -1
        remaining = abs(diff)
        while remaining > 0:
            idx = order[i % K]
            if step < 0 and freq[idx] <= 1:
                i += 1
                continue
            freq[idx] += step
            remaining -= 1
            i += 1
    cum = np.zeros(K + 1, dtype=np.int64)
    cum[1:] = np.cumsum(freq)
    return cum


class _BitWriter:
    def __init__(self):
        self.bits: List[int] = []

    def emit(self, bit: int, pending: int) -> int:
        self.bits.append(bit)
        for _ in range(pending):
            self.bits.append(1 - bit)
        return 0

    def to_bytes(self) -> bytes:
        b = self.bits
        pad = (-len(b)) % 8
        b = b + [0] * pad
        out = bytearray(len(b) // 8)
        for i in range(0, len(b), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | b[i + j]
            out[i // 8] = byte
        return bytes(out)


class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.nbits = len(data) * 8

    def next_bit(self) -> int:
        if self.pos >= self.nbits:
            self.pos += 1
            return 0  # zero-pad past end (WNC tolerates this)
        byte = self.data[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


def encode_markov(states: np.ndarray,
                  T: np.ndarray,
                  pi: np.ndarray) -> bytes:
    """Range-code an integer state sequence against transition matrix T,
    using pi for the first symbol. Returns the compressed byte string."""
    states = np.asarray(states, dtype=np.int64)
    K = T.shape[0]
    # Precompute K+1 cumulative tables: index 0 = initial (pi), 1..K = rows.
    cum_tables = [_probs_to_cumfreq(pi)]
    for i in range(K):
        cum_tables.append(_probs_to_cumfreq(T[i]))

    w = _BitWriter()
    low, high, pending = 0, _MASK, 0
    prev = -1  # -1 => use initial table
    for s in states:
        cum = cum_tables[0] if prev < 0 else cum_tables[prev + 1]
        c_low = int(cum[s])
        c_high = int(cum[s + 1])
        rng = high - low + 1
        high = low + (rng * c_high) // _FREQ_TOTAL - 1
        low = low + (rng * c_low) // _FREQ_TOTAL
        while True:
            if high < _HALF:
                pending = w.emit(0, pending)
            elif low >= _HALF:
                pending = w.emit(1, pending)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _3QUARTER:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) | 1) & _MASK
        prev = int(s)
    # flush
    pending += 1
    if low < _QUARTER:
        w.emit(0, pending)
    else:
        w.emit(1, pending)
    return w.to_bytes()


def decode_markov(data: bytes,
                  n: int,
                  T: np.ndarray,
                  pi: np.ndarray) -> np.ndarray:
    """Inverse of encode_markov. Decodes exactly n states."""
    K = T.shape[0]
    cum_tables = [_probs_to_cumfreq(pi)]
    for i in range(K):
        cum_tables.append(_probs_to_cumfreq(T[i]))

    r = _BitReader(data)
    low, high = 0, _MASK
    code = 0
    for _ in range(_PREC):
        code = ((code << 1) | r.next_bit()) & _MASK

    out = np.empty(n, dtype=np.int64)
    prev = -1
    for t in range(n):
        cum = cum_tables[0] if prev < 0 else cum_tables[prev + 1]
        rng = high - low + 1
        value = (((code - low) + 1) * _FREQ_TOTAL - 1) // rng
        # locate symbol s with cum[s] <= value < cum[s+1]
        s = int(np.searchsorted(cum, value, side='right') - 1)
        if s < 0:
            s = 0
        elif s >= K:
            s = K - 1
        c_low = int(cum[s])
        c_high = int(cum[s + 1])
        high = low + (rng * c_high) // _FREQ_TOTAL - 1
        low = low + (rng * c_low) // _FREQ_TOTAL
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                code -= _HALF
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _3QUARTER:
                code -= _QUARTER
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) | 1) & _MASK
            code = ((code << 1) | r.next_bit()) & _MASK
        out[t] = s
        prev = s
    return out


# ============================================================================
# 2. Rigid alignment (Kabsch)
# ============================================================================

def kabsch_align(frames: np.ndarray, ref: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Superpose each frame onto a reference by optimal rotation+translation.
    frames: (T, N, 3). Returns (aligned (T,N,3), reference (N,3)).
    Compression is on the aligned (internal) configuration, which is also what
    structural/kinetic analysis uses."""
    X = np.asarray(frames, dtype=np.float64)
    if ref is None:
        ref = X[0]
    ref = ref - ref.mean(axis=0, keepdims=True)
    out = np.empty_like(X)
    for t in range(X.shape[0]):
        P = X[t] - X[t].mean(axis=0, keepdims=True)
        H = P.T @ ref
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        out[t] = P @ R.T
    return out, ref


# ============================================================================
# 3. Transform interface + linear whitening + TICA
# ============================================================================

class Transform:
    """Invertible map Y <-> Z. Linear whitening below is the tested default;
    a normalizing flow drops in by implementing forward/inverse with the same
    signatures (forward = x->base z, inverse = z->x)."""
    def fit(self, Y: np.ndarray) -> "Transform": ...
    def forward(self, Y: np.ndarray) -> np.ndarray: ...
    def inverse(self, Z: np.ndarray) -> np.ndarray: ...


@dataclass
class WhiteningTransform(Transform):
    """Full-rank PCA/Mahalanobis whitening: exactly invertible linear map to a
    standardized Gaussian reference. This is a linear normalizing flow; the KL
    bound is preserved under it (Gaussian-reference GLIDE). For large 3N keep it
    low-rank (set `rank`), but then reconstruction is lossy beyond quantization."""
    rank: Optional[int] = None
    mean_: np.ndarray = field(default=None, repr=False)
    W_: np.ndarray = field(default=None, repr=False)      # forward (whiten)
    Winv_: np.ndarray = field(default=None, repr=False)   # inverse (color)

    def fit(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        self.mean_ = Y.mean(axis=0)
        Yc = Y - self.mean_
        cov = np.cov(Yc, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += 1e-9 * np.eye(cov.shape[0])
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        if self.rank is not None:
            evals, evecs = evals[:self.rank], evecs[:, :self.rank]
        s = np.sqrt(np.maximum(evals, 1e-12))
        self.W_ = evecs / s                 # (D, k): whiten
        self.Winv_ = (evecs * s).T          # (k, D): color
        return self

    def forward(self, Y):
        return (np.asarray(Y, dtype=np.float64) - self.mean_) @ self.W_

    def inverse(self, Z):
        return np.asarray(Z, dtype=np.float64) @ self.Winv_ + self.mean_


@dataclass
class TICA:
    """Time-lagged independent component analysis: leading slow collective
    variables via the generalized eigenproblem C(tau) v = C(0) v l. Used only to
    pick the discretization features (the kinetics live in the slow modes)."""
    lag: int = 1
    n_components: Optional[int] = None
    mean_: np.ndarray = field(default=None, repr=False)
    eigvecs_: np.ndarray = field(default=None, repr=False)
    timescales_: np.ndarray = field(default=None, repr=False)

    def _solve(self, c0, ct):
        """Solve the generalized eigenproblem C(tau) v = C(0) v l and store the
        leading slow modes. Shared by fit() and finalize()."""
        from scipy.linalg import eigh
        evals, evecs = eigh(ct, c0)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        k = self.n_components or evecs.shape[1]
        evecs = evecs[:, :k]
        # Canonicalize eigenvector SIGNS (they are otherwise arbitrary): make the
        # largest-magnitude component of each mode positive. This makes the CVs --
        # and therefore the seeded flow / IGFS selection -- DETERMINISTIC and identical
        # between the batch fit() and the streaming finalize() (whose covariances differ
        # only by float roundoff), so streaming GLIDE reproduces the in-RAM result.
        cols = np.arange(evecs.shape[1])
        signs = np.sign(evecs[np.argmax(np.abs(evecs), axis=0), cols])
        signs[signs == 0] = 1.0
        self.eigvecs_ = evecs * signs
        with np.errstate(divide='ignore', invalid='ignore'):
            self.timescales_ = -self.lag / np.log(np.clip(evals[:k], 1e-12, 0.999999))
        return self

    def fit(self, runs: List[np.ndarray]):
        feats = [np.asarray(r, dtype=np.float64) for r in runs]
        allf = np.concatenate(feats, axis=0)
        self.mean_ = allf.mean(axis=0)
        c0 = np.zeros((allf.shape[1],) * 2)
        ct = np.zeros_like(c0)
        n0 = nt = 0
        for f in feats:
            fc = f - self.mean_
            c0 += fc.T @ fc
            n0 += fc.shape[0]
            if fc.shape[0] > self.lag:
                a = fc[:-self.lag]
                b = fc[self.lag:]
                ct += a.T @ b
                nt += a.shape[0]
        c0 /= max(n0, 1)
        ct /= max(nt, 1)
        ct = 0.5 * (ct + ct.T)              # symmetrize (reversibility)
        c0 += 1e-9 * np.eye(c0.shape[0])
        return self._solve(c0, ct)

    # ----- streaming / out-of-core fitting (T5) -----
    # Accumulate the SAME C(0), C(tau), mean as fit(), but chunk by chunk, so the CV
    # step scales past RAM (md.iterload). Cross-chunk lagged pairs are preserved by
    # carrying the last `lag` frames across calls, so streaming == batch EXACTLY for a
    # continuous run. Pass run_start=True on the first chunk of each run (resets the
    # lag buffer so no pair spans a run seam).
    def partial_fit(self, Y, run_start=False):
        Y = np.asarray(Y, dtype=np.float64)
        if not hasattr(self, "_n0"):
            D = Y.shape[1]
            self._S0 = np.zeros((D, D)); self._sum0 = np.zeros(D); self._n0 = 0
            self._St = np.zeros((D, D)); self._suma = np.zeros(D)
            self._sumb = np.zeros(D); self._nt = 0; self._tail = None
        self._S0 += Y.T @ Y
        self._sum0 += Y.sum(axis=0)
        self._n0 += Y.shape[0]
        if run_start:
            self._tail = None
        ext = Y if self._tail is None else np.concatenate([self._tail, Y], axis=0)
        if ext.shape[0] > self.lag:
            a = ext[:-self.lag]; b = ext[self.lag:]
            self._St += a.T @ b
            self._suma += a.sum(axis=0); self._sumb += b.sum(axis=0)
            self._nt += a.shape[0]
        self._tail = ext[-self.lag:]
        return self

    def finalize(self):
        """Solve TICA from the streamed moments. Equivalent to fit() on the same data."""
        n0 = max(self._n0, 1)
        mean = self._sum0 / n0
        c0 = self._S0 / n0 - np.outer(mean, mean)
        nt = max(self._nt, 1)
        ct = (self._St / nt - np.outer(self._suma / nt, mean)
              - np.outer(mean, self._sumb / nt) + np.outer(mean, mean))
        ct = 0.5 * (ct + ct.T)
        c0 = c0 + 1e-9 * np.eye(c0.shape[0])
        self.mean_ = mean
        # free the heavy accumulators once solved
        out = self._solve(c0, ct)
        for attr in ("_S0", "_St"):
            setattr(self, attr, None)
        return out

    def transform(self, Y):
        return (np.asarray(Y, dtype=np.float64) - self.mean_) @ self.eigvecs_


# ============================================================================
# 4. Discretization (run-aware) + MSM
# ============================================================================

def discretize(runs_feat: List[np.ndarray], n_states: int, seed: int = 0
               ) -> Tuple[List[np.ndarray], np.ndarray]:
    """k-means microstates. Returns (per-run integer label arrays, centers)."""
    allf = np.concatenate(runs_feat, axis=0)
    km = MiniBatchKMeans(n_clusters=n_states, random_state=seed,
                         n_init=3, batch_size=max(256, 3 * n_states))
    km.fit(allf)
    labels = [km.predict(f).astype(np.int64) for f in runs_feat]
    return labels, km.cluster_centers_


def count_matrix(labels: List[np.ndarray], n_states: int, lag: int = 1
                 ) -> np.ndarray:
    """Transition counts at lag tau, tallied WITHIN each run (no cross-seam
    transitions). This is the run-aggregation discipline."""
    C = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in labels:
        if len(seq) > lag:
            a = seq[:-lag]
            b = seq[lag:]
            np.add.at(C, (a, b), 1.0)
    return C


def transition_matrix(C: np.ndarray, reversible: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Transition matrix from counts. `reversible` enforces detailed balance by the
    standard symmetrization C <- (C + C^T)/2 -- a simple, robust, *reversible* estimator,
    but NOT the maximum-likelihood one (it biases the stationary distribution toward
    uniform when state populations are unequal). Kept as the pure-numpy fallback /
    entropy-coding model; for REPORTED kinetics use `estimate_reversible_T`, which prefers
    deeptime's reversible MLE. Returns (T, stationary pi)."""
    C = C.copy()
    if reversible:
        C = 0.5 * (C + C.T)
    rs = C.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    T = C / rs
    pi = stationary_distribution(T)
    return T, pi


def estimate_reversible_T(C: np.ndarray, prefer: str = "auto"
                          ) -> Tuple[np.ndarray, str]:
    """Reversible transition matrix from counts, PREFERRING deeptime's reversible
    maximum-likelihood estimator (the publishable one) and FALLING BACK to the
    (C+C^T)/2 symmetrization when deeptime is unavailable or its MLE fails. This is the
    estimator that backs GLIDE's REPORTED timescales and the path bound.

    Returns (T, estimator_tag). T has the SAME shape as C; any state deeptime drops from
    the largest connected set becomes an absorbing self-loop (T_ii=1) -- honest, since a
    state the chain never leaves IS absorbing, and the path bound's support check flags
    the resulting structural zeros. estimator_tag is 'deeptime-mle' or 'symmetrized-cc'.
    prefer='cc' forces the symmetrized estimator; 'mle' forces deeptime (raising if
    unavailable)."""
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    if prefer in ("auto", "mle"):
        try:
            from .kinetics_deeptime import reversible_mle_from_counts
            T_act, active = reversible_mle_from_counts(C, reversible=True)
            T = np.eye(n)
            T[np.ix_(active, active)] = T_act
            return T, "deeptime-mle"
        except Exception:
            if prefer == "mle":
                raise
    T, _ = transition_matrix(C, reversible=True)
    return T, "symmetrized-cc"


def largest_connected_set(C: np.ndarray) -> np.ndarray:
    """Indices of the largest strongly-connected (ergodic) set of microstates,
    by total counts. Standard MSM hygiene: implied timescales are only meaningful
    on a single communicating class; peripheral/absorbing k-means microstates
    otherwise inject spurious unit eigenvalues (infinite timescales)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    A = (np.asarray(C) > 0).astype(np.int8)
    n_comp, labels = connected_components(csr_matrix(A), directed=True,
                                          connection='strong')
    best, best_mass = np.array([0]), -1.0
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        mass = float(C[np.ix_(idx, idx)].sum())
        if mass > best_mass:
            best_mass, best = mass, idx
    return np.sort(best)


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eig(T.T)
    i = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real(evecs[:, i])
    pi = np.abs(pi)
    s = pi.sum()
    return pi / s if s > 0 else np.ones(T.shape[0]) / T.shape[0]


def implied_timescales(T: np.ndarray, lag: int = 1, k: int = 5) -> np.ndarray:
    evals = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
    evals = np.clip(evals[1:k + 1], 1e-12, 0.999999)  # skip stationary eval=1
    return -lag / np.log(evals)


def entropy_rate(T: np.ndarray, pi: np.ndarray) -> float:
    """Markov-chain entropy rate in BITS/step: H = -sum_i pi_i sum_j T_ij log2 T_ij
    (Ekroot & Cover). This is the information-theoretic floor for coding the
    state sequence; the range coder should approach it."""
    with np.errstate(divide='ignore', invalid='ignore'):
        logT = np.where(T > 0, np.log2(T), 0.0)
    return float(-(pi[:, None] * T * logT).sum())


# ============================================================================
# 5. Subtractive-dithered uniform scalar quantizer (per-state)
# ============================================================================

@dataclass
class DitheredResidualCodec:
    """Quantizes whitened residuals with a fixed step. Subtractive dither makes
    reconstruction unbiased (linear ensemble observables preserved) and the
    reconstruction density continuous (KL finite -> Pinsker applies).
    Per-state mean subtraction shrinks the residual => fewer bits for structure."""
    n_bits: int = 4
    seed: int = 0

    def _dither(self, shape, step):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(-0.5 * step, 0.5 * step, size=shape)

    def quantize(self, Z: np.ndarray, step: float):
        """Returns integer levels (same shape as Z). Dither is reproducible from
        seed, so the decoder regenerates it identically and subtracts it back."""
        d = self._dither(Z.shape, step)
        q = np.round((Z + d) / step).astype(np.int64)
        return q

    def dequantize(self, q: np.ndarray, step: float):
        d = self._dither(q.shape, step)
        return q.astype(np.float64) * step - d


# ============================================================================
# 6. End-to-end codec
# ============================================================================

@dataclass
class CompressedTrajectory:
    """The single artifact. Everything needed to (a) do kinetics directly from
    `T`/`pi`, (b) reconstruct coordinates, (c) reproduce bounded observables."""
    run_lengths: List[int]
    coded_states: List[bytes]      # one range-coded blob per run
    quant_residuals: np.ndarray    # (T_total, d) integer levels
    step: float
    n_states: int
    T: np.ndarray                  # the MSM = the entropy model = your kinetics
    pi: np.ndarray
    state_means: np.ndarray        # (K, d) per-state mean in whitened space
    whitener: WhiteningTransform
    centers: np.ndarray            # k-means centers (TICA space) for relabeling
    tica: TICA
    ref: np.ndarray                # alignment reference
    lag: int
    reconstruct_dim: int
    n_bits: int                    # residual quantizer bit depth
    seed: int                      # dither seed (must match for exact decode)
    counts: np.ndarray = None      # raw count matrix (for connectivity, bootstrap)

    # ----- analysis straight off the compressed object, no decompression -----
    def kinetics(self, k: int = 5):
        """Kinetics restricted to the largest ergodic set (correct MSM practice).
        The full T is kept for decoding; timescales are computed on the
        communicating class so peripheral microstates don't fake infinite ones."""
        C = self.counts if self.counts is not None else self.T
        active = largest_connected_set(C)
        Tc, pic = transition_matrix(C[np.ix_(active, active)], reversible=True)
        return {
            "active_states": active,
            "stationary_distribution": pic,
            "implied_timescales": implied_timescales(Tc, self.lag, k),
            "entropy_rate_bits": entropy_rate(self.T, self.pi),
        }


class KineticCodec:
    def __init__(self, tica_lag: int = 1, tica_dim: int = 2,
                 n_states: int = 100, msm_lag: int = 1,
                 n_bits: int = 4, reversible: bool = True, seed: int = 0):
        self.tica_lag = tica_lag
        self.tica_dim = tica_dim
        self.n_states = n_states
        self.msm_lag = msm_lag
        self.n_bits = n_bits
        self.reversible = reversible
        self.seed = seed

    def fit_encode(self, runs: List[np.ndarray]) -> CompressedTrajectory:
        """runs: list of (T_i, N, 3) arrays (already comparable; pre-align if
        across separate simulations). Returns the compressed artifact."""
        # ----- align + flatten -----
        ref = None
        aligned = []
        for r in runs:
            a, ref = kabsch_align(np.asarray(r, dtype=np.float64), ref)
            aligned.append(a.reshape(a.shape[0], -1))   # (T_i, 3N)
        run_lengths = [a.shape[0] for a in aligned]

        # ----- whitening (reconstruction transform; linear flow) -----
        wh = WhiteningTransform().fit(np.concatenate(aligned, axis=0))
        white_runs = [wh.forward(a) for a in aligned]   # (T_i, d), d = 3N here
        d = white_runs[0].shape[1]

        # ----- TICA slow modes (discretization features) -----
        tica = TICA(lag=self.tica_lag, n_components=self.tica_dim)
        tica.fit(white_runs)
        tica_runs = [tica.transform(w) for w in white_runs]

        # ----- microstates (run-aware) -----
        labels, centers = discretize(tica_runs, self.n_states, self.seed)

        # ----- MSM = entropy model = kinetics -----
        C = count_matrix(labels, self.n_states, self.msm_lag)
        T, pi = transition_matrix(C, self.reversible)

        # ----- per-state means in whitened space (structural prior) -----
        all_white = np.concatenate(white_runs, axis=0)
        all_lab = np.concatenate(labels, axis=0)
        state_means = np.zeros((self.n_states, d))
        for s in range(self.n_states):
            m = all_lab == s
            if m.any():
                state_means[s] = all_white[m].mean(axis=0)

        # ----- residuals + dithered quantization -----
        residual = all_white - state_means[all_lab]
        step = (residual.std() + 1e-12) * (2.0 ** (1 - self.n_bits)) * 3.0
        codec = DitheredResidualCodec(n_bits=self.n_bits, seed=self.seed)
        q = codec.quantize(residual, step)

        # ----- range-code each run's state sequence against T (with pi init) ---
        coded = [encode_markov(seq, T, pi) for seq in labels]

        return CompressedTrajectory(
            run_lengths=run_lengths, coded_states=coded, quant_residuals=q,
            step=step, n_states=self.n_states, T=T, pi=pi,
            state_means=state_means, whitener=wh, centers=centers, tica=tica,
            ref=ref, lag=self.msm_lag, reconstruct_dim=d,
            n_bits=self.n_bits, seed=self.seed, counts=C,
        )

    @staticmethod
    def decode(ct: CompressedTrajectory) -> List[np.ndarray]:
        """Reconstruct aligned coordinates per run: decode states -> add per-state
        mean -> add dequantized residual -> inverse-whiten -> reshape (T,N,3)."""
        # decode state sequences
        labels = []
        for blob, n in zip(ct.coded_states, ct.run_lengths):
            labels.append(decode_markov(blob, n, ct.T, ct.pi))
        all_lab = np.concatenate(labels, axis=0)
        codec = DitheredResidualCodec(n_bits=ct.n_bits, seed=ct.seed)
        residual = codec.dequantize(ct.quant_residuals, ct.step)
        white = ct.state_means[all_lab] + residual
        coords_flat = ct.whitener.inverse(white)
        N3 = coords_flat.shape[1]
        N = N3 // 3
        out, off = [], 0
        for n in ct.run_lengths:
            out.append(coords_flat[off:off + n].reshape(n, N, 3))
            off += n
        return out

    # --------------------------------------------------------------------- #
    @staticmethod
    def report(ct: CompressedTrajectory) -> dict:
        """Bit accounting. Residuals are charged at the fixed quantizer rate
        (n_bits/value), an honest upper bound; entropy-coding them would only
        lower this. Side info is the one-time model cost, amortized over T."""
        T_total = sum(ct.run_lengths)
        N3 = ct.reconstruct_dim
        state_bits = 8 * sum(len(b) for b in ct.coded_states)
        residual_bits = T_total * N3 * ct.n_bits
        stream_bits = state_bits + residual_bits
        # one-time side info (amortized over T)
        side_bits = (
            ct.T.size * 16                       # transition matrix @16-bit
            + ct.state_means.size * 32           # per-state means @32-bit
            + ct.whitener.W_.size * 32           # whitening matrix @32-bit
            + ct.centers.size * 32               # k-means centers
        )
        orig_bits = T_total * N3 * 32            # DCD = 32-bit float coords
        H = entropy_rate(ct.T, ct.pi)
        return {
            "frames": T_total,
            "coords_per_frame": N3,
            "state_bits_per_frame": state_bits / T_total,
            "msm_entropy_rate_bits_per_frame": H,
            "residual_bits_per_frame": residual_bits / T_total,
            "stream_bits_per_coord": stream_bits / (T_total * N3),
            "ratio_vs_float32_stream_only": orig_bits / stream_bits,
            "side_info_bytes": side_bits / 8,
            "side_info_bits_per_frame_amortized": side_bits / T_total,
        }
