"""
Analysis-native compression of a molecular-dynamics trajectory stored as its own
validated kinetic model.

A run is kept as three coupled parts, each of which is also a piece of the kinetic
model, so the compressed object is the analysis substrate itself and is read without a
decompression step.

The configuration is first carried through an exactly invertible linear map. TICA
supplies the slow collective variables that drive the kinetics, and full-rank PCA
whitening (x - mu) V L^{-1/2}, where V and L are the eigenvectors and eigenvalues of
the coordinate covariance, standardizes the coordinates onto a unit Gaussian reference.
Whitening is a linear normalizing flow, and the Kullback-Leibler divergence is
invariant under an invertible map, so the KATE path-space bound carries over in its
Gaussian-reference form without assuming the CV data are Gaussian. A Boltzmann-generator
flow may be supplied through the Transform interface where the Gaussian reference is too
coarse.

The discrete microstate sequence is range-coded against its Markov state model. The MSM
transition matrix T_ij is the entropy model: coding each state against its
row-conditional probabilities drives the bit cost toward the Markov-chain entropy rate

    H = -sum_i pi_i sum_j T_ij log2(T_ij)    [bits/step],

where pi_i is the stationary weight of microstate i (Ekroot & Cover, IEEE Trans. Inf.
Theory 39, 1418 (1993)). Metastable dynamics carry a small H, so the sequence compresses
strongly, and the same T_ij coded against is the kinetic model itself, carrying the
implied timescales and mean first-passage times.

The per-state continuous residual is quantized in the whitened space by a
subtractive-dithered uniform scalar quantizer. Subtracting the per-state mean shrinks
the residual, so structural correlation costs fewer bits. Subtractive dither leaves the
reconstruction unbiased, preserving linear ensemble observables, and keeps the
reconstruction density continuous, so the KL divergence stays finite and the Pinsker
inequality applies to any bounded observable.

The core needs only numpy, scipy, and scikit-learn. The input is a list of (T_i, N, 3)
float arrays, one per run; file formats are a reader's concern, not the codec's, and
transition counts are tallied within a run and never across a concatenation seam.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# scikit-learn is required only for the clustering step.
from sklearn.cluster import MiniBatchKMeans


# The Markov range coder, an integer arithmetic coder in the Witten-Neal-Cleary
# form, exact and reversible. It codes a state sequence against per-step
# conditional distributions. A first-order MSM has only K distinct conditionals,
# one per previous state, plus an initial distribution, so the K integer
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
    """Map a probability vector to an integer cumulative-frequency table.

    The returned table has length K+1 with cum[0]=0 and cum[K]=total. Each symbol
    is assigned a frequency of at least one, so that any symbol remains decodable;
    this provides robustness to transitions unseen at fit time. The construction is
    deterministic, so the encoder and decoder build identical tables from the same
    probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    K = p.size
    if s <= 0:
        freq = np.ones(K, dtype=np.int64)
    else:
        freq = np.floor(p / s * total).astype(np.int64)
        freq = np.maximum(freq, 1)
    # Correct the sum to exactly `total` by adjusting the largest bins, which
    # minimizes the relative perturbation of the frequency table.
    diff = total - int(freq.sum())
    if diff != 0:
        order = np.argsort(-freq)  # adjust the largest bins first
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
            return 0  # zero-pad past the end, which the WNC coder tolerates
        byte = self.data[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


def encode_markov(states: np.ndarray,
                  T: np.ndarray,
                  pi: np.ndarray) -> bytes:
    """Range-code an integer state sequence against a transition matrix.

    The sequence is coded against the row-conditional probabilities of T, with the
    stationary or initial distribution pi used for the first symbol.

    Parameters
    ----------
    states : np.ndarray
        Integer microstate sequence to encode.
    T : np.ndarray
        Row-stochastic transition matrix of shape (K, K).
    pi : np.ndarray
        Initial-symbol distribution of length K.

    Returns
    -------
    bytes
        The compressed byte string.
    """
    states = np.asarray(states, dtype=np.int64)
    K = T.shape[0]
    # Precompute K+1 cumulative tables: index 0 holds the initial distribution
    # (pi) and indices 1..K hold the transition-matrix rows.
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
    # Flush the remaining interval.
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
    """Decode exactly n states, inverting encode_markov.

    Parameters
    ----------
    data : bytes
        Compressed byte string produced by encode_markov.
    n : int
        Number of states to decode.
    T : np.ndarray
        Row-stochastic transition matrix of shape (K, K).
    pi : np.ndarray
        Initial-symbol distribution of length K.

    Returns
    -------
    np.ndarray
        The decoded integer state sequence of length n.
    """
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
        # Locate the symbol s satisfying cum[s] <= value < cum[s+1].
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


def kabsch_align(frames: np.ndarray, ref: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Superpose each frame onto a reference by optimal rotation and translation.

    The rotation is obtained from the Kabsch solution to the orthogonal Procrustes
    problem. Compression is performed on the aligned (internal) configuration,
    which is also the configuration used by structural and kinetic analysis.

    Parameters
    ----------
    frames : np.ndarray
        Coordinates of shape (T, N, 3).
    ref : np.ndarray, optional
        Reference configuration of shape (N, 3); the first frame is used when
        none is supplied.

    Returns
    -------
    tuple of np.ndarray
        The aligned coordinates of shape (T, N, 3) and the reference of shape
        (N, 3).
    """
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


class Transform:
    """Invertible map between configuration space Y and latent space Z.

    Linear whitening (below) is the default implementation. A normalizing flow may
    be substituted by implementing forward and inverse with the same signatures,
    where forward maps configurations to the base variable and inverse maps the
    base variable back to configurations."""
    def fit(self, Y: np.ndarray) -> "Transform": ...
    def forward(self, Y: np.ndarray) -> np.ndarray: ...
    def inverse(self, Z: np.ndarray) -> np.ndarray: ...


@dataclass
class WhiteningTransform(Transform):
    """Full-rank PCA/Mahalanobis whitening to a standardized Gaussian reference.

    The map is an exactly invertible linear transform, equivalent to a linear
    normalizing flow, under which the KL bound is preserved (the Gaussian-reference
    form of KATE). For large 3N a low-rank variant may be selected by setting
    `rank`, in which case reconstruction becomes lossy beyond quantization."""
    rank: Optional[int] = None
    mean_: np.ndarray = field(default=None, repr=False)
    W_: np.ndarray = field(default=None, repr=False)      # forward (whitening) map
    Winv_: np.ndarray = field(default=None, repr=False)   # inverse (coloring) map

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
        self.W_ = evecs / s                 # (D, k): whitening map
        self.Winv_ = (evecs * s).T          # (k, D): coloring map
        return self

    def forward(self, Y):
        return (np.asarray(Y, dtype=np.float64) - self.mean_) @ self.W_

    def inverse(self, Z):
        return np.asarray(Z, dtype=np.float64) @ self.Winv_ + self.mean_


@dataclass
class TICA:
    """Time-lagged independent component analysis (Perez-Hernandez et al.,
    J. Chem. Phys. 139, 015102, 2013).

    The leading slow collective variables are obtained from the generalized
    eigenproblem

        C(tau) v = C(0) v l,

    where C(0) and C(tau) are the instantaneous and time-lagged covariance
    matrices. TICA is used only to select the discretization features, since the
    slow kinetics are concentrated in these modes."""
    lag: int = 1
    n_components: Optional[int] = None
    mean_: np.ndarray = field(default=None, repr=False)
    eigvecs_: np.ndarray = field(default=None, repr=False)
    timescales_: np.ndarray = field(default=None, repr=False)

    def _solve(self, c0, ct):
        """Solve the generalized eigenproblem and store the leading slow modes.

        The eigenproblem C(tau) v = C(0) v l is solved and the leading modes are
        retained. This routine is shared by fit() and finalize()."""
        from scipy.linalg import eigh
        evals, evecs = eigh(ct, c0)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        k = self.n_components or evecs.shape[1]
        evecs = evecs[:, :k]
        # Canonicalize the otherwise arbitrary eigenvector signs by forcing the
        # largest-magnitude component of each mode to be positive. This renders the
        # collective variables, and therefore the seeded flow and IGFS selection,
        # deterministic and identical between the batch fit() and the streaming
        # finalize(), whose covariances differ only by floating-point roundoff. The
        # streaming KATE pipeline therefore reproduces the in-memory result.
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
        ct = 0.5 * (ct + ct.T)              # symmetrize to enforce reversibility
        c0 += 1e-9 * np.eye(c0.shape[0])
        return self._solve(c0, ct)

    # The same C(0), C(tau), and mean as fit() are accumulated chunk by chunk, so
    # that the collective-variable step scales beyond available memory (e.g. via
    # md.iterload). Cross-chunk lagged pairs are preserved by carrying the last
    # `lag` frames across calls, so streaming reproduces the batch result exactly
    # for a continuous run. The caller passes run_start=True on the first chunk of
    # each run, which resets the lag buffer so that no lagged pair spans a run seam.
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
        """Solve TICA from the streamed moments.

        The result is equivalent to fit() applied to the same data."""
        n0 = max(self._n0, 1)
        mean = self._sum0 / n0
        c0 = self._S0 / n0 - np.outer(mean, mean)
        nt = max(self._nt, 1)
        ct = (self._St / nt - np.outer(self._suma / nt, mean)
              - np.outer(mean, self._sumb / nt) + np.outer(mean, mean))
        ct = 0.5 * (ct + ct.T)
        c0 = c0 + 1e-9 * np.eye(c0.shape[0])
        self.mean_ = mean
        # Release the heavy moment accumulators once the modes are solved.
        out = self._solve(c0, ct)
        for attr in ("_S0", "_St"):
            setattr(self, attr, None)
        return out

    def transform(self, Y):
        return (np.asarray(Y, dtype=np.float64) - self.mean_) @ self.eigvecs_


def discretize(runs_feat: List[np.ndarray], n_states: int, seed: int = 0
               ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Assign k-means microstates to the feature trajectories.

    Parameters
    ----------
    runs_feat : list of np.ndarray
        Per-run feature arrays.
    n_states : int
        Number of microstates (cluster centers).
    seed : int
        Random seed for the clustering.

    Returns
    -------
    tuple
        A list of per-run integer label arrays and the array of cluster centers.
    """
    allf = np.concatenate(runs_feat, axis=0)
    km = MiniBatchKMeans(n_clusters=n_states, random_state=seed,
                         n_init=3, batch_size=max(256, 3 * n_states))
    km.fit(allf)
    labels = [km.predict(f).astype(np.int64) for f in runs_feat]
    return labels, km.cluster_centers_


def count_matrix(labels: List[np.ndarray], n_states: int, lag: int = 1
                 ) -> np.ndarray:
    """Accumulate transition counts at lag tau, tallied within each run.

    Counts are accumulated separately for each run so that no transition spans a
    concatenation seam, which is the run-aggregation discipline required for
    correct estimation from multiple trajectories."""
    C = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in labels:
        if len(seq) > lag:
            a = seq[:-lag]
            b = seq[lag:]
            np.add.at(C, (a, b), 1.0)
    return C


def transition_matrix(C: np.ndarray, reversible: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a transition matrix from a count matrix.

    When `reversible` is set, detailed balance is enforced by the standard
    symmetrization C <- (C + C^T)/2. This estimator is simple, robust, and
    reversible, but it is not the maximum-likelihood estimator: it biases the
    stationary distribution toward uniform when state populations are unequal. It
    is retained here as the pure-numpy fallback and as the entropy-coding model.
    For reported kinetics, `estimate_reversible_T` should be used instead, since it
    prefers the deeptime reversible maximum-likelihood estimator.

    Returns
    -------
    tuple of np.ndarray
        The transition matrix T and its stationary distribution pi.
    """
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
    """Estimate a reversible transition matrix from a count matrix.

    The deeptime reversible maximum-likelihood estimator is preferred, with a
    fallback to the (C + C^T)/2 symmetrization when deeptime is unavailable or its
    MLE fails. This is the estimator that backs the reported KATE timescales and
    the path bound.

    The returned matrix has the same shape as C. Any state that deeptime drops from
    the largest connected set is represented as an absorbing self-loop (T_ii = 1).
    This is faithful, since a state the chain never leaves is indeed absorbing, and
    the support check of the path bound flags the resulting structural zeros.

    Parameters
    ----------
    C : np.ndarray
        Transition count matrix.
    prefer : str
        Estimator selection. 'auto' prefers deeptime with fallback; 'mle' forces
        the deeptime estimator and raises if it is unavailable; 'cc' forces the
        symmetrized count estimator.

    Returns
    -------
    tuple
        The transition matrix and an estimator tag, either 'deeptime-mle' or
        'symmetrized-cc'.
    """
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
    """Return the indices of the largest strongly connected set of microstates.

    The largest strongly connected (ergodic) set is selected by total count mass.
    This is standard MSM practice: implied timescales are meaningful only on a
    single communicating class, since peripheral or absorbing k-means microstates
    otherwise introduce spurious unit eigenvalues and hence infinite timescales."""
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
    evals = np.clip(evals[1:k + 1], 1e-12, 0.999999)  # skip the stationary eigenvalue 1
    return -lag / np.log(evals)


def entropy_rate(T: np.ndarray, pi: np.ndarray) -> float:
    """Compute the Markov-chain entropy rate.

    The entropy rate is

        H = -sum_i pi_i sum_j T_ij log2(T_ij)    [bits/step]

    (Ekroot & Cover, IEEE Trans. Inf. Theory 39, 1418, 1993). It is the
    information-theoretic lower bound for coding the state sequence, which the range
    coder approaches."""
    with np.errstate(divide='ignore', invalid='ignore'):
        logT = np.where(T > 0, np.log2(T), 0.0)
    return float(-(pi[:, None] * T * logT).sum())


@dataclass
class DitheredResidualCodec:
    """Quantize whitened residuals with a fixed step and subtractive dither.

    Subtractive dither makes the reconstruction unbiased, preserving linear
    ensemble observables, and keeps the reconstruction density continuous, so that
    the KL divergence is finite and the Pinsker inequality applies. Per-state mean
    subtraction reduces the residual magnitude, lowering the bit cost of structural
    correlations."""
    n_bits: int = 4
    seed: int = 0

    def _dither(self, shape, step):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(-0.5 * step, 0.5 * step, size=shape)

    def quantize(self, Z: np.ndarray, step: float):
        """Return integer quantization levels of the same shape as Z.

        The dither is reproducible from the seed, so the decoder regenerates it
        identically and subtracts it during dequantization."""
        d = self._dither(Z.shape, step)
        q = np.round((Z + d) / step).astype(np.int64)
        return q

    def dequantize(self, q: np.ndarray, step: float):
        d = self._dither(q.shape, step)
        return q.astype(np.float64) * step - d


@dataclass
class CompressedTrajectory:
    """The single compressed artifact.

    It carries everything required to (a) perform kinetics directly from `T` and
    `pi`, (b) reconstruct coordinates, and (c) reproduce bounded observables."""
    run_lengths: List[int]
    coded_states: List[bytes]      # one range-coded blob per run
    quant_residuals: np.ndarray    # (T_total, d) integer quantization levels
    step: float
    n_states: int
    T: np.ndarray                  # MSM transition matrix: entropy model and kinetics
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

    def kinetics(self, k: int = 5):
        """Compute kinetics on the largest ergodic set.

        The full transition matrix T is retained for decoding, while timescales are
        computed on the communicating class so that peripheral microstates do not
        produce spurious infinite timescales."""
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
        """Fit the codec to a set of runs and encode them.

        Parameters
        ----------
        runs : list of np.ndarray
            Per-run coordinate arrays of shape (T_i, N, 3). The runs must be
            mutually comparable; pre-alignment is required when they originate from
            separate simulations.

        Returns
        -------
        CompressedTrajectory
            The compressed artifact.
        """
        # Align each run to a shared reference and flatten to (T_i, 3N).
        ref = None
        aligned = []
        for r in runs:
            a, ref = kabsch_align(np.asarray(r, dtype=np.float64), ref)
            aligned.append(a.reshape(a.shape[0], -1))   # (T_i, 3N)
        run_lengths = [a.shape[0] for a in aligned]

        # Whitening is the reconstruction transform, an invertible linear flow.
        wh = WhiteningTransform().fit(np.concatenate(aligned, axis=0))
        white_runs = [wh.forward(a) for a in aligned]   # (T_i, d), with d = 3N here
        d = white_runs[0].shape[1]

        # TICA slow modes are the discretization features.
        tica = TICA(lag=self.tica_lag, n_components=self.tica_dim)
        tica.fit(white_runs)
        tica_runs = [tica.transform(w) for w in white_runs]

        # Cluster the slow modes into microstates.
        labels, centers = discretize(tica_runs, self.n_states, self.seed)

        # The MSM is both the entropy model and the kinetic model.
        C = count_matrix(labels, self.n_states, self.msm_lag)
        T, pi = transition_matrix(C, self.reversible)

        # Per-state means in whitened space act as a structural prior.
        all_white = np.concatenate(white_runs, axis=0)
        all_lab = np.concatenate(labels, axis=0)
        state_means = np.zeros((self.n_states, d))
        for s in range(self.n_states):
            m = all_lab == s
            if m.any():
                state_means[s] = all_white[m].mean(axis=0)

        # Residual about the per-state mean, then dithered quantization.
        residual = all_white - state_means[all_lab]
        step = (residual.std() + 1e-12) * (2.0 ** (1 - self.n_bits)) * 3.0
        codec = DitheredResidualCodec(n_bits=self.n_bits, seed=self.seed)
        q = codec.quantize(residual, step)

        # Range-code each run's state sequence against T, with pi as the initial law.
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
        """Reconstruct aligned coordinates for each run.

        The reconstruction proceeds by decoding the state sequences, adding the
        per-state means, adding the dequantized residuals, applying the inverse
        whitening transform, and reshaping to (T, N, 3)."""
        # Decode the state sequences.
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

    @staticmethod
    def report(ct: CompressedTrajectory) -> dict:
        """Compute the bit accounting for the compressed artifact.

        Residuals are charged at the fixed quantizer rate (n_bits per value), which
        is an upper bound; entropy-coding the residuals would only reduce it. The
        side information is the one-time model cost, amortized over the T frames."""
        T_total = sum(ct.run_lengths)
        N3 = ct.reconstruct_dim
        state_bits = 8 * sum(len(b) for b in ct.coded_states)
        residual_bits = T_total * N3 * ct.n_bits
        stream_bits = state_bits + residual_bits
        # One-time side information, amortized over the T frames.
        side_bits = (
            ct.T.size * 16                       # transition matrix at 16-bit
            + ct.state_means.size * 32           # per-state means at 32-bit
            + ct.whitener.W_.size * 32           # whitening matrix at 32-bit
            + ct.centers.size * 32               # k-means centers
        )
        orig_bits = T_total * N3 * 32            # DCD baseline: 32-bit float coords
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
