"""Shared synthetic helpers for the test suite (no torch)."""
import numpy as np

from epc.kinetic_codec import count_matrix, transition_matrix
from epc.artifact import Artifact


def two_state_dtraj(n=20000, a=0.02, seed=0):
    """A reversible 2-state chain: identical (uniform) stationary distribution for any
    switching rate `a`, so changing `a` changes kinetics but NOT the ensemble."""
    rng = np.random.default_rng(seed)
    P = np.array([[1 - a, a], [a, 1 - a]])
    cdf = np.cumsum(P, axis=1)
    s = np.zeros(n, dtype=np.int64)
    u = rng.random(n)
    for t in range(1, n):
        s[t] = np.searchsorted(cdf[s[t - 1]], u[t])
    return s


def metastable_coords(n_steps=1500, n_atoms=6, a=0.01, intra=0.25, noise=0.10, seed=0):
    """A tiny 3-well metastable trajectory (T, N, 3) for end-to-end codec tests."""
    rng = np.random.default_rng(seed)
    P = np.array([[1 - a, a, 0.0], [a, 1 - 2 * a, a], [0.0, a, 1 - a]])
    cdf = np.cumsum(P, axis=1)
    m = np.zeros(n_steps, dtype=int)
    u = rng.random(n_steps)
    for t in range(1, n_steps):
        m[t] = np.searchsorted(cdf[m[t - 1]], u[t])
    wells = np.array([-2.0, 0.0, 2.0])
    xi = wells[m] + intra * rng.standard_normal(n_steps)
    ref = rng.standard_normal((n_atoms, 3)) * 2.0
    mode = rng.standard_normal((n_atoms, 3)); mode /= np.linalg.norm(mode)
    xyz = (ref[None] + xi[:, None, None] * mode[None]
           + noise * rng.standard_normal((n_steps, n_atoms, 3)))
    return xyz.astype(np.float64)


def toy_artifact(n=20000, a=0.02, seed=0):
    """A minimal, torch-free Artifact carrying a real 2-state MSM (no flow). Enough to
    exercise `epc bound` / save / load without training anything."""
    labels = [two_state_dtraj(n=n, a=a, seed=seed)]
    C = count_matrix(labels, 2, 1)
    T, _ = transition_matrix(C, reversible=True)
    return Artifact(
        cv_dim=1, L=1 << 12, zmax=6.0, n_keep=2,
        coded_latents=b"", kept_idx=np.array([0, 1], dtype=np.int64),
        run_lengths=[n], dtraj=labels, centers=np.zeros((2, 1)),
        counts=C, T_msm=T, n_states=2, lag=1,
        stride=1, dt_ps=100.0, dt_strided_ns=0.1,
        flow_arch={"dim": 1, "hidden": 64, "n_layers": 10},
    )
