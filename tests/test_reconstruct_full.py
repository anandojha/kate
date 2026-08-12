"""Full-length reconstruction from an artifact.

The representative of a microstate must be a retained conformation, not the mean of its
members. A mean is more compact than the frames it averages, by convexity of the norm,
and under a distance featurization that compaction merges populations and shortens the
slowest implied timescale. These pin the length, the frame-exactness of the retained
subset, and the fact that representatives are real stored conformations."""
import numpy as np
import pytest

from kate.runner import compress_trajectory, reconstruct_full_length


def _two_well_runs(seed=0, T=600, n_atoms=6):
    """Two runs of a diffusive two-well process embedded in Cartesian coordinates."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(2):
        s = np.zeros(T)
        for t in range(1, T):
            drift = -4.0 * s[t - 1] * (s[t - 1] ** 2 - 1.0)
            s[t] = s[t - 1] + 0.01 * drift + 0.10 * rng.normal()
        base = rng.normal(size=(n_atoms, 3)) * 0.1
        x = base[None] + s[:, None, None] * 0.3 + 0.02 * rng.normal(size=(T, n_atoms, 3))
        runs.append(x)
    return runs


@pytest.fixture(scope="module")
def artifact():
    runs = _two_well_runs()
    art, _ = compress_trajectory(runs, cv="tica", cv_dim=2, keep_frac=0.10, epochs=6,
                                nstates=8, lag=5, stride=1, dt_ps=100.0, seed=0,
                                verbose=False)
    return art, runs


def test_reconstruction_has_the_original_length(artifact):
    art, runs = artifact
    X, out = reconstruct_full_length(art)
    assert X.shape[0] == sum(len(r) for r in runs)
    assert X.shape[1:] == runs[0].shape[1:]
    assert [len(r) for r in out] == list(art.run_lengths)


def test_retained_frames_use_their_own_coded_values(artifact):
    """A retained frame must not be replaced by its state representative."""
    art, runs = artifact
    X, _ = reconstruct_full_length(art)
    kept = np.asarray(art.kept_idx, dtype=int)
    dtraj = np.concatenate([np.asarray(d) for d in art.dtraj]).astype(int)
    truth = np.concatenate(runs).reshape(len(dtraj), -1)
    err_kept = np.abs(X.reshape(len(dtraj), -1)[kept] - truth[kept]).mean()
    others = np.setdiff1d(np.arange(len(dtraj)), kept)
    err_other = np.abs(X.reshape(len(dtraj), -1)[others] - truth[others]).mean()
    assert err_kept < err_other


def test_representatives_are_retained_conformations_not_means(artifact):
    """Every frame's structure must coincide with some retained frame's structure."""
    art, _ = artifact
    X, _ = reconstruct_full_length(art)
    flat = X.reshape(X.shape[0], -1)
    kept = np.asarray(art.kept_idx, dtype=int)
    d = np.abs(flat[:, None, :] - flat[None, kept, :]).max(axis=2)
    assert d.min(axis=1).max() < 1e-8
