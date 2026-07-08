"""T4: full-atom (3N) reconstruction. The kept frames' CV recovers the SLOW modes
(flow inverse); the per-state dithered residual stage recovers the fast modes TICA
discarded. `decompress --full-atom` must return 3N coordinates, and the residual must
recover substantially more than CV-space alone. torch-gated."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kate.runner import compress_trajectory
from kate.artifact import save_artifact
from kate.cli import main
from kate.kinetic_codec import kabsch_align
from _synth import metastable_coords


def _rmsd(A, B, N):
    return float(np.sqrt(((A - B) ** 2).reshape(-1, N, 3).sum(2).mean()))


def test_full_atom_roundtrip_and_beats_cv_only(tmp_path):
    coords = metastable_coords(n_steps=2000, n_atoms=8, seed=0)
    N = coords.shape[1]
    art, report = compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=50,
                                      nstates=30, lag=10, n_bits=8, seed=0, verbose=False)
    # the residual stage is present and the self-check RMSD is small
    assert art.residual is not None
    assert report["fullatom_rmsd"] < 0.05

    # ground-truth aligned kept frames
    aligned, _ = kabsch_align(coords, art.align_ref)
    Xkept = aligned.reshape(len(coords), -1)[art.kept_idx]

    p = str(tmp_path / "e.kate"); save_artifact(art, p)

    # full-atom reconstruction
    out_full = str(tmp_path / "full.npy")
    main(["decompress", p, "-o", out_full, "--full-atom"])
    rec = np.load(out_full)
    assert rec.shape == (art.n_keep, N, 3)
    rmsd_full = _rmsd(rec.reshape(art.n_keep, -1), Xkept, N)
    assert rmsd_full < 0.05
    # decompress reproduces the compress-time self-check exactly
    assert abs(rmsd_full - report["fullatom_rmsd"]) < 1e-5

    # CV-space alone (slow modes only) misses the fast modes -> much larger error
    out_cv = str(tmp_path / "cv.npy")
    main(["decompress", p, "-o", out_cv])
    cv = np.load(out_cv)
    Vp = np.linalg.pinv(np.asarray(art.tica_eigvecs))
    X_approx = np.asarray(art.tica_mean) + cv @ Vp
    rmsd_cv_only = _rmsd(X_approx, Xkept, N)
    assert rmsd_cv_only > 3 * rmsd_full   # the residual recovers most of the structure
