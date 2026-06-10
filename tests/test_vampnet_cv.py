"""T6: learned nonlinear slow CVs via VAMPnets. Verifies the deeptime VAMPNet wrapper
learns CVs with a sane VAMP score, and that `compress --cv vampnet` runs end-to-end
with the CV-agnostic full-atom reconstruction intact. deeptime+torch gated."""
import numpy as np
import pytest

pytest.importorskip("deeptime")
pytest.importorskip("torch")

from epc.kinetic_codec import kabsch_align, TICA
from epc import vampnet_cv as vc
from _synth import metastable_coords


def _feat(coords):
    a, _ = kabsch_align(coords, None)
    return a.reshape(len(coords), -1)


def test_vampnet_learns_cvs_with_sane_score():
    X = _feat(metastable_coords(n_steps=4000, n_atoms=8, seed=0))
    model, cvs, score = vc.vampnet_cvs([X], lag=10, dim=2, n_epochs=25, seed=0)
    assert len(cvs) == 1 and cvs[0].shape == (4000, 2)
    # VAMP2 score for a dim-d embedding lies in [1, 1+d] (trivial mode + slow modes)
    assert 1.0 <= score <= 1.0 + 2 + 1e-3


def test_vampnet_and_tica_scores_both_finite():
    X = _feat(metastable_coords(n_steps=5000, n_atoms=8, seed=1))
    tica = TICA(lag=10, n_components=2).fit([X])
    s_tica = vc.vamp_score([tica.transform(X)], lag=10)
    _, _, s_vnet = vc.vampnet_cvs([X], lag=10, dim=2, n_epochs=40, seed=0)
    assert np.isfinite(s_tica) and np.isfinite(s_vnet)
    assert s_vnet > 1.0           # captures more than the trivial constant mode


def test_compress_cv_vampnet_end_to_end(tmp_path):
    from epc.runner import compress_trajectory
    from epc.artifact import save_artifact
    from epc.cli import main
    coords = metastable_coords(n_steps=2000, n_atoms=6, seed=0)
    art, rep = compress_trajectory([coords], cv="vampnet", cv_dim=2, keep_frac=0.1,
                                   epochs=40, nstates=30, lag=10, seed=0, verbose=False)
    assert art.cv == "vampnet"
    assert rep["vamp_score"] is not None and np.isfinite(rep["vamp_score"])
    # full-atom reconstruction works with VAMPnet CVs (the decoder is CV-agnostic)
    assert rep["fullatom_rmsd"] < 0.1
    p = str(tmp_path / "v.epc")
    save_artifact(art, p)
    out = str(tmp_path / "full.npy")
    main(["decompress", p, "-o", out, "--full-atom"])
    rec = np.load(out)
    assert rec.shape == (art.n_keep, 6, 3)
