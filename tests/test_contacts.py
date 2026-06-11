"""Internal-coordinate (contacts) featurization: TICA on rotation/translation-invariant
inter-atomic distances instead of aligned Cartesian -- removes the spurious rigid-body
'slow modes' TICA-on-Cartesian can fabricate (REVIEW item 7). It must (a) produce finite,
ordered implied timescales, (b) be flagged in the report, and (c) round-trip through
save/load + the full-atom residual exactly like the cartesian path -- the reconstruction
is fit on coordinates and is independent of the CV featurization."""
import numpy as np
import pytest

pytest.importorskip("torch")

from glide.runner import compress_trajectory
from glide.artifact import save_artifact, load_artifact
from _synth import metastable_coords


def test_contacts_featurization_round_trip(tmp_path):
    coords = metastable_coords(1500, 8, seed=0)
    art, rep = compress_trajectory([coords], features="contacts", cv_dim=2,
                                   keep_frac=0.1, epochs=30, nstates=20, lag=10,
                                   seed=0, verbose=False)
    assert rep["features"] == "contacts"
    its = rep["implied_timescales_ns"]
    assert np.all(np.isfinite(its[:2])) and its[0] >= its[1] > 0     # valid kinetics
    # save/load + full-atom residual round-trips losslessly (featurization-agnostic recon)
    p = str(tmp_path / "c.glide")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=False)
    assert loaded.residual is not None
    assert np.array_equal(np.asarray(loaded.residual["q"]),
                          np.asarray(art.residual["q"]))


def test_contacts_and_cartesian_both_run():
    coords = metastable_coords(1200, 8, seed=1)
    _, r_c = compress_trajectory([coords], features="contacts", cv_dim=2, keep_frac=0.1,
                                 epochs=20, nstates=15, lag=10, seed=0, verbose=False)
    _, r_x = compress_trajectory([coords], features="cartesian", cv_dim=2, keep_frac=0.1,
                                 epochs=20, nstates=15, lag=10, seed=0, verbose=False)
    assert r_c["features"] == "contacts" and r_x["features"] == "cartesian"
    # full-atom reconstruction is reported for both
    assert r_c["fullatom_rmsd"] >= 0 and r_x["fullatom_rmsd"] >= 0
