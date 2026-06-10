"""Fuller coverage of inspect_traj: a DCD with timestamps, a periodic box, and a
non-standard (ligand) residue exercises the save-interval / units / footprint /
extrapolation branches and the residue-composition report; the --compress path is
driven through the module __main__."""
import os
import runpy
import sys

import numpy as np
import pytest

from epc import inspect_traj as it
from _synth import metastable_coords


def _write_rich_dcd(tmp, n=300, n_prot=4, dt_ps=2.5):
    import mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    rp = top.add_residue("ALA", ch)
    for i in range(n_prot):
        top.add_atom("C%d" % i, md.element.carbon, rp)
    rl = top.add_residue("BEN", ch)                      # non-standard ligand
    for i in range(2):
        top.add_atom("L%d" % i, md.element.carbon, rl)
    rw = top.add_residue("HOH", ch)                      # water
    top.add_atom("O", md.element.oxygen, rw)
    xyz = (metastable_coords(n, n_prot + 3, seed=0).astype(np.float32)) * 0.05
    traj = md.Trajectory(xyz, top)
    traj.time = (np.arange(n) * dt_ps).astype(np.float32)
    traj.unitcell_lengths = np.ones((n, 3), dtype=np.float32) * 3.0
    traj.unitcell_angles = np.ones((n, 3), dtype=np.float32) * 90.0
    pdb = os.path.join(str(tmp), "rich.pdb")
    dcd = os.path.join(str(tmp), "rich.dcd")
    traj[0].save_pdb(pdb)
    traj.save_dcd(dcd)
    return pdb, dcd


def test_inspect_reports_ligand_box_and_footprint(tmp_path, capsys):
    pytest.importorskip("mdtraj")
    pdb, dcd = _write_rich_dcd(tmp_path, n=300, n_prot=4, dt_ps=2.5)
    facts = it.inspect(pdb, dcd)
    out = capsys.readouterr().out
    assert facts["n_frames"] == 300
    assert "NON-STANDARD" in out               # the BEN ligand branch
    assert "FOOTPRINT" in out
    assert "water=1" in out or "water" in out   # residue composition


def test_inspect_traj_main_with_compress(tmp_path):
    pytest.importorskip("mdtraj")
    pytest.importorskip("torch")  # run_compress uses the classical codec (no torch),
    pdb, dcd = _write_rich_dcd(tmp_path, n=300, n_prot=6, dt_ps=2.0)
    old = sys.argv
    sys.argv = ["inspect_traj", pdb, dcd, "--compress", "--stride", "1",
                "--lag", "10", "--nbits", "4", "--nstates", "20"]
    try:
        runpy.run_module("epc.inspect_traj", run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = old


def test_compress_verbose_temporal_and_predictive(capsys):
    pytest.importorskip("torch")
    from epc.runner import compress_trajectory
    coords = metastable_coords(1000, 6, seed=0)
    compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=10, nstates=20,
                        lag=10, seed=0, verbose=True, entropy="temporal")
    compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=10, nstates=20,
                        lag=10, seed=0, verbose=True, entropy="predictive")
    out = capsys.readouterr().out
    assert "training temporal prior" in out and "predictor on the" in out


def test_decode_markov_robust_to_overread():
    from epc.kinetic_codec import encode_markov, decode_markov
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    pi = np.array([0.5, 0.5])
    states = np.array([0, 1, 0, 1, 0])
    blob = encode_markov(states, T, pi)
    dec = decode_markov(blob, 50, T, pi)         # over-read -> exercises bounds clamps
    assert dec.shape[0] == 50 and dec.min() >= 0 and dec.max() <= 1
