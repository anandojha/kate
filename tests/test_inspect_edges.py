"""Edge branches of inspect_traj: non-uniform timestamps, the run_compress
real-dt summary, and a non-positive (header-unreliable) dt."""
import os

import numpy as np
import pytest

from epc import inspect_traj as it
from _synth import metastable_coords


def _save_nc(tmp, name, time, n_atoms=6):
    import mdtraj as md
    n = len(time)
    top = md.Topology(); ch = top.add_chain(); r = top.add_residue("ALA", ch)
    for i in range(n_atoms):
        top.add_atom("C%d" % i, md.element.carbon, r)
    xyz = (metastable_coords(n, n_atoms, seed=0) * 0.05).astype("float32")
    traj = md.Trajectory(xyz, top)
    traj.time = np.asarray(time, dtype="float32")
    pdb = os.path.join(str(tmp), name + ".pdb")
    nc = os.path.join(str(tmp), name + ".nc")
    traj[0].save_pdb(pdb)
    traj.save(nc)
    return pdb, nc


def test_inspect_nonuniform_dt_and_run_compress_summary(tmp_path, capsys):
    pytest.importorskip("mdtraj")
    t = np.arange(300, dtype="float32") * 2.5
    t[150:] += 1.0                                    # a non-uniform step
    pdb, nc = _save_nc(tmp_path, "nu", t)
    facts = it.inspect(pdb, nc)                        # -> NON-uniform branch
    it.run_compress(pdb, nc, stride=1, lag=10, nbits=4, nstates=20, facts=facts)
    out = capsys.readouterr().out
    assert "NON-uniform" in out and "COMPRESSION RUN" in out


def test_inspect_nonpositive_dt(tmp_path, capsys):
    pytest.importorskip("mdtraj")
    pdb, nc = _save_nc(tmp_path, "flat", np.zeros(120))   # constant time -> dt <= 0
    it.inspect(pdb, nc)
    assert "non-positive" in capsys.readouterr().out
