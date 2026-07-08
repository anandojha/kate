"""Cover inspect_traj's save-interval / extrapolation branches using trajectory
formats that preserve per-frame timestamps (NetCDF = real dt; DCD = integer
frame-index dt)."""
import os

import numpy as np
import pytest

from kate import inspect_traj as it


def _save(tmp, ext, dt_ps, n=150, n_atoms=5):
    import mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    r = top.add_residue("ALA", ch)
    for i in range(n_atoms):
        top.add_atom("C%d" % i, md.element.carbon, r)
    xyz = (np.random.default_rng(0).standard_normal((n, n_atoms, 3)) * 0.05).astype("float32")
    traj = md.Trajectory(xyz, top)
    traj.time = (np.arange(n) * dt_ps).astype("float32")
    pdb = os.path.join(str(tmp), "s.pdb")
    trj = os.path.join(str(tmp), "s." + ext)
    traj[0].save_pdb(pdb)
    traj.save(trj)
    return pdb, trj


def test_inspect_netcdf_real_dt_and_extrapolation(tmp_path, capsys):
    pytest.importorskip("mdtraj")
    pdb, nc = _save(tmp_path, "nc", dt_ps=2.5)
    facts = it.inspect(pdb, nc)
    out = capsys.readouterr().out
    assert "save interval" in out
    # real (non-integer) dt -> the duration + 125 us extrapolation branch
    assert "EXTRAPOLATION" in out or facts.get("dt_ns") is not None


def test_inspect_dcd_integer_dt(tmp_path, capsys):
    pytest.importorskip("mdtraj")
    pdb, dcd = _save(tmp_path, "dcd", dt_ps=1.0)
    it.inspect(pdb, dcd)
    assert "save interval" in capsys.readouterr().out
