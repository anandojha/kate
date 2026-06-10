"""Coverage for inspect_traj: residue classification, heavy-atom selection, and the
inspect() + --compress paths driven by the tiny synthetic DCD fixture."""
import numpy as np
import pytest

from epc import inspect_traj as it
from _synth import write_tiny_dcd


def test_classify_all_branches():
    assert it.classify("ALA") == "protein"
    assert it.classify("HIE") == "protein"
    assert it.classify("HOH") == "water"
    assert it.classify("NA") == "ion"
    assert it.classify("BEN") == "other"


def test_heavy_indices_excludes_hydrogens_and_filters_resnames():
    pytest.importorskip("mdtraj")
    import mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    r = top.add_residue("ALA", ch)
    top.add_atom("CA", md.element.carbon, r)
    top.add_atom("H1", md.element.hydrogen, r)
    assert len(it.heavy_indices(top)) == 1                  # H dropped
    assert len(it.heavy_indices(top, resnames={"GLY"})) == 0  # resname filter


def test_inspect_and_run_compress_on_tiny_dcd(tmp_path, capsys):
    pytest.importorskip("mdtraj")
    pdb, dcd = write_tiny_dcd(tmp_path, n_frames=300, n_atoms=8, seed=0)
    facts = it.inspect(pdb, dcd)
    assert facts["n_frames"] == 300
    assert len(facts["stored"]) == 8
    out = capsys.readouterr().out
    assert "TOPOLOGY" in out and "FOOTPRINT" in out
    # the classical-codec compression path
    it.run_compress(pdb, dcd, stride=1, lag=10, nbits=4, nstates=20, facts=facts)
    out2 = capsys.readouterr().out
    assert "COMPRESSION RUN" in out2 and "implied timescales" in out2
