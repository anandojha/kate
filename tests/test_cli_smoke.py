"""DCD-free CLI smoke tests for the two subcommands whose backends are tested but whose
CLI dispatch (argparse -> cmd_* -> run_kate/run_benchmark) was previously not exercised:
`compress` and `benchmark`. A tiny synthetic PDB+DCD fixture lets us drive them through
main([...]) without the real cluster trajectory."""
import os

import numpy as np
import pytest

from kate.cli import main
from _synth import write_tiny_dcd


def test_cli_compress_smoke(tmp_path):
    pytest.importorskip("mdtraj")
    pytest.importorskip("torch")
    pdb, dcd = write_tiny_dcd(tmp_path, n_frames=400, n_atoms=6, seed=0)
    art = str(tmp_path / "out.kate")
    main(["compress", pdb, dcd, "-o", art, "--cv-dim", "2", "--nstates", "20",
          "--epochs", "15", "--keep-frac", "0.2", "--stride", "1", "--dt-ps", "100",
          "--lag-ns", "1.0"])
    # the artifact directory + its parts were written
    assert os.path.isdir(art)
    assert os.path.exists(os.path.join(art, "config.json"))
    assert os.path.exists(os.path.join(art, "arrays.npz"))
    assert os.path.exists(os.path.join(art, "flow.pt"))


def test_cli_benchmark_smoke(tmp_path):
    pytest.importorskip("mdtraj")
    pytest.importorskip("matplotlib")
    pdb, dcd = write_tiny_dcd(tmp_path, n_frames=400, n_atoms=6, seed=1)
    out = str(tmp_path / "bench")
    # kate + a local pseudo-baseline -> no external tool, no torch needed
    main(["benchmark", pdb, dcd, "--methods", "kate,shuffle", "--lag", "10",
          "--nstates", "20", "--stride", "1", "--dt-ps", "100", "--out", out])
    assert os.path.exists(out + ".png")
