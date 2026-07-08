"""Coverage for the module __main__ self-tests and the script entry points (run via
runpy so the `if __name__ == '__main__'` blocks execute under coverage)."""
import runpy
import sys

import pytest

from kate.artifact import save_artifact
from _synth import toy_artifact, write_tiny_dcd


def _run_module_main(module, argv):
    old = sys.argv
    sys.argv = list(argv)
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = old


def test_flow_selftest_main():
    pytest.importorskip("torch")
    _run_module_main("kate.flow", ["kate.flow"])


def test_spline_flow_selftest_main():
    pytest.importorskip("torch")
    _run_module_main("kate.spline_flow", ["kate.spline_flow"])


def test_kinetics_deeptime_selftest_main():
    pytest.importorskip("deeptime")
    _run_module_main("kate.kinetics_deeptime", ["kate.kinetics_deeptime"])


def test_package_main_dispatches_bound(tmp_path):
    # python -m kate bound ... -> __main__.py -> cli.main (pure numpy)
    q = str(tmp_path / "q.kate"); r = str(tmp_path / "r.kate")
    save_artifact(toy_artifact(a=0.05, seed=1), q)
    save_artifact(toy_artifact(a=0.01, seed=2), r)
    _run_module_main("kate.__main__", ["kate", "bound", q, r])


def test_cli_module_main_guard(tmp_path):
    # python -m kate.cli bound ... -> cli.py's __main__ guard
    q = str(tmp_path / "q.kate"); r = str(tmp_path / "r.kate")
    save_artifact(toy_artifact(a=0.05, seed=1), q)
    save_artifact(toy_artifact(a=0.01, seed=2), r)
    _run_module_main("kate.cli", ["kate", "bound", q, r])


def test_runner_main_entry(tmp_path):
    pytest.importorskip("mdtraj")
    pytest.importorskip("torch")
    pdb, dcd = write_tiny_dcd(tmp_path, n_frames=300, n_atoms=6, seed=0)
    out = str(tmp_path / "r.kate")
    _run_module_main("kate.runner", [
        "kate.runner", pdb, dcd, "--cv-dim", "2", "--nstates", "20", "--epochs", "10",
        "--keep-frac", "0.2", "--stride", "1", "--dt-ps", "100", "--lag-ns", "1.0",
        "-o", out])
