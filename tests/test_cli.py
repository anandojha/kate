"""CLI / `kate bound` tests. The bound is the pure-numpy scorer: it must run without
torch, give ~0 transition term against itself (KATE retains the MSM), and a large
transition term against a kinetically-different reference (ensemble preserved,
kinetics not)."""
import subprocess
import sys

import numpy as np
import pytest

from kate.cli import main
from kate.artifact import save_artifact
from kate.pathbound import report_kinetic_fidelity
from kate.kinetic_codec import transition_matrix
from _synth import toy_artifact, metastable_coords


def test_bound_self_is_near_zero_cross_is_large():
    fast = toy_artifact(a=0.05, seed=1)
    slow = toy_artifact(a=0.01, seed=2)
    Tf, _ = transition_matrix(fast.counts, reversible=True)
    Ts, _ = transition_matrix(slow.counts, reversible=True)
    r_self = report_kinetic_fidelity(Tf, Tf, lag=1)
    r_cross = report_kinetic_fidelity(Tf, Ts, lag=1)
    # KATE vs itself: kinetics preserved
    assert r_self["transition_kl_rate_nats_per_step"] < 1e-9
    # vs a slower chain with the SAME (uniform) stationary distribution:
    assert r_cross["ensemble_kl_nats"] < 1e-2          # ensemble preserved
    assert r_cross["transition_kl_rate_nats_per_step"] > 1e-3   # kinetics NOT
    assert r_cross["its_cmp"][0] > 1.5 * r_cross["its_ref"][0]


def test_bound_cli_runs_and_reports(tmp_path, capsys):
    q = str(tmp_path / "q.kate")
    r = str(tmp_path / "r.kate")
    save_artifact(toy_artifact(a=0.05, seed=1), q)
    save_artifact(toy_artifact(a=0.01, seed=2), r)
    main(["bound", q, r])
    out = capsys.readouterr().out
    assert "KINETIC FIDELITY" in out
    assert "transition term" in out
    assert "Pinsker PAIR bound" in out


def test_bound_cli_is_torch_free(tmp_path):
    q = str(tmp_path / "q.kate")
    r = str(tmp_path / "r.kate")
    save_artifact(toy_artifact(a=0.05, seed=1), q)
    save_artifact(toy_artifact(a=0.01, seed=2), r)
    code = (
        "import sys\n"
        "from kate.cli import main\n"
        f"main(['bound', {q!r}, {r!r}])\n"
        "raise SystemExit(1 if 'torch' in sys.modules else 0)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"torch imported on the bound path:\n{res.stdout}{res.stderr}"


def test_compress_decompress_bound_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from kate.runner import compress_trajectory
    from kate.artifact import save_artifact as _save, load_artifact as _load
    coords = metastable_coords(n_steps=1500, n_atoms=6, seed=0)
    art, report = compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=40,
                                      nstates=30, lag=10, seed=0, verbose=False)
    # KATE retains the MSM -> the self path bound's transition term is ~0
    assert report["kinetic_bound"]["transition_kl_rate_nats_per_step"] < 1e-6
    p = str(tmp_path / "e.kate")
    _save(art, p)
    loaded = _load(p, with_flow=True)
    assert loaded.n_keep == art.n_keep
    assert len(loaded.dtraj) == 1 and loaded.dtraj[0].shape[0] == 1500
    # decompress (CV-space kept frames) via the CLI
    out = str(tmp_path / "kept.npy")
    main(["decompress", p, "-o", out])
    cv = np.load(out)
    assert cv.shape == (art.n_keep, art.cv_dim)
