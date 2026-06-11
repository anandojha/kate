"""
The hard guarantee that `glide bound` (and the rest of the pure-numpy path) runs on a
box with NEITHER torch NOR deeptime installed: importing `glide` -- or
`from glide import pathbound` -- must not drag either heavy library into sys.modules.

Run in a SUBPROCESS, not in-process: within one pytest session another test
(test_flow / test_codec) imports torch, which would pollute this process's
sys.modules and make an in-process check meaningless.
"""
import subprocess
import sys


def _modules_after(import_stmt: str):
    code = (
        "import sys\n"
        f"{import_stmt}\n"
        "bad = sorted(m for m in ('torch', 'deeptime') if m in sys.modules)\n"
        "print('LOADED:' + ','.join(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r


def test_import_glide_pulls_in_neither_torch_nor_deeptime():
    r = _modules_after("import glide")
    assert r.returncode == 0, f"eager heavy import on `import glide`: {r.stdout}{r.stderr}"


def test_from_glide_import_pathbound_pulls_in_neither():
    r = _modules_after("from glide import pathbound")
    assert r.returncode == 0, (
        f"eager heavy import on `from glide import pathbound`: {r.stdout}{r.stderr}"
    )


def test_pathbound_report_runs_without_torch_or_deeptime():
    # The kinetic bound must be fully usable with neither library present.
    code = (
        "import sys, numpy as np\n"
        "from glide import report_kinetic_fidelity\n"
        "P = np.array([[0.9, 0.1], [0.1, 0.9]])\n"
        "Q = np.array([[0.8, 0.2], [0.2, 0.8]])\n"
        "out = report_kinetic_fidelity(P, Q, lag=1)\n"
        "assert out['transition_kl_rate_nats_per_step'] > 0\n"
        "bad = [m for m in ('torch', 'deeptime') if m in sys.modules]\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}{r.stderr}"
