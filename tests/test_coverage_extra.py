"""Targeted coverage for remaining CLI/runner branches and the import guards."""
import numpy as np
import pytest

from kate.cli import main, _load_reference_counts
from kate.artifact import save_artifact
from _synth import toy_artifact


# ---- CLI: reference loading + bound branches ----
def test_load_reference_counts_npz_and_npy(tmp_path):
    C = np.array([[10.0, 1.0], [1.0, 10.0]])
    pnpz = str(tmp_path / "ref.npz"); np.savez(pnpz, counts=C)
    pnpy = str(tmp_path / "ref.npy"); np.save(pnpy, C)
    assert np.allclose(_load_reference_counts(pnpz), C)
    assert np.allclose(_load_reference_counts(pnpy), C)


def test_bound_with_matrix_ref_and_shape_mismatch(tmp_path, capsys):
    q = str(tmp_path / "q.kate"); save_artifact(toy_artifact(a=0.05, seed=1), q)
    ref3 = str(tmp_path / "r.npy"); np.save(ref3, np.eye(3) + 0.2)   # 3 states vs 2
    main(["bound", q, ref3])
    out = capsys.readouterr().out
    assert "KINETIC FIDELITY" in out and "WARNING" in out


# ---- CLI: decompress --full-atom on an artifact with NO residual stage ----
def test_decompress_full_atom_without_residual(tmp_path, capsys):
    pytest.importorskip("torch")
    from kate.flow import RealNVP
    from kate.artifact import Artifact
    from kate.codec import gaussian_cumfreq, encode_iid
    flow = RealNVP(2, hidden=16, n_layers=4)
    L, zmax = 1 << 12, 6.0
    levels = np.array([100, 200, 150, 250], dtype=np.int64)
    coded = encode_iid(levels, gaussian_cumfreq(L, zmax))
    art = Artifact(cv_dim=2, L=L, zmax=zmax, n_keep=2, coded_latents=coded,
                   kept_idx=np.array([0, 1]), run_lengths=[2], dtraj=[np.array([0, 1])],
                   centers=np.zeros((2, 2)), counts=np.eye(2) + 1.0, T_msm=np.eye(2),
                   n_states=2, lag=1, stride=1, dt_ps=100.0, dt_strided_ns=0.1,
                   flow_arch={"dim": 2, "hidden": 16, "n_layers": 4}, residual=None,
                   flow_state={k: v.detach().cpu() for k, v in flow.state_dict().items()})
    p = str(tmp_path / "nf.kate"); save_artifact(art, p)
    main(["decompress", p, "-o", str(tmp_path / "o.npy"), "--full-atom"])
    assert "NO residual" in capsys.readouterr().out


# ---- CLI: analyze branches (explicit lags + the deeptime guard) ----
def test_analyze_explicit_lags(tmp_path, capsys):
    pytest.importorskip("deeptime")
    from _synth import kinetics_artifact
    p = str(tmp_path / "k.kate"); save_artifact(kinetics_artifact(n_steps=3000, nstates=20, lag=10), p)
    main(["analyze", p, "--lag-scan", "--lags", "5,10,20", "--k", "2"])
    assert "LAG SCAN" in capsys.readouterr().out


def test_analyze_requires_deeptime(tmp_path, monkeypatch):
    p = str(tmp_path / "k.kate"); save_artifact(toy_artifact(), p)
    import kate.kinetics_deeptime as kd
    monkeypatch.setattr(kd, "_HAVE_DEEPTIME", False)
    with pytest.raises(SystemExit):
        main(["analyze", p])


# ---- runner: helpers, the streaming+vampnet rejection, print_report branches ----
def test_runner_helpers_free_energy_and_kl():
    from kate.runner import free_energy_1d, kl_1d
    v = np.random.default_rng(0).standard_normal(2000)
    bins = np.linspace(-3, 3, 21)
    assert free_energy_1d(v, bins).shape[0] == 20
    assert kl_1d(v, v, bins) >= 0.0


def test_streaming_vampnet_is_rejected(tmp_path):
    pytest.importorskip("mdtraj")
    from kate.runner import run_kate
    from _synth import write_tiny_dcd
    pdb, dcd = write_tiny_dcd(tmp_path, n_frames=120, n_atoms=6, seed=0)
    with pytest.raises(SystemExit):
        run_kate(pdb, dcd, streaming=True, cv="vampnet", verbose=False)


def test_print_report_all_optional_branches(capsys):
    pytest.importorskip("torch")
    from kate.runner import compress_trajectory, print_report
    from _synth import metastable_coords
    coords = metastable_coords(n_steps=1200, n_atoms=6, seed=0)
    for kw in (dict(entropy="temporal"), dict(entropy="predictive")):
        _, rep = compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=20,
                                     nstates=20, lag=10, seed=0, verbose=False, **kw)
        print_report(rep)
    out = capsys.readouterr().out
    assert "LEARNED-ENTROPY" in out and "PREDICTIVE" in out


def test_print_report_vampnet_branch(capsys):
    pytest.importorskip("deeptime")
    pytest.importorskip("torch")
    from kate.runner import compress_trajectory, print_report
    from _synth import metastable_coords
    coords = metastable_coords(n_steps=1500, n_atoms=6, seed=0)
    _, rep = compress_trajectory([coords], cv="vampnet", cv_dim=2, keep_frac=0.1,
                                 epochs=20, nstates=20, lag=10, seed=0, verbose=False)
    print_report(rep)
    assert "VAMPnet" in capsys.readouterr().out


# ---- import guards (simulate the library being absent) ----
def test_kinetics_deeptime_require_raises(monkeypatch):
    import kate.kinetics_deeptime as kd
    monkeypatch.setattr(kd, "_HAVE_DEEPTIME", False)
    monkeypatch.setattr(kd, "_IMPORT_ERR", ImportError("simulated"))
    with pytest.raises(ImportError):
        kd._require()


def test_vampnet_require_raises(monkeypatch):
    import kate.vampnet_cv as vc
    monkeypatch.setattr(vc, "_HAVE_DEEPTIME", False)
    monkeypatch.setattr(vc, "_IMPORT_ERR", ImportError("simulated"))
    with pytest.raises(ImportError):
        vc._require()
