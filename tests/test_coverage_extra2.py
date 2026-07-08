"""Final coverage pass: verbose branches, artifact build-guards, the make_predictor
TCN branch, default-mu_p path, npz fallback, benchmark skip/plot, and kinetic_codec
edge cases (all-zero PMF, low-rank whitening)."""
import os

import numpy as np
import pytest


def _corr(T=300, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((T, dim))
    for t in range(1, T):
        z[t] = 0.9 * z[t - 1] + 0.4 * rng.standard_normal(dim)
    return z


def test_fit_verbose_and_make_predictor_tcn(capsys):
    pytest.importorskip("torch")
    from kate.temporal_prior import TemporalPrior
    from kate.predictive_coder import CausalGRUPredictor, CausalTCNPredictor, make_predictor
    z = _corr()
    TemporalPrior(2, hidden=16, n_layers=2).fit(z, epochs=10, verbose=True)
    CausalGRUPredictor(2, hidden=16).fit(z, epochs=10, verbose=True)
    assert isinstance(make_predictor(2, kind="tcn"), CausalTCNPredictor)
    assert "NLL" in capsys.readouterr().out


def test_codec_fit_encode_verbose(capsys):
    pytest.importorskip("torch")
    from kate.codec import KateCodec
    from _synth import metastable_coords
    KateCodec(n_keep_frac=0.1, flow_epochs=10, tica_dim=2, n_states=20).fit_encode(
        [metastable_coords(400, 6, seed=0)], verbose=True)
    assert "training flow density" in capsys.readouterr().out


def test_compress_streaming_verbose(capsys):
    pytest.importorskip("torch")
    from kate.runner import compress_streaming
    from _synth import metastable_coords
    coords = metastable_coords(1200, 6, seed=0)

    def factory():
        return (coords[i:i + 400] for i in range(0, len(coords), 400))
    compress_streaming(factory, cv_dim=2, keep_frac=0.1, epochs=10, nstates=20,
                       lag=10, seed=0, verbose=True)
    out = capsys.readouterr().out
    assert "pass 1" in out and "pass 2" in out and "pass 3" in out


def test_run_kate_streaming_dcd(tmp_path):
    pytest.importorskip("mdtraj"); pytest.importorskip("torch")
    from kate.runner import run_kate
    from _synth import write_tiny_dcd
    pdb, dcd = write_tiny_dcd(tmp_path, n_frames=300, n_atoms=6, seed=0)
    art, _ = run_kate(pdb, dcd, streaming=True, cv_dim=2, nstates=20, epochs=10,
                     keep_frac=0.2, stride=1, dt_ps=100, lag_ns=1.0, verbose=True)
    assert art.n_keep >= 2


def test_vampnet_verbose(capsys):
    pytest.importorskip("deeptime"); pytest.importorskip("torch")
    from kate.vampnet_cv import vampnet_cvs
    from kate.kinetic_codec import kabsch_align
    from _synth import metastable_coords
    c = metastable_coords(1500, 6, seed=0)
    a, _ = kabsch_align(c, None)
    vampnet_cvs([a.reshape(len(c), -1)], lag=10, dim=2, n_epochs=10, seed=0, verbose=True)
    assert "VAMPNet CVs" in capsys.readouterr().out


def test_artifact_build_guards(tmp_path):
    pytest.importorskip("torch")
    from kate.artifact import Artifact, save_artifact, load_artifact
    from kate.flow import RealNVP
    flow = RealNVP(2, hidden=16, n_layers=4)
    art = Artifact(cv_dim=2, L=1 << 12, zmax=6.0, n_keep=2, coded_latents=b"",
                   kept_idx=np.array([0, 1]), run_lengths=[2], dtraj=[np.array([0, 1])],
                   centers=np.zeros((2, 2)), counts=np.eye(2) + 1.0, T_msm=np.eye(2),
                   n_states=2, lag=1, stride=1, dt_ps=100.0, dt_strided_ns=0.1,
                   flow_arch={"dim": 2, "hidden": 16, "n_layers": 4},
                   flow_state={k: v.detach().cpu() for k, v in flow.state_dict().items()})
    assert art.build_temporal() is None and art.build_predictor() is None   # 77, 87
    p = str(tmp_path / "a.kate"); save_artifact(art, p)
    nf = load_artifact(p, with_flow=False)
    with pytest.raises(ValueError):
        nf.build_flow()                                                     # 101
    art.flow_kind = "bogus"
    with pytest.raises(ValueError):
        art.build_flow()                                                   # 113


def test_pathbound_transition_kl_default_mu():
    from kate.pathbound import transition_kl_rate
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    Q = np.array([[0.8, 0.2], [0.2, 0.8]])
    assert transition_kl_rate(P, Q) > 0                                     # default mu_p


def test_require_external_success_path(monkeypatch):
    import kate.baselines as bl
    monkeypatch.setenv("KATE_SZ3_BIN", "/usr/bin/true")
    assert bl._require_external("sz3") == "/usr/bin/true"                   # 63


def test_cli_load_reference_npz_first_key(tmp_path):
    from kate.cli import _load_reference_counts
    p = str(tmp_path / "x.npz"); np.savez(p, weird=np.eye(2) + 1.0)
    assert _load_reference_counts(p).shape == (2, 2)                        # cli 40


def test_benchmark_verbose_skip_and_empty_plot(tmp_path, capsys):
    from kate.benchmark import run_benchmark, _plot
    from _synth import metastable_coords
    res = run_benchmark([metastable_coords(1500, 6, seed=0)], methods=["kate", "sz3"],
                        lag=10, nstates=20, verbose=True)                   # 89, 124
    assert any(not r.get("available", False) for r in res)
    # _plot returns early when nothing is available (148)
    assert _plot([{"method": "sz3", "available": False}], np.array([1.0, 2.0]),
                 str(tmp_path / "z")) is None
    assert not os.path.exists(str(tmp_path / "z.png"))


def test_kinetic_codec_edges():
    from kate.kinetic_codec import _probs_to_cumfreq, WhiteningTransform
    assert _probs_to_cumfreq(np.zeros(4))[-1] > 0                           # 89 all-zero -> uniform
    wh = WhiteningTransform(rank=3).fit(np.random.default_rng(0).standard_normal((120, 6)))
    assert wh.W_.shape[1] == 3                                             # 313 low-rank
