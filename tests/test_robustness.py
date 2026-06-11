"""Regression tests for the robustness/efficiency fixes found by the first real-data
run (NTL9): (1) `analyze` must not crash when the deeptime reversible-MLE fails to
converge on an ill-conditioned discretization -- it flags the lag and continues;
(2) the residual is stored compactly (int16 + compressed npz) and still round-trips."""
import numpy as np
import pytest


# ---- Fix 1: analyze robustness to deeptime MLE non-convergence ----
def test_its_lag_scan_returns_nan_on_mle_failure(monkeypatch):
    pytest.importorskip("deeptime")
    from glide import kinetics_deeptime as kd
    from _synth import kinetics_artifact
    art = kinetics_artifact(n_steps=3000, nstates=20, lag=10)
    real = kd.mlmsm

    def flaky(dtrajs, lag, reversible=True, **kw):
        if int(lag) == 20:
            raise RuntimeError("Stationary distribution contains entries smaller than 0")
        return real(dtrajs, lag, reversible)
    monkeypatch.setattr(kd, "mlmsm", flaky)

    scan = kd.its_lag_scan(art.dtraj, [10, 20, 40], k=2)
    assert scan.shape == (3, 2)
    assert np.all(np.isnan(scan[1]))       # the failing lag -> NaN row, no raise
    assert np.all(np.isfinite(scan[0]))    # the others still estimated


def test_cmd_analyze_does_not_crash_on_mle_failure(tmp_path, monkeypatch, capsys):
    pytest.importorskip("deeptime")
    from glide import kinetics_deeptime as kd
    from glide.artifact import save_artifact
    from glide.cli import main
    from _synth import kinetics_artifact
    p = str(tmp_path / "k.glide")
    save_artifact(kinetics_artifact(n_steps=3000, nstates=20, lag=10), p)

    def boom(*a, **k):
        raise RuntimeError("reversible MLE did not converge")
    monkeypatch.setattr(kd, "implied_timescales", boom)
    main(["analyze", p])                   # must NOT raise
    out = capsys.readouterr().out
    assert "did not converge" in out


def test_cmd_analyze_bayes_failure_is_reported(tmp_path, monkeypatch, capsys):
    pytest.importorskip("deeptime")
    from glide import kinetics_deeptime as kd
    from glide.artifact import save_artifact
    from glide.cli import main
    from _synth import kinetics_artifact
    p = str(tmp_path / "k.glide")
    save_artifact(kinetics_artifact(n_steps=3000, nstates=20, lag=10), p)
    monkeypatch.setattr(kd, "bayes_timescales",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no converge")))
    main(["analyze", p, "--bayes"])
    assert "did not converge" in capsys.readouterr().out


# ---- Fix 2: compact residual storage that still round-trips ----
def test_residual_stored_compact_and_lossless(tmp_path):
    pytest.importorskip("torch")
    from glide.runner import compress_trajectory
    from glide.artifact import save_artifact, load_artifact
    from _synth import metastable_coords
    art, _ = compress_trajectory([metastable_coords(1500, 6, seed=0)], cv_dim=2,
                                 keep_frac=0.1, epochs=30, nstates=20, lag=10,
                                 seed=0, verbose=False)
    p = str(tmp_path / "r.glide")
    save_artifact(art, p)
    z = np.load(p + "/arrays.npz")
    # residual quantizer levels stored as int16, not int64
    assert z["residual__q"].dtype == np.int16
    # the values are preserved exactly (lossless round-trip of the levels)
    loaded = load_artifact(p, with_flow=False)
    assert np.array_equal(np.asarray(loaded.residual["q"]),
                          np.asarray(art.residual["q"]))
