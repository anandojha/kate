"""`glide analyze` (T2): deeptime kinetics from the artifact's stored dtraj -- a
reversible-MLE MSM, an implied-timescale lag scan, and Bayesian error bars, with NO
coordinate decode. deeptime-gated."""
import numpy as np
import pytest

pytest.importorskip("deeptime")

from glide.cli import main
from glide.artifact import save_artifact
from _synth import kinetics_artifact


def test_analyze_default_runs_and_reports(tmp_path, capsys):
    art = kinetics_artifact(n_steps=3000, nstates=30, lag=10, seed=0)
    p = str(tmp_path / "k.glide")
    save_artifact(art, p)
    main(["analyze", p, "--k", "3"])
    out = capsys.readouterr().out
    assert "reversible-MLE MSM" in out
    assert "IMPLIED TIMESCALES" in out


def test_analyze_lag_scan_and_bayes(tmp_path, capsys):
    art = kinetics_artifact(n_steps=4000, nstates=30, lag=10, seed=1)
    p = str(tmp_path / "k.glide")
    save_artifact(art, p)
    main(["analyze", p, "--lag-scan", "--bayes", "--k", "2", "--n-samples", "20"])
    out = capsys.readouterr().out
    assert "LAG SCAN" in out
    assert "BAYESIAN error bars" in out
    assert "+/-" in out


def test_analyze_kinetics_are_finite():
    # the underlying deeptime estimate yields finite, positive slow timescales
    from glide import kinetics_deeptime as kd
    art = kinetics_artifact(n_steps=3000, nstates=30, lag=10, seed=2)
    its = kd.implied_timescales(art.dtraj, lag=10, k=2)
    assert np.all(np.isfinite(its)) and np.all(its > 0)
