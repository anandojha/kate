"""Kinetic-resolution accounting (kinetics_deeptime.kinetic_resolution): the honest
'what can this trajectory actually validate' report. A well-sampled FAST process (many
independent events, tight Bayesian CI) must read as RESOLVED; an undersampled SLOW
process (few events) must read as NOT resolved -- you cannot certify a kinetic
observable the source never sampled. This is the discipline that keeps the NTL9
conclusions honest (its slow folding mode is sampling-limited, the fast band is not)."""
import numpy as np
import pytest

pytest.importorskip("deeptime")

from glide import kinetics_deeptime as kd
from _synth import two_state_dtraj


def test_well_sampled_fast_process_is_resolved():
    # t ~ 1/(2a) ~ 10 steps; 30k steps -> ~3000 independent events
    dt = [two_state_dtraj(n=30000, a=0.05, seed=0)]
    rep = kd.kinetic_resolution(dt, lag=1, dt_ns=1.0, k=1, n_samples=40)
    r = rep[0]
    assert r["n_events"] > 100              # abundantly sampled
    assert r["rel_err"] < 0.1               # tight Bayesian CI
    assert r["resolved"] is True
    assert r["ci_lo_ns"] >= 0.0             # timescales clamped non-negative


def test_undersampled_slow_process_is_not_resolved():
    # t ~ 500 steps; only 3k steps -> ~6 independent events -> cannot be certified
    dt = [two_state_dtraj(n=3000, a=0.001, seed=1)]
    rep = kd.kinetic_resolution(dt, lag=1, dt_ns=1.0, k=1, n_samples=40, min_events=10)
    r = rep[0]
    assert r["n_events"] < 10               # too few round trips
    assert r["resolved"] is False           # event-count gate fails it


def test_format_resolution_reports_verdict():
    resolved = [{"process": 1, "timescale_ns": 10.0, "ci_lo_ns": 9.0, "ci_hi_ns": 11.0,
                 "rel_err": 0.05, "n_events": 2500.0, "resolved": True}]
    s = kd.format_resolution(resolved, total_us=25.0)
    assert "25.0 us" in s and "YES" in s and "resolved:" in s
    none = [{"process": 1, "timescale_ns": 4e4, "ci_lo_ns": 0.0, "ci_hi_ns": 8e4,
             "rel_err": 0.6, "n_events": 0.6, "resolved": False}]
    s2 = kd.format_resolution(none)
    assert "NONE" in s2
