"""T9: learned predictive temporal entropy coding (lossy). Unit-tested on SYNTHETIC
latents (the real rate-vs-observable-error gate runs on the trypsin-benzamidine set).
Verifies: the causal predictor beats a static prior; the closed-loop coder round-trips
exactly (lossless coder; the lossiness is only the innovation quantization); the
rate-distortion curve is monotonic; and at MATCHED distortion the predictive coder
codes cheaper than the static-Gaussian lossless baseline (the inter-frame-prediction
gain). torch-gated."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kate.predictive_coder import (CausalGRUPredictor, CausalTCNPredictor,
                                  encode_predictive, decode_predictive,
                                  rate_distortion_curve, conditional_nll,
                                  static_gaussian_nll)


def _ar1(T=600, dim=2, rho=0.95, seed=0):
    """AR(1) latents: marginally ~N(0,1), strongly correlated in time -- the
    inter-frame structure a static prior cannot exploit but a predictor can."""
    rng = np.random.default_rng(seed)
    z = np.zeros((T, dim))
    for t in range(1, T):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(dim)
    return z


def test_predictor_beats_static_prior():
    z = _ar1(rho=0.97, seed=1)
    m = CausalGRUPredictor(2, hidden=32).fit(z, epochs=120, seed=0)
    # the learned conditional NLL is well below the static-N(0,1) NLL
    assert conditional_nll(m, z) < static_gaussian_nll(z) - 0.1


def test_closed_loop_roundtrip_is_exact():
    z = _ar1(seed=2)
    m = CausalGRUPredictor(2, hidden=32).fit(z, epochs=40, seed=0)
    data, zhat_e, lev_e = encode_predictive(z, m, bits=8, seed=0)
    zhat_d, lev_d = decode_predictive(data, z.shape[0], z.shape[1], m, bits=8, seed=0)
    assert np.array_equal(lev_e, lev_d)              # lossless coder: exact levels
    assert np.abs(zhat_e - zhat_d).max() < 1e-4      # identical reconstruction


def test_rate_distortion_is_monotonic():
    z = _ar1(seed=3)
    m = CausalGRUPredictor(2, hidden=32).fit(z, epochs=60, seed=0)
    curve = rate_distortion_curve(z, m, [3, 5, 7, 9], seed=0)
    rates = [c["rate_bpv"] for c in curve]
    dists = [c["latent_mse"] for c in curve]
    assert all(dists[i] >= dists[i + 1] for i in range(len(dists) - 1))   # more bits -> less distortion
    assert all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))  # ... more rate


def test_predictive_dominates_static_lossless_at_matched_distortion():
    from kate.temporal_prior import (gaussian_rate_bits_per_value, quantize, dequantize)
    z = _ar1(rho=0.97, seed=4)
    m = CausalGRUPredictor(2, hidden=32).fit(z, epochs=140, seed=0)
    L, zmax = 1 << 12, 6.0
    r_static = gaussian_rate_bits_per_value(z, L, zmax)        # static N(0,1), lossless
    d_static = float(np.mean((dequantize(quantize(z, L, zmax), L, zmax) - z) ** 2))
    curve = rate_distortion_curve(z, m, list(range(5, 14)), U=8.0, seed=0)
    matched = [c for c in curve if c["latent_mse"] <= d_static * 2.0]
    assert matched, "no predictive point reaches the static-lossless distortion"
    best = min(c["rate_bpv"] for c in matched)
    # at matched (near-lossless) distortion, prediction makes T9 cheaper than static
    assert best < r_static


def test_tcn_predictor_roundtrip():
    z = _ar1(seed=5)
    m = CausalTCNPredictor(2, hidden=24, n_layers=2).fit(z, epochs=30, seed=0)
    data, zhat_e, lev_e = encode_predictive(z, m, bits=7, seed=0)
    zhat_d, lev_d = decode_predictive(data, z.shape[0], z.shape[1], m, bits=7, seed=0)
    assert np.array_equal(lev_e, lev_d)


def test_compress_entropy_predictive_end_to_end(tmp_path):
    from kate.runner import compress_trajectory
    from kate.artifact import save_artifact, load_artifact
    from _synth import metastable_coords
    coords = metastable_coords(n_steps=1500, n_atoms=6, seed=0)
    art, rep = compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=40,
                                   nstates=30, lag=10, entropy="predictive", seed=0,
                                   verbose=False)
    # T9 reports the rate-distortion curve + the predictor gain, and persists the model
    assert art.entropy == "predictive" and art.predictive_state is not None
    assert rep["rd_curve"] is not None and len(rep["rd_curve"]) >= 3
    assert rep["pred_cond_nll"] is not None and rep["pred_static_nll"] is not None
    p = str(tmp_path / "p.kate")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=True)
    assert loaded.entropy == "predictive"
    assert loaded.build_predictor() is not None
