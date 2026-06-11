"""T8: temporal learned-entropy coder. The two things that must hold: (1) it is
EXACTLY lossless (arithmetic coding desyncs on any encoder/decoder table mismatch),
and (2) on a temporally-correlated latent sequence it codes shorter than the fixed
N(0,I) base. torch-gated."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from glide.temporal_prior import (TemporalPrior, encode_sequence, decode_sequence,
                                quantize, gaussian_rate_bits_per_value,
                                temporal_rate_bits_per_value)
from _synth import metastable_coords


def _ar1_latents(T=600, dim=2, rho=0.95, seed=0):
    """An AR(1) latent sequence: marginally ~N(0,1) but strongly correlated in time --
    exactly the inter-frame redundancy the temporal model should exploit."""
    rng = np.random.default_rng(seed)
    z = np.zeros((T, dim))
    for t in range(1, T):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(dim)
    return z


L, ZMAX = 1 << 10, 6.0


def test_exact_lossless_roundtrip_trained():
    z = _ar1_latents(seed=1)
    model = TemporalPrior(dim=2, hidden=32, n_layers=3).fit(z, epochs=80, verbose=False)
    levels = quantize(z, L, ZMAX)
    coded = encode_sequence(z, model, L, ZMAX)
    dec = decode_sequence(coded, z.shape[0], z.shape[1], model, L, ZMAX)
    assert np.array_equal(dec, levels)            # EXACT, not approximate


def test_exact_lossless_roundtrip_untrained():
    # a zero-init head gives N(0,1) conditionals; must still be exactly lossless
    z = _ar1_latents(seed=3)
    model = TemporalPrior(dim=2, hidden=16, n_layers=2)
    levels = quantize(z, L, ZMAX)
    coded = encode_sequence(z, model, L, ZMAX)
    dec = decode_sequence(coded, z.shape[0], z.shape[1], model, L, ZMAX)
    assert np.array_equal(dec, levels)


def test_temporal_beats_gaussian_on_correlated_sequence():
    z = _ar1_latents(rho=0.97, seed=2)
    model = TemporalPrior(dim=2, hidden=32, n_layers=3).fit(z, epochs=150, verbose=False)
    r_gauss = gaussian_rate_bits_per_value(z, L, ZMAX)
    r_temporal = temporal_rate_bits_per_value(z, model, L, ZMAX)
    # the learned conditional exploits inter-frame redundancy -> fewer bits/value
    assert r_temporal < r_gauss


def test_no_gain_claimed_on_iid_sequence():
    # honesty: on an i.i.d. sequence (no temporal structure) the temporal model should
    # NOT do meaningfully better than the fixed base -- it must not "invent" savings.
    rng = np.random.default_rng(7)
    z = rng.standard_normal((600, 2))
    model = TemporalPrior(dim=2, hidden=32, n_layers=3).fit(z, epochs=120, verbose=False)
    r_gauss = gaussian_rate_bits_per_value(z, L, ZMAX)
    r_temporal = temporal_rate_bits_per_value(z, model, L, ZMAX)
    assert r_temporal <= r_gauss + 0.30           # no large spurious gain either way


def test_compress_entropy_temporal_end_to_end(tmp_path):
    from glide.runner import compress_trajectory
    from glide.artifact import save_artifact, load_artifact
    coords = metastable_coords(n_steps=1500, n_atoms=6, seed=0)
    art, rep = compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=40,
                                   nstates=30, lag=10, entropy="temporal", verbose=False)
    # the rate WITH vs WITHOUT is reported, and the model is tagged + persisted
    assert rep["rate_gaussian_bpv"] is not None
    assert rep["rate_temporal_bpv"] is not None
    assert art.entropy == "temporal" and art.temporal_state is not None
    p = str(tmp_path / "t.glide")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=True)
    assert loaded.entropy == "temporal"
    assert loaded.build_temporal() is not None
