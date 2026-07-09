"""Tests for the kinetic rate-distortion training module (T11).

These exercise the mechanism -- the flow-based rate, the frozen analysis transform, the
differentiable training step, and the honest hard-state certificate check. Whether the
kinetic objective beats mean-squared error is an empirical demonstration (see
examples/demo_rd_train.py), not asserted here.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kate.flow import RealNVP
from kate.bound_loss import SoftStateEncoder
from kate.rd_train import (latent_rate_bits, KineticRDCompressor, train_rd,
                           hard_state_error, rate_distortion_curve)


def _system(T=6000, switch=0.01, seed=0):
    """Small bistable slow coordinate (dim0, low amplitude) plus fast noise (dims 1-3)."""
    rng = np.random.default_rng(seed)
    s = np.ones(T)
    for t in range(1, T):
        s[t] = -s[t - 1] if rng.random() < switch else s[t - 1]
    z = s + 0.25 * rng.standard_normal(T)
    Y = np.zeros((T, 4), np.float32)
    Y[:, 0] = 0.4 * z
    Y[:, 1:] = rng.standard_normal((T, 3)).astype(np.float32)
    return Y


def _fitted(Y, lag=10, seed=0):
    flow = RealNVP(4, hidden=32, n_layers=6).fit(Y, epochs=40, verbose=False, seed=seed)
    enc = SoftStateEncoder(4, 3, hidden=16).fit_vamp(Y, lag=lag, epochs=60, seed=seed)
    for p in enc.parameters():
        p.requires_grad_(False)
    return flow, enc


def test_latent_rate_decreases_with_bin_width():
    torch.manual_seed(0)
    z = torch.randn(300, 4)
    r_fine = latent_rate_bits(z, torch.full((4,), -1.0))     # small step -> more bits
    r_coarse = latent_rate_bits(z, torch.full((4,), 1.0))    # large step -> fewer bits
    assert float(r_fine) > float(r_coarse) >= 0.0


def test_compressor_freezes_transform_and_round_trips():
    torch.manual_seed(0)
    Y = _system()
    flow, enc = _fitted(Y)
    comp = KineticRDCompressor(flow, enc, lag=10)
    CV_hat, bits = comp.compress_hard(torch.as_tensor(Y))
    assert CV_hat.shape == (len(Y), 4)
    assert bits >= 0.0
    # only the bit allocation is trainable; the flow and encoder stay frozen
    assert comp.log_width.requires_grad
    assert not any(p.requires_grad for p in comp.flow.parameters())
    assert not any(p.requires_grad for p in comp.encoder.parameters())


def test_train_runs_and_hard_eval_is_finite():
    torch.manual_seed(0)
    Y = _system()
    flow, enc = _fitted(Y)
    comp = KineticRDCompressor(flow, enc, lag=10)
    train_rd(comp, Y, lam=1e3, kind="kinetic", steps=40)
    assert torch.isfinite(comp.log_width).all()
    ev = hard_state_error(comp, Y, n_states=40, lag=10)
    assert np.isfinite(ev["folding_err_pct"])
    assert ev["rate_bits"] >= 0.0 and ev["t1_ref"] > 0.0


def test_mse_objective_also_runs():
    torch.manual_seed(0)
    Y = _system()
    flow, enc = _fitted(Y)
    comp = KineticRDCompressor(flow, enc, lag=10)
    train_rd(comp, Y, lam=1e2, kind="mse", steps=30)
    assert torch.isfinite(comp.log_width).all()


def test_rate_increases_with_lambda():
    # higher lambda weights the distortion, so more bits are spent and the rate rises
    Y = _system()
    flow, enc = _fitted(Y)
    pts = rate_distortion_curve(Y, lag=10, lambdas=[1.0, 1e4], kind="kinetic",
                                flow=flow, encoder=enc, n_states=40, steps=60)
    rates = [p["rate_bits"] for p in pts]
    assert rates[1] >= rates[0] - 0.5           # monotone up to optimization slack
