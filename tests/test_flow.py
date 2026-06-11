"""
RealNVP normalizing-flow tests (torch-gated). The flow is the learned density at
the heart of GLIDE; the property that matters for the bound is EXACT invertibility
(a diffeomorphism), plus a sane learned density.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from glide.flow import RealNVP  # noqa: E402


def _three_wells(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 3, size=n)
    centers = np.array([-2.0, 0.0, 2.0])
    x0 = centers[comp] + 0.35 * rng.standard_normal(n)
    x1 = 0.6 * rng.standard_normal(n)
    return np.stack([x0, x1], 1).astype(np.float32)


def test_invertibility_before_and_after_training():
    torch.manual_seed(0)
    X = _three_wells(1500)
    flow = RealNVP(dim=2, hidden=32, n_layers=8)
    xt = torch.as_tensor(X[:400])
    z, _ = flow.forward(xt)
    err0 = (flow.inverse(z) - xt).abs().max().item()
    assert err0 < 1e-5                       # identity-init is exactly invertible
    flow.fit(X, epochs=30, batch=256, verbose=False)
    z, _ = flow.forward(xt)
    err1 = (flow.inverse(z) - xt).abs().max().item()
    assert err1 < 1e-3                       # stays invertible after training


def test_density_and_log_prob_ordering():
    torch.manual_seed(0)
    X = _three_wells(4000)
    flow = RealNVP(dim=2, hidden=64, n_layers=10)
    flow.fit(X, epochs=120, lr=2e-3, batch=512, verbose=False)
    # in-distribution log-prob much higher than far out-of-distribution
    with torch.no_grad():
        lp_in = flow.log_prob(torch.as_tensor(X[:1000])).mean().item()
        rng = np.random.default_rng(1)
        oob = torch.as_tensor(rng.uniform([-5, -3], [5, 3], size=(1000, 2)),
                              dtype=torch.float32)
        lp_oob = flow.log_prob(oob).mean().item()
    assert lp_in > lp_oob + 2.0
    # crude density normalization on a grid ~ 1
    gx = np.linspace(-5, 5, 120); gy = np.linspace(-3, 3, 80)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.as_tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        dens = torch.exp(flow.log_prob(grid)).numpy()
    mass = dens.sum() * (gx[1] - gx[0]) * (gy[1] - gy[0])
    assert 0.8 < mass < 1.2
