"""T7: rational-quadratic neural-spline flow. The property that matters for the bound
is EXACT invertibility (a diffeomorphism); the point of the upgrade is a better density
(>= RealNVP) -> a tighter KL/Pinsker bound. torch-gated."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from epc.spline_flow import SplineFlow
from epc.flow import RealNVP
from _synth import metastable_coords


def _three_wells(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 3, size=n)
    x0 = np.array([-2.0, 0.0, 2.0])[comp] + 0.35 * rng.standard_normal(n)
    x1 = 0.6 * rng.standard_normal(n)
    return np.stack([x0, x1], 1).astype(np.float32)


def test_spline_invertibility_before_and_after_training():
    torch.manual_seed(0)
    X = _three_wells(1500)
    flow = SplineFlow(2, hidden=32, n_layers=6, n_bins=8)
    xt = torch.as_tensor(X[:400])
    z, _ = flow.forward(xt)
    assert (flow.inverse(z) - xt).abs().max().item() < 1e-4
    flow.fit(X, epochs=30, batch=256, verbose=False)
    z, _ = flow.forward(xt)
    assert (flow.inverse(z) - xt).abs().max().item() < 1e-4


def test_spline_density_normalized_and_ordered():
    torch.manual_seed(0)
    X = _three_wells(4000)
    flow = SplineFlow(2, hidden=64, n_layers=8, n_bins=8).fit(
        X, epochs=120, lr=2e-3, batch=512, verbose=False)
    with torch.no_grad():
        lp_in = flow.log_prob(torch.as_tensor(X[:1000])).mean().item()
        rng = np.random.default_rng(1)
        oob = torch.as_tensor(rng.uniform([-5, -3], [5, 3], size=(1000, 2)),
                              dtype=torch.float32)
        lp_oob = flow.log_prob(oob).mean().item()
    assert lp_in > lp_oob + 2.0
    gx = np.linspace(-5, 5, 120); gy = np.linspace(-3, 3, 80)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.as_tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        dens = torch.exp(flow.log_prob(grid)).numpy()
    mass = dens.sum() * (gx[1] - gx[0]) * (gy[1] - gy[0])
    assert 0.8 < mass < 1.2


def test_spline_density_competitive_with_realnvp():
    X = _three_wells(5000, seed=2)
    torch.manual_seed(0)
    rn = RealNVP(2, hidden=64, n_layers=10).fit(X, epochs=150, lr=2e-3, batch=512, verbose=False)
    torch.manual_seed(0)
    sp = SplineFlow(2, hidden=64, n_layers=10, n_bins=8).fit(X, epochs=150, lr=2e-3, batch=512, verbose=False)
    with torch.no_grad():
        nll_rn = -rn.log_prob(torch.as_tensor(X)).mean().item()
        nll_sp = -sp.log_prob(torch.as_tensor(X)).mean().item()
    # the spline should be AT LEAST as good as the affine flow (tighter or comparable)
    assert nll_sp <= nll_rn + 0.10


def test_compress_flow_spline_end_to_end(tmp_path):
    from epc.runner import compress_trajectory
    from epc.artifact import save_artifact, load_artifact
    from epc.cli import main
    coords = metastable_coords(n_steps=1500, n_atoms=6, seed=0)
    art, rep = compress_trajectory([coords], flow_kind="spline", cv_dim=2, keep_frac=0.1,
                                   epochs=40, nstates=30, lag=10, seed=0, verbose=False)
    assert art.flow_kind == "spline"
    p = str(tmp_path / "s.epc")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=True)
    assert loaded.flow_kind == "spline"
    assert type(loaded.build_flow()).__name__ == "SplineFlow"
    out = str(tmp_path / "f.npy")
    main(["decompress", p, "-o", out, "--full-atom"])
    assert np.load(out).shape == (art.n_keep, 6, 3)
