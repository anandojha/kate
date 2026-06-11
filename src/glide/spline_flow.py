"""
spline_flow.py
==============
T7 -- a more expressive normalizing flow: rational-quadratic NEURAL SPLINE coupling
(Durkan et al., "Neural Spline Flows", 2019) reimplemented from the method. A monotonic
piecewise rational-quadratic transform is far more flexible than RealNVP's affine
coupling, so it fits the latent density better -> a tighter KL and therefore a tighter
Pinsker bound -- with EXACTLY the same invertibility (the spline is monotone by
construction, with closed-form inverse). RealNVP stays the reproducible default
(flow.RealNVP); SplineFlow is a drop-in alternative (same forward/inverse/log_prob/
sample/fit interface) selected by `glide compress --flow spline`.

Neural spline flows / MAF / equivariant flows are PRIOR ART -- cited, not claimed; the
architecture is NOT the headline (the kinetic bound is). The optional ``nflows`` extra
provides a reference implementation for cross-checking. (Stretch, not implemented here:
an SE(3)/permutation-equivariant coupling conditioner -- a GNN, not a grid CNN -- so the
density respects molecular symmetry.)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MIN_BIN = 1e-3
_MIN_DERIV = 1e-3


def _searchsorted(bin_locations, inputs):
    return torch.sum(inputs[..., None] >= bin_locations[..., :-1], dim=-1) - 1


def _rqs(inputs, uw, uh, ud, inverse, tail_bound):
    """Rational-quadratic spline on [-tb, tb] (Durkan et al.). inputs: (...,);
    uw/uh/ud: (..., K)/(..., K)/(..., K-1). Returns (outputs, logabsdet)."""
    tb = tail_bound
    K = uw.shape[-1]
    # widths -> x-knots
    widths = F.softmax(uw, dim=-1)
    widths = _MIN_BIN + (1 - _MIN_BIN * K) * widths
    cumw = torch.cumsum(widths, dim=-1)
    cumw = F.pad(cumw, (1, 0), value=0.0) * (2 * tb) - tb
    cumw[..., 0] = -tb; cumw[..., -1] = tb
    widths = cumw[..., 1:] - cumw[..., :-1]
    # heights -> y-knots
    heights = F.softmax(uh, dim=-1)
    heights = _MIN_BIN + (1 - _MIN_BIN * K) * heights
    cumh = torch.cumsum(heights, dim=-1)
    cumh = F.pad(cumh, (1, 0), value=0.0) * (2 * tb) - tb
    cumh[..., 0] = -tb; cumh[..., -1] = tb
    heights = cumh[..., 1:] - cumh[..., :-1]
    # derivatives at the K+1 knots: pad the raw inner derivatives with the boundary
    # constant so the boundary derivative is exactly 1 (C1 linear tails), then softplus.
    const = math.log(math.exp(1 - _MIN_DERIV) - 1)
    ud = F.pad(ud, (1, 1), value=const)
    derivs = _MIN_DERIV + F.softplus(ud)

    knots = cumh if inverse else cumw
    bin_idx = _searchsorted(knots, inputs).clamp(0, K - 1)[..., None]
    in_cumw = cumw.gather(-1, bin_idx)[..., 0]
    in_w = widths.gather(-1, bin_idx)[..., 0]
    in_cumh = cumh.gather(-1, bin_idx)[..., 0]
    in_h = heights.gather(-1, bin_idx)[..., 0]
    delta = (heights / widths).gather(-1, bin_idx)[..., 0]
    d_k = derivs.gather(-1, bin_idx)[..., 0]
    d_k1 = derivs[..., 1:].gather(-1, bin_idx)[..., 0]

    if inverse:
        a = ((inputs - in_cumh) * (d_k + d_k1 - 2 * delta) + in_h * (delta - d_k))
        b = (in_h * d_k - (inputs - in_cumh) * (d_k + d_k1 - 2 * delta))
        c = -delta * (inputs - in_cumh)
        disc = (b ** 2 - 4 * a * c).clamp_min(0.0)
        theta = (2 * c) / (-b - torch.sqrt(disc))
        outputs = theta * in_w + in_cumw
        tomt = theta * (1 - theta)
        denom = delta + (d_k + d_k1 - 2 * delta) * tomt
        dnum = delta ** 2 * (d_k1 * theta ** 2 + 2 * delta * tomt + d_k * (1 - theta) ** 2)
        logabsdet = torch.log(dnum) - 2 * torch.log(denom)
        return outputs, -logabsdet
    else:
        theta = ((inputs - in_cumw) / in_w).clamp(0.0, 1.0)
        tomt = theta * (1 - theta)
        num = in_h * (delta * theta ** 2 + d_k * tomt)
        denom = delta + (d_k + d_k1 - 2 * delta) * tomt
        outputs = in_cumh + num / denom
        dnum = delta ** 2 * (d_k1 * theta ** 2 + 2 * delta * tomt + d_k * (1 - theta) ** 2)
        logabsdet = torch.log(dnum) - 2 * torch.log(denom)
        return outputs, logabsdet


def _unconstrained_rqs(inputs, uw, uh, ud, inverse, tail_bound):
    """RQ spline inside [-tb, tb], identity (linear) tails outside."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    out = torch.where(inside, torch.zeros_like(inputs), inputs)
    lad = torch.zeros_like(inputs)
    if inside.any():
        o, l = _rqs(inputs[inside], uw[inside], uh[inside], ud[inside], inverse, tail_bound)
        out = out.clone(); lad = lad.clone()
        out[inside] = o
        lad[inside] = l
    return out, lad


class RQSplineCoupling(nn.Module):
    """Mask-based coupling: the frozen half conditions a net producing RQ-spline
    parameters for the active half. Invertible by construction."""

    def __init__(self, dim, hidden, mask, num_bins=8, tail_bound=5.0):
        super().__init__()
        self.register_buffer("mask", mask)
        self.dim = dim
        self.K = num_bins
        self.tail_bound = tail_bound
        n_params = 3 * num_bins - 1                    # K widths, K heights, K-1 derivs
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * n_params),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x_frozen):
        p = self.net(x_frozen).view(-1, self.dim, 3 * self.K - 1)
        return p[..., :self.K], p[..., self.K:2 * self.K], p[..., 2 * self.K:]

    def _apply_spline(self, x, inverse):
        xf = x * self.mask
        uw, uh, ud = self._params(xf)
        y_all, lad_all = _unconstrained_rqs(x, uw, uh, ud, inverse, self.tail_bound)
        active = 1.0 - self.mask
        y = self.mask * x + active * y_all
        logdet = (active * lad_all).sum(-1)
        return y, logdet

    def forward(self, x):
        return self._apply_spline(x, inverse=False)

    def inverse(self, y):
        return self._apply_spline(y, inverse=True)[0]


class SplineFlow(nn.Module):
    """Drop-in replacement for flow.RealNVP using RQ-spline coupling layers. Same
    interface (forward/inverse/log_prob/sample/fit) so codec/runner use it unchanged."""

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 8, n_bins: int = 8):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            m = (torch.arange(dim) % 2).float()
            if i % 2 == 0:
                m = 1.0 - m
            layers.append(RQSplineCoupling(dim, hidden, m, num_bins=n_bins))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("mean_", torch.zeros(dim))
        self.register_buffer("std_", torch.ones(dim))

    def forward(self, x):
        z = (x - self.mean_) / self.std_
        logdet = -torch.log(self.std_).sum().expand(x.shape[0]).clone()
        for layer in self.layers:
            z, d = layer(z)
            logdet = logdet + d
        return z, logdet

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x * self.std_ + self.mean_

    def log_prob(self, x):
        z, logdet = self.forward(x)
        base = -0.5 * (z ** 2 + math.log(2 * math.pi))
        return base.sum(-1) + logdet

    @torch.no_grad()
    def sample(self, n):
        return self.inverse(torch.randn(n, self.dim))

    def fit(self, X, epochs: int = 200, lr: float = 1e-3, batch: int = 256,
            weight_decay: float = 0.0, verbose: bool = True, seed: int = 0):
        torch.manual_seed(seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            self.mean_.copy_(X.mean(0))
            self.std_.copy_(X.std(0) + 1e-6)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        n = X.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = -self.log_prob(X[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print("  spline epoch %4d  NLL/dim = %.4f" % (ep, tot / n / self.dim))
        return self


if __name__ == "__main__":
    # quick invertibility + density self-test (mirrors glide.flow)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 6000
    comp = rng.integers(0, 3, size=n)
    x0 = np.array([-2., 0., 2.])[comp] + 0.35 * rng.standard_normal(n)
    x1 = 0.6 * rng.standard_normal(n)
    X = np.stack([x0, x1], 1).astype(np.float32)
    flow = SplineFlow(2, hidden=64, n_layers=8, n_bins=8)
    xt = torch.as_tensor(X[:512])
    z, _ = flow.forward(xt)
    print("INVERTIBILITY before train: %.2e" % (flow.inverse(z) - xt).abs().max().item())
    flow.fit(X, epochs=150, lr=2e-3, batch=512, verbose=True)
    z, _ = flow.forward(xt)
    print("INVERTIBILITY after train : %.2e" % (flow.inverse(z) - xt).abs().max().item())
