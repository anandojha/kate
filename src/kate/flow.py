"""
Normalizing Flow Density Model (RealNVP)
========================================

Background
----------
This module implements a RealNVP normalizing flow in PyTorch, providing the
learned density model used by KATE. It replaces the linear PCA-whitening
employed in earlier stages.

A normalizing flow is an exact diffeomorphism x <-> z and is therefore invertible
by construction, incurring no architectural information loss, in contrast to a
lossy autoencoder. The Kullback-Leibler divergence is invariant under such a
transformation, so a divergence bound measured in the Gaussian base space z
transfers exactly to configuration space x. The bound consequently no longer
depends on a Gaussian-reference assumption, which raw molecular-dynamics data
violate, and instead becomes assumption-free. The flow additionally provides a
tractable density log p(x), which defines the information-gain signal used for
frame selection and serves as the model against which the entropy coder operates,
the coding cost being -log2 p(x), the negative log-likelihood expressed in bits.

Affine coupling
---------------
The affine coupling layer (RealNVP: Dinh et al., ICLR 2017) partitions the
coordinates according to a binary mask b. The frozen half x_b conditions a network
that produces a per-dimension scale s and shift t; the active half is transformed
as y = x * exp(s) + t. The Jacobian log-determinant is log|det J| = sum(s) over
the active dimensions, and the inverse is available in closed form. The masks
alternate between successive layers so that every coordinate is transformed.

Standardization, x -> (x - mean) / std, is folded into the flow as a fixed affine
layer, so that log_prob is a proper density over the original coordinates.

The self-test in __main__ verifies exact invertibility, the training negative
log-likelihood on a three-well mixture, integration of the density to
approximately unity on a grid, and the contrast between in-distribution and
out-of-distribution log_prob values.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)            # binary mask of shape (dim,)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        # Zero-initialize the final layer so the layer begins as the identity map,
        # which provides a numerically stable starting point.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_frozen):
        s, t = self.net(x_frozen).chunk(2, dim=-1)
        comp = 1.0 - self.mask
        s = torch.tanh(s) * comp        # bounded scale applied to the active half
        t = t * comp
        return s, t

    def forward(self, x):
        xf = x * self.mask
        s, t = self._st(xf)
        y = xf + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(-1)             # log|det J| is the sum of active-dim log-scales

    def inverse(self, y):
        yf = y * self.mask
        s, t = self._st(yf)
        return yf + (1.0 - self.mask) * ((y - t) * torch.exp(-s))


class RealNVP(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 8):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            m = (torch.arange(dim) % 2).float()
            if i % 2 == 0:
                m = 1.0 - m
            layers.append(AffineCoupling(dim, hidden, m))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("mean_", torch.zeros(dim))
        self.register_buffer("std_", torch.ones(dim))

    # ----- change of variables -----
    def forward(self, x):
        """Map x to the base variable z and return (z, total log|det J| of x -> z)."""
        z = (x - self.mean_) / self.std_
        logdet = -torch.log(self.std_).sum().expand(x.shape[0]).clone()
        for layer in self.layers:
            z, d = layer(z)
            logdet = logdet + d
        return z, logdet

    def inverse(self, z):
        """Map the base variable z back to x exactly."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x * self.std_ + self.mean_

    def log_prob(self, x):
        """Evaluate log p(x) via the change-of-variables identity.

        The density follows from log p(x) = log p(z) + log|det J|, where p(z) is the
        standard-normal base density and log|det J| is the accumulated Jacobian
        log-determinant of the forward map.
        """
        z, logdet = self.forward(x)
        base = -0.5 * (z ** 2 + math.log(2 * math.pi))   # standard normal, per dimension
        return base.sum(-1) + logdet

    @torch.no_grad()
    def sample(self, n):
        return self.inverse(torch.randn(n, self.dim))

    # ----- training -----
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
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print("  epoch %4d  NLL/dim = %.4f" % (ep, tot / n / self.dim))
        return self


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Known target: three wells along dimension 0, analogous to a slow collective
    # variable, with Gaussian noise on dimension 1.
    n = 12000
    comp = rng.integers(0, 3, size=n)
    centers = np.array([-2.0, 0.0, 2.0])
    x0 = centers[comp] + 0.35 * rng.standard_normal(n)
    x1 = 0.6 * rng.standard_normal(n)
    X = np.stack([x0, x1], 1).astype(np.float32)

    flow = RealNVP(dim=2, hidden=64, n_layers=10)

    # (1) Exact invertibility before training (identity initialization) and after.
    xt = torch.as_tensor(X[:512])
    z, _ = flow.forward(xt)
    err0 = (flow.inverse(z) - xt).abs().max().item()

    print("training the flow on a 3-well density:")
    flow.fit(X, epochs=200, lr=2e-3, batch=512, verbose=True)

    z, _ = flow.forward(xt)
    err1 = (flow.inverse(z) - xt).abs().max().item()
    print("\nINVERTIBILITY  max|inverse(forward(x))-x|:  before=%.2e  after=%.2e"
          % (err0, err1))

    # (2) Verify that the learned density integrates to approximately unity by grid
    # quadrature.
    gx = np.linspace(-5, 5, 200)
    gy = np.linspace(-3, 3, 120)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.as_tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        dens = torch.exp(flow.log_prob(grid)).numpy()
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    mass = dens.sum() * cell
    print("DENSITY normalization  integral p(x) dx ~ %.4f  (should be ~1)" % mass)

    # (3) In-distribution versus out-of-distribution log-probability.
    with torch.no_grad():
        lp_in = flow.log_prob(torch.as_tensor(X[:2000])).mean().item()
        oob = torch.as_tensor(rng.uniform([-5, -3], [5, 3], size=(2000, 2)),
                              dtype=torch.float32)
        lp_oob = flow.log_prob(oob).mean().item()
    print("LOG-PROB  in-distribution=%.3f   uniform-random=%.3f   (in >> oob)"
          % (lp_in, lp_oob))

    # (4) Recovery of the well structure from samples.
    with torch.no_grad():
        s = flow.sample(20000).numpy()
    frac = [(np.abs(s[:, 0] - c) < 1.0).mean() for c in centers]
    print("SAMPLES  fraction near each well (-2/0/+2): %s  (target ~0.33 each)"
          % np.round(frac, 3))
