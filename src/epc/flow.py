"""
epc_flow.py
===========
A normalizing flow (RealNVP) built from scratch in PyTorch -- the learned density
model at the heart of EPC. This is the piece the linear PCA-whitening was standing
in for.

Why a flow, precisely (this is the abstract's logic):
  - It is an EXACT diffeomorphism x <-> z, so it is invertible by construction
    (no architectural information loss), unlike a lossy autoencoder.
  - KL divergence is invariant under it, so a divergence bound measured in the
    Gaussian base space z transfers exactly to configuration space x -- the
    bound stops depending on a Gaussian-reference *assumption* (which raw MD
    violates) and becomes assumption-free.
  - It gives a tractable density log p(x), which (a) defines the information-gain
    signal for frame selection and (b) is the model the entropy coder codes
    against (coding cost = -log2 p, i.e. the NLL in bits).

Affine coupling layer (Dinh et al. 2017): split coordinates by a binary mask b.
The frozen half x_b conditions a network producing per-dim scale s and shift t;
the active half is transformed y = x*exp(s)+t. log|det J| = sum(s) over the
active dims; the inverse is closed-form. Masks alternate between layers so every
coordinate is transformed.

Standardization (x -> (x-mean)/std) is folded into the flow as a fixed affine
layer, so log_prob is a proper density over the original coordinates.

Tested in __main__: exact invertibility, training NLL on a 3-well mixture, the
density integrating to ~1 on a grid, and in- vs out-of-distribution log_prob.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)            # (dim,) of 0/1
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        # zero-init the last layer => the layer starts as the identity (stable start)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_frozen):
        s, t = self.net(x_frozen).chunk(2, dim=-1)
        comp = 1.0 - self.mask
        s = torch.tanh(s) * comp        # bounded scale, only on the active half
        t = t * comp
        return s, t

    def forward(self, x):
        xf = x * self.mask
        s, t = self._st(xf)
        y = xf + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(-1)             # logdet = sum of active-dim log-scales

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
        """x -> base z, returning (z, total logdet of x->z)."""
        z = (x - self.mean_) / self.std_
        logdet = -torch.log(self.std_).sum().expand(x.shape[0]).clone()
        for layer in self.layers:
            z, d = layer(z)
            logdet = logdet + d
        return z, logdet

    def inverse(self, z):
        """base z -> x (exact)."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x * self.std_ + self.mean_

    def log_prob(self, x):
        z, logdet = self.forward(x)
        base = -0.5 * (z ** 2 + math.log(2 * math.pi))   # standard normal, per dim
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

    # --- a known target: 3 wells along dim0 (like a slow CV), noise on dim1 ---
    n = 12000
    comp = rng.integers(0, 3, size=n)
    centers = np.array([-2.0, 0.0, 2.0])
    x0 = centers[comp] + 0.35 * rng.standard_normal(n)
    x1 = 0.6 * rng.standard_normal(n)
    X = np.stack([x0, x1], 1).astype(np.float32)

    flow = RealNVP(dim=2, hidden=64, n_layers=10)

    # (1) exact invertibility BEFORE training (identity-init) and AFTER
    xt = torch.as_tensor(X[:512])
    z, _ = flow.forward(xt)
    err0 = (flow.inverse(z) - xt).abs().max().item()

    print("training the flow on a 3-well density:")
    flow.fit(X, epochs=200, lr=2e-3, batch=512, verbose=True)

    z, _ = flow.forward(xt)
    err1 = (flow.inverse(z) - xt).abs().max().item()
    print("\nINVERTIBILITY  max|inverse(forward(x))-x|:  before=%.2e  after=%.2e"
          % (err0, err1))

    # (2) does the learned density integrate to ~1? (grid quadrature)
    gx = np.linspace(-5, 5, 200)
    gy = np.linspace(-3, 3, 120)
    XX, YY = np.meshgrid(gx, gy)
    grid = torch.as_tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        dens = torch.exp(flow.log_prob(grid)).numpy()
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    mass = dens.sum() * cell
    print("DENSITY normalization  integral p(x) dx ~ %.4f  (should be ~1)" % mass)

    # (3) in- vs out-of-distribution log-prob
    with torch.no_grad():
        lp_in = flow.log_prob(torch.as_tensor(X[:2000])).mean().item()
        oob = torch.as_tensor(rng.uniform([-5, -3], [5, 3], size=(2000, 2)),
                              dtype=torch.float32)
        lp_oob = flow.log_prob(oob).mean().item()
    print("LOG-PROB  in-distribution=%.3f   uniform-random=%.3f   (in >> oob)"
          % (lp_in, lp_oob))

    # (4) recovered well structure from samples
    with torch.no_grad():
        s = flow.sample(20000).numpy()
    frac = [(np.abs(s[:, 0] - c) < 1.0).mean() for c in centers]
    print("SAMPLES  fraction near each well (-2/0/+2): %s  (target ~0.33 each)"
          % np.round(frac, 3))
