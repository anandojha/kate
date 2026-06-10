"""
predictive_coder.py
===================
T9 -- learned PREDICTIVE temporal entropy coding (LOSSY), positioned as the novel-ML
contribution. T8 (temporal_prior) codes the flow latents LOSSLESSLY against a learned
causal conditional. T9 adds a LOSSY rate-distortion mode and is built on three decisions:

  D1 -- predictor: a CAUSAL GRU over the flow latents outputs, per step, a conditional
        Gaussian (mu_t, log sigma_t) for the NEXT latent z_{t+1} given the reconstructed
        past. GRU (not a transformer) so it stays STREAMING-compatible with T5 (online
        hidden state, no full-context window). A causal-TCN predictor is available via
        ``kind="tcn"``.
  D2 -- objective (bound-as-loss, NOT coordinate MSE): the predictor is trained on the
        conditional NLL of the next latent -- a tractable SURROGATE for the
        TRANSITION-kernel term of the path bound. (The ENSEMBLE-term distortion is
        realized at coding time by the quantizer; the rate-distortion curve sweeps it.)
        Because the NLL is only a surrogate, the GATE measures the TRUE observable error
        (CV-histogram KL, MSM implied-timescale error) and checks it against the
        path-space Pinsker bound -- it is NOT assumed from the loss.
  D3 -- residual coding: the STANDARDIZED innovation u = (z - mu)/sigma is quantized
        with subtractive dithering and entropy-coded against a UNIT Gaussian; the
        quantizer bit-width is the rate knob. Standardizing by the predicted scale is
        the point -- the coded residual is ~unit-Gaussian regardless of frame, so where
        the predictor is confident (small sigma) a given bit-width buys finer z-fidelity
        -> fewer bits at equal distortion. T8's lossless coder is retained (it provides
        the head-to-head baseline; the lossy-over-lossless gain is measured on the same
        latents).

Coding is CLOSED-LOOP (DPCM): both encoder and decoder feed the RECONSTRUCTED latents
into the GRU, so they predict identically; decode is strictly frame-by-frame.

Prior art (predictive / context-model entropy coding) -- CITE, do not claim:
  * DPCM (differential pulse-code modulation): Cutler, 1952.
  * Learned scale-hyperprior entropy model: Balle, Minnen, Singh, Hwang, Johnston,
    "Variational image compression with a scale hyperprior", ICLR 2018 (arXiv:1802.01436).
  * Autoregressive/context entropy model: Minnen, Balle, Toderici, "Joint Autoregressive
    and Hierarchical Priors for Learned Image Compression", NeurIPS 2018 (arXiv:1809.02736).
  VERIFY exact citations before the paper. The GRU/flow/CV are MACHINERY; the
  contribution is the learned conditional entropy model + bound-as-loss + the integration.

HONESTY: T9's rate gain over T8 is EMPIRICAL, measured on the trypsin-benzamidine set,
reported against T8 -- never assumed. If T9 does not dominate T8 at equal observable
error, that is reported as-is.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .codec import gaussian_cumfreq
from .temporal_prior import _RangeEncoder, _RangeDecoder, _CausalConv1d


# ============================================================================
# 1. predictors: causal GRU (default) and causal TCN, both predicting z_{t+1}
#    from z_{<=t}. Interface: forward (teacher-forced) + advance (closed-loop).
# ============================================================================
class CausalGRUPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 1):
        super().__init__()
        self.dim, self.hidden, self.n_layers = dim, hidden, n_layers
        self.gru = nn.GRU(dim, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 2 * dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        """z: (B, T, dim) -> (mu, log_sigma), each (B, T, dim). mu[:, t] predicts
        z[:, t+1] (causal: depends only on z[:, 0..t])."""
        o, _ = self.gru(z)
        mu, log_s = self.head(o).chunk(2, dim=-1)
        return mu, torch.tanh(log_s) * 3.0

    def init_state(self):
        return torch.zeros(self.n_layers, 1, self.hidden)

    @torch.no_grad()
    def advance(self, h, z_t):
        """Feed the (reconstructed) latent z_t and return the prediction (mu, log_s)
        for z_{t+1} plus the new hidden state. Deterministic (eval, CPU)."""
        self.eval()
        zt = torch.as_tensor(np.asarray(z_t), dtype=torch.float32).view(1, 1, self.dim)
        o, h = self.gru(zt, h)
        out = self.head(o)[0, 0]
        mu = out[:self.dim].double().numpy()
        log_s = (torch.tanh(out[self.dim:]) * 3.0).double().numpy()
        return mu, log_s, h

    def fit(self, z, epochs=200, lr=1e-3, weight_decay=0.0, verbose=False, seed=0):
        """Train on the conditional NLL (the transition surrogate) -- NOT MSE."""
        torch.manual_seed(seed)
        Z = torch.as_tensor(np.asarray(z), dtype=torch.float32).unsqueeze(0)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(epochs):
            mu, log_s = self.forward(Z)
            tgt, pmu, plog = Z[:, 1:], mu[:, :-1], log_s[:, :-1]
            nll = (0.5 * ((tgt - pmu) ** 2) * torch.exp(-2 * plog) + plog
                   + 0.5 * math.log(2 * math.pi)).mean()
            opt.zero_grad(); nll.backward(); opt.step()
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print("  GRU epoch %4d  cond-NLL/dim = %.4f" % (ep, nll.item()))
        return self


class CausalTCNPredictor(nn.Module):
    """Causal dilated-CNN predictor (the TCN swap behind ``kind='tcn'``). Same train/
    advance interface as the GRU; advance re-runs the net on the reconstructed prefix
    (O(T*receptive-field)). Streaming via a fixed-length tail buffer."""

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 3, kernel: int = 3):
        super().__init__()
        self.dim = dim
        layers, in_ch, d = [], dim, 1
        for _ in range(n_layers):
            layers += [_CausalConv1d(in_ch, hidden, kernel, d), nn.ReLU()]
            in_ch, d = hidden, d * 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 2 * dim, 1)
        self.rf = 1 + (kernel - 1) * (2 ** n_layers - 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        x = z.transpose(1, 2)                       # (B, dim, T); causal, no shift
        out = self.head(self.body(x)).transpose(1, 2)
        mu, log_s = out.chunk(2, dim=-1)            # mu[:, t] predicts z[:, t+1]
        return mu, torch.tanh(log_s) * 3.0

    def init_state(self):
        return np.zeros((0, self.dim))              # the reconstructed-prefix buffer

    @torch.no_grad()
    def advance(self, buf, z_t):
        self.eval()
        buf = np.concatenate([buf, np.asarray(z_t, float)[None]], axis=0)[-self.rf:]
        Z = torch.as_tensor(buf, dtype=torch.float32).unsqueeze(0)
        out = self.head(self.body(Z.transpose(1, 2))).transpose(1, 2)[0, -1]
        mu = out[:self.dim].double().numpy()
        log_s = (torch.tanh(out[self.dim:]) * 3.0).double().numpy()
        return mu, log_s, buf

    def fit(self, z, epochs=200, lr=1e-3, weight_decay=0.0, verbose=False, seed=0):
        torch.manual_seed(seed)
        Z = torch.as_tensor(np.asarray(z), dtype=torch.float32).unsqueeze(0)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(epochs):
            mu, log_s = self.forward(Z)
            tgt, pmu, plog = Z[:, 1:], mu[:, :-1], log_s[:, :-1]
            nll = (0.5 * ((tgt - pmu) ** 2) * torch.exp(-2 * plog) + plog
                   + 0.5 * math.log(2 * math.pi)).mean()
            opt.zero_grad(); nll.backward(); opt.step()
        return self


def make_predictor(dim, kind="gru", hidden=64, n_layers=None):
    if kind == "tcn":
        return CausalTCNPredictor(dim, hidden=hidden, n_layers=n_layers or 3)
    return CausalGRUPredictor(dim, hidden=hidden, n_layers=n_layers or 1)


def conditional_nll(model, z):
    """Mean conditional NLL (nats/value) of z_{t+1} under the predictor -- the
    transition surrogate. Lower than the static-N(0,1) NLL means the predictor exploits
    inter-frame structure (what a static prior cannot)."""
    model.eval()
    with torch.no_grad():
        Z = torch.as_tensor(np.asarray(z), dtype=torch.float32).unsqueeze(0)
        mu, log_s = model.forward(Z)
        tgt, pmu, plog = Z[:, 1:], mu[:, :-1], log_s[:, :-1]
        return float((0.5 * ((tgt - pmu) ** 2) * torch.exp(-2 * plog) + plog
                      + 0.5 * math.log(2 * math.pi)).mean())


def static_gaussian_nll(z):
    """Mean NLL of z under a STATIC N(0,1) prior (nats/value) -- the no-predictor floor."""
    z = np.asarray(z, float)
    return float((0.5 * z ** 2 + 0.5 * math.log(2 * math.pi)).mean())


# ============================================================================
# 2. closed-loop standardized-innovation codec (LOSSY; bit-width = rate knob)
# ============================================================================
def _dither(T, dim, step, seed):
    return np.random.default_rng(seed).uniform(-step / 2, step / 2, size=(T, dim))


def encode_predictive(z, model, bits, U=8.0, seed=0):
    """Closed-loop DPCM encode of the latent sequence z (T, dim). Quantizes the
    STANDARDIZED innovation at `bits` bits, codes it against a unit Gaussian. Returns
    (coded_bytes, z_reconstructed, coded_levels)."""
    z = np.asarray(z, dtype=np.float64)
    T, dim = z.shape
    L = 1 << bits
    step = 2 * U / L
    cum = gaussian_cumfreq(L, U)
    dith = _dither(T, dim, step, seed)
    enc = _RangeEncoder()
    h = model.init_state()
    mu = np.zeros(dim); log_s = np.zeros(dim)        # z_0 prior: N(0,1)
    zhat = np.zeros((T, dim)); levels = np.zeros((T, dim), dtype=np.int64)
    for t in range(T):
        sigma = np.exp(log_s)
        u = (z[t] - mu) / sigma
        lev = np.clip(np.floor((np.clip(u + dith[t], -U, U) + U) / (2 * U) * L)
                      .astype(np.int64), 0, L - 1)
        for j in range(dim):
            enc.encode(int(lev[j]), cum)
        uhat = -U + (lev + 0.5) * step - dith[t]
        zhat[t] = mu + sigma * uhat
        levels[t] = lev
        mu, log_s, h = model.advance(h, zhat[t])     # feed the RECONSTRUCTION
    return enc.finish(), zhat, levels


def decode_predictive(data, T, dim, model, bits, U=8.0, seed=0):
    """Inverse of encode_predictive: decode strictly frame-by-frame, feeding each
    reconstructed latent back into the GRU. Returns (z_reconstructed, levels)."""
    L = 1 << bits
    step = 2 * U / L
    cum = gaussian_cumfreq(L, U)
    dith = _dither(T, dim, step, seed)
    dec = _RangeDecoder(data)
    h = model.init_state()
    mu = np.zeros(dim); log_s = np.zeros(dim)
    zhat = np.zeros((T, dim)); levels = np.zeros((T, dim), dtype=np.int64)
    for t in range(T):
        sigma = np.exp(log_s)
        lev = np.array([dec.decode(cum) for _ in range(dim)], dtype=np.int64)
        uhat = -U + (lev + 0.5) * step - dith[t]
        zhat[t] = mu + sigma * uhat
        levels[t] = lev
        mu, log_s, h = model.advance(h, zhat[t])
    return zhat, levels


# ============================================================================
# 3. rate-distortion accounting (T9) + the head-to-head with T8 (lossless)
# ============================================================================
def rate_distortion_curve(z, model, bits_list, U=8.0, seed=0):
    """For each bit-width: (bits, rate bits/value, latent MSE distortion). The latent
    MSE is the ENSEMBLE-term distortion proxy; the true observable error (CV-KL, ITS
    error) is measured by the gate on real data."""
    z = np.asarray(z, float)
    n = z.shape[0] * z.shape[1]
    out = []
    for b in bits_list:
        data, zhat, _ = encode_predictive(z, model, b, U=U, seed=seed)
        out.append({"bits": int(b), "rate_bpv": 8.0 * len(data) / n,
                    "latent_mse": float(np.mean((zhat - z) ** 2))})
    return out
