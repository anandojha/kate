"""
Learned predictive temporal entropy coding of the flow latents (KATE stage T9).

Stage T8 codes the flow latents losslessly against a learned causal conditional;
T9 keeps that machinery and adds a lossy rate-distortion mode. A causal predictor
reads the reconstructed latent history z_{<=t} and emits a conditional Gaussian
N(mu_t, sigma_t) for the next latent z_{t+1}. The mean and log-scale come from a
GRU rather than a transformer, so the predictor stays streaming-compatible with
the T5 online hidden state and needs no full-context window; a causal
temporal-convolutional predictor is available through kind="tcn".

The predictor is trained on the conditional negative log-likelihood of the next
latent, not on coordinate mean-squared error. The conditional NLL is a tractable
surrogate for the transition-kernel term of the path-space bound. The ensemble
term of that bound is realized as distortion at coding time by the quantizer, and
the rate-distortion curve sweeps it. Because the NLL is only a surrogate, the gate
does not read distortion off the training loss: it measures the true observable
error (collective-variable histogram KL, MSM implied-timescale error) and checks
it against the path-space Pinsker bound.

Coding runs on the standardized innovation u = (z - mu)/sigma, quantized at a
chosen bit-width with subtractive dithering and entropy-coded against a unit
Gaussian; the bit-width is the rate parameter. Standardizing by the predicted
scale makes the coded residual approximately unit-Gaussian frame to frame, so
where the predictor is confident (small sigma) a fixed bit-width resolves z more
finely and thus spends fewer bits at equal distortion. The loop is closed
(differential pulse-code modulation, Cutler 1952): encoder and decoder both feed
the reconstructed latents into the predictor, so they predict identically and
decoding proceeds strictly frame by frame.

The predictive and context-model entropy coders are prior art and are cited, not
claimed: differential pulse-code modulation (Cutler 1952); the learned
scale-hyperprior entropy model (Balle, Minnen, Singh, Hwang, Johnston, ICLR 2018,
arXiv:1802.01436); and the autoregressive context model (Minnen, Balle, Toderici,
NeurIPS 2018, arXiv:1809.02736). The contribution is the conditional entropy
model, the bound-as-loss objective, and their integration into the codec. The rate
gain of T9 over T8 is measured on the trypsin-benzamidine set and reported against
T8; if T9 does not dominate T8 at equal observable error, that is reported as
observed.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .codec import gaussian_cumfreq
from .temporal_prior import _RangeEncoder, _RangeDecoder, _CausalConv1d


# Predictors over the flow latents: a causal GRU (default) and a causal TCN, each
# predicting z_{t+1} from z_{<=t}, with a forward (teacher-forced) and an advance
# (closed-loop) interface.
class CausalGRUPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 1):
        super().__init__()
        self.dim, self.hidden, self.n_layers = dim, hidden, n_layers
        self.gru = nn.GRU(dim, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 2 * dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        """Map z (B, T, dim) to (mu, log_sigma), each (B, T, dim).

        The mean mu[:, t] predicts z[:, t+1] causally, depending only on
        z[:, 0..t].
        """
        o, _ = self.gru(z)
        mu, log_s = self.head(o).chunk(2, dim=-1)
        return mu, torch.tanh(log_s) * 3.0

    def init_state(self):
        return torch.zeros(self.n_layers, 1, self.hidden)

    @torch.no_grad()
    def advance(self, h, z_t):
        """Advance the closed loop by one step on the reconstructed latent z_t.

        The prediction is deterministic (eval mode, CPU).

        Returns
        -------
        tuple
            (mu, log_s) for z_{t+1} and the updated hidden state.
        """
        self.eval()
        zt = torch.as_tensor(np.asarray(z_t), dtype=torch.float32).view(1, 1, self.dim)
        o, h = self.gru(zt, h)
        out = self.head(o)[0, 0]
        mu = out[:self.dim].double().numpy()
        log_s = (torch.tanh(out[self.dim:]) * 3.0).double().numpy()
        return mu, log_s, h

    def fit(self, z, epochs=200, lr=1e-3, weight_decay=0.0, verbose=False, seed=0):
        """Train on the conditional NLL (the transition surrogate), not on MSE.

        Returns the fitted model.
        """
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
    """Causal dilated-CNN predictor selected by ``kind='tcn'``.

    The training and advance interface matches the GRU. The advance step re-runs
    the network on the reconstructed prefix, with cost O(T x receptive_field), and
    streaming is supported through a fixed-length tail buffer.
    """

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
        return np.zeros((0, self.dim))              # reconstructed-prefix buffer

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
    """Compute the mean conditional NLL (nats/value) of z_{t+1} under the predictor.

    This is the transition surrogate. A value below the static N(0, 1) NLL
    indicates that the predictor exploits inter-frame structure that a static
    prior cannot capture.
    """
    model.eval()
    with torch.no_grad():
        Z = torch.as_tensor(np.asarray(z), dtype=torch.float32).unsqueeze(0)
        mu, log_s = model.forward(Z)
        tgt, pmu, plog = Z[:, 1:], mu[:, :-1], log_s[:, :-1]
        return float((0.5 * ((tgt - pmu) ** 2) * torch.exp(-2 * plog) + plog
                      + 0.5 * math.log(2 * math.pi)).mean())


def static_gaussian_nll(z):
    """Compute the mean NLL of z under a static N(0, 1) prior (nats/value).

    This is the no-predictor floor.
    """
    z = np.asarray(z, float)
    return float((0.5 * z ** 2 + 0.5 * math.log(2 * math.pi)).mean())


# Closed-loop standardized-innovation codec. The quantizer bit-width is the rate
# parameter that trades bits against latent fidelity.
def _dither(T, dim, step, seed):
    return np.random.default_rng(seed).uniform(-step / 2, step / 2, size=(T, dim))


def encode_predictive(z, model, bits, U=8.0, seed=0):
    """Closed-loop DPCM encode of the latent sequence z (T, dim).

    The standardized innovation is quantized at ``bits`` bits and coded against a
    unit Gaussian.

    Returns
    -------
    tuple
        (coded_bytes, z_reconstructed, coded_levels).
    """
    z = np.asarray(z, dtype=np.float64)
    T, dim = z.shape
    L = 1 << bits
    step = 2 * U / L
    cum = gaussian_cumfreq(L, U)
    dith = _dither(T, dim, step, seed)
    enc = _RangeEncoder()
    h = model.init_state()
    mu = np.zeros(dim); log_s = np.zeros(dim)        # z_0 prior: N(0, 1)
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
        mu, log_s, h = model.advance(h, zhat[t])     # feed the reconstruction
    return enc.finish(), zhat, levels


def decode_predictive(data, T, dim, model, bits, U=8.0, seed=0):
    """Decode the inverse of encode_predictive strictly frame by frame.

    Each reconstructed latent is fed back into the GRU.

    Returns
    -------
    tuple
        (z_reconstructed, levels).
    """
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


# Rate-distortion accounting and the head-to-head against the T8 lossless coder.
def rate_distortion_curve(z, model, bits_list, U=8.0, seed=0):
    """Compute the rate-distortion curve over a list of bit-widths.

    Each entry reports the bit-width, the rate in bits/value, and the latent MSE
    distortion. The latent MSE is the ensemble-term distortion proxy; the true
    observable error (collective-variable KL, implied-timescale error) is measured
    by the gate on real data.
    """
    z = np.asarray(z, float)
    n = z.shape[0] * z.shape[1]
    out = []
    for b in bits_list:
        data, zhat, _ = encode_predictive(z, model, b, U=U, seed=seed)
        out.append({"bits": int(b), "rate_bpv": 8.0 * len(data) / n,
                    "latent_mse": float(np.mean((zhat - z) ** 2))})
    return out
