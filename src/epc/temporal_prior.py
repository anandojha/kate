"""
temporal_prior.py
=================
T8 -- the temporal + learned-entropy model (the defensibly-novel ML piece).

The flow already Gaussianizes EACH FRAME, so its latents z_t are ~N(0,I) marginally.
But consecutive frames are correlated, so the SEQUENCE z_1..z_T is not i.i.d. A causal
sequence model learns p(z_t | z_{<t}); coding each latent against that LEARNED
conditional (a "context model" / "hyperprior", cf. learned image/video compression)
instead of the fixed N(0,I) base shortens the code wherever frames are predictable.

What this does and does NOT touch (the thesis stays intact):
  * It changes ONLY the entropy coder's probability model -> only the CODE LENGTH.
  * The flow, the reconstruction, and the KL/Pinsker bound are UNCHANGED. The coder
    stays EXACTLY lossless (arithmetic coding is exact regardless of the model used).

LOSSLESSNESS -- the correctness landmine. Arithmetic coding desyncs catastrophically
if the encoder and decoder ever build a different probability table for a symbol. So:
  1. The model predicts the conditional DETERMINISTICALLY (CPU, float32, eval() mode,
     no dropout / nondeterministic ops).
  2. The predicted Gaussian is discretized onto the SAME fixed grid as the i.i.d.
     coder and converted via the SAME integer ``_probs_to_cumfreq`` -> an INTEGER
     cumulative-frequency table. Coding is against that integer table, so sub-ULP
     float wobble that leaves the table unchanged is harmless.
  3. The context is the QUANTIZED reconstruction (dequantized levels), never the
     original continuous z -- so encode and decode condition on the SAME values.
  4. Decode is strictly FRAME-BY-FRAME, feeding each decoded latent back as context.
``encode_sequence`` / ``decode_sequence`` are exact inverses (asserted in tests).

HONEST RATE CLAIM: at 100 ps frame spacing consecutive frames are fairly
decorrelated, so the real-data gain may be MODEST. The synthetic test proves the
MECHANISM (rate(temporal) <= rate(gaussian) on a correlated sequence); the real-data
gain is empirical -- do not pre-promise a big win.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import erf

from .kinetic_codec import (_BitWriter, _BitReader, _HALF, _QUARTER, _3QUARTER,
                            _MASK, _PREC, _FREQ_TOTAL, _probs_to_cumfreq)


# ============================================================================
# 1. causal sequence model: predict (mu_t, log_sigma_t) from z_{<t}
# ============================================================================
class _CausalConv1d(nn.Module):
    def __init__(self, ci, co, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ci, co, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TemporalPrior(nn.Module):
    """A small causal dilated-CNN that outputs, for each step t, the mean and
    log-scale of a per-dimension Gaussian conditional p(z_t | z_{<t}). Output t depends
    ONLY on inputs < t (a right shift + causal convolutions), so frame 0 is predicted
    from "nothing" (-> ~N(0,1), the base)."""

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 3, kernel: int = 3):
        super().__init__()
        self.dim = dim
        layers = []
        in_ch = dim
        d = 1
        for _ in range(n_layers):
            layers += [_CausalConv1d(in_ch, hidden, kernel, d), nn.ReLU()]
            in_ch = hidden
            d *= 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 2 * dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        """z: (B, T, dim) -> (mu, log_sigma), each (B, T, dim). Causal."""
        x = z.transpose(1, 2)                       # (B, dim, T)
        x = F.pad(x, (1, 0))[:, :, :-1]             # shift right: output t sees < t
        h = self.body(x)
        out = self.head(h).transpose(1, 2)          # (B, T, 2*dim)
        mu, log_s = out.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 3.0             # bound the scale for stability
        return mu, log_s

    def fit(self, z, epochs=200, lr=1e-3, weight_decay=0.0, verbose=False, seed=0):
        """Train on one latent sequence z (T, dim) by minimizing the Gaussian NLL."""
        torch.manual_seed(seed)
        Z = torch.as_tensor(np.asarray(z), dtype=torch.float32).unsqueeze(0)  # (1,T,dim)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(epochs):
            mu, log_s = self.forward(Z)
            nll = (0.5 * ((Z - mu) ** 2) * torch.exp(-2 * log_s)
                   + log_s + 0.5 * math.log(2 * math.pi)).mean()
            opt.zero_grad(); nll.backward(); opt.step()
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print("  temporal epoch %4d  NLL/dim = %.4f" % (ep, nll.item()))
        return self

    @torch.no_grad()
    def predict_all(self, zq):
        """One-shot causal predictions on the QUANTIZED context zq (T, dim). Returns
        (mu, log_sigma) as float64 numpy. Deterministic (eval mode, CPU)."""
        self.eval()
        Z = torch.as_tensor(np.asarray(zq), dtype=torch.float32).unsqueeze(0)
        mu, log_s = self.forward(Z)
        return mu[0].double().numpy(), log_s[0].double().numpy()

    @torch.no_grad()
    def predict_step(self, zq_prefix, t):
        """Causal prediction for step t from a (T, dim) buffer whose rows < t hold the
        decoded latents (rows >= t are ignored by causality). Returns (mu_t, log_s_t)
        as float64 numpy vectors. Used by the frame-by-frame decoder."""
        self.eval()
        Z = torch.as_tensor(np.asarray(zq_prefix), dtype=torch.float32).unsqueeze(0)
        mu, log_s = self.forward(Z)
        return mu[0, t].double().numpy(), log_s[0, t].double().numpy()


# ============================================================================
# 2. quantization grid (shared by encoder and decoder; matches codec.gaussian_cumfreq)
# ============================================================================
def _grid(L, zmax):
    return np.linspace(-zmax, zmax, L + 1)


def quantize(z, L, zmax):
    lev = np.floor((np.clip(z, -zmax, zmax) + zmax) / (2 * zmax) * L).astype(np.int64)
    return np.clip(lev, 0, L - 1)


def dequantize(levels, L, zmax):
    return -zmax + (levels + 0.5) * (2 * zmax / L)


def _cond_cumfreq(mu, log_s, edges):
    """Integer cumfreq table for a Gaussian(mu, exp(log_s)) discretized on `edges`.
    Deterministic; the integer table absorbs sub-ULP float wobble."""
    sigma = max(float(np.exp(log_s)), 1e-6)
    cdf = 0.5 * (1.0 + erf((edges - mu) / (sigma * np.sqrt(2.0))))
    p = np.clip(np.diff(cdf), 1e-12, None)
    p /= p.sum()
    return _probs_to_cumfreq(p)


# ============================================================================
# 3. stateful WNC range coder (per-symbol cumulative-frequency tables)
# ============================================================================
class _RangeEncoder:
    def __init__(self):
        self.w = _BitWriter(); self.low = 0; self.high = _MASK; self.pending = 0

    def encode(self, s, cum):
        total = int(cum[-1]); rng = self.high - self.low + 1
        self.high = self.low + (rng * int(cum[s + 1])) // total - 1
        self.low = self.low + (rng * int(cum[s])) // total
        while True:
            if self.high < _HALF:
                self.pending = self.w.emit(0, self.pending)
            elif self.low >= _HALF:
                self.pending = self.w.emit(1, self.pending)
                self.low -= _HALF; self.high -= _HALF
            elif self.low >= _QUARTER and self.high < _3QUARTER:
                self.pending += 1; self.low -= _QUARTER; self.high -= _QUARTER
            else:
                break
            self.low = (self.low << 1) & _MASK
            self.high = ((self.high << 1) | 1) & _MASK

    def finish(self):
        self.pending += 1
        self.w.emit(0 if self.low < _QUARTER else 1, self.pending)
        return self.w.to_bytes()


class _RangeDecoder:
    def __init__(self, data):
        self.r = _BitReader(data); self.low = 0; self.high = _MASK; self.code = 0
        for _ in range(_PREC):
            self.code = ((self.code << 1) | self.r.next_bit()) & _MASK

    def decode(self, cum):
        total = int(cum[-1]); rng = self.high - self.low + 1
        value = (((self.code - self.low) + 1) * total - 1) // rng
        s = int(np.searchsorted(cum, value, side="right") - 1)
        s = min(max(s, 0), cum.size - 2)
        self.high = self.low + (rng * int(cum[s + 1])) // total - 1
        self.low = self.low + (rng * int(cum[s])) // total
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.code -= _HALF; self.low -= _HALF; self.high -= _HALF
            elif self.low >= _QUARTER and self.high < _3QUARTER:
                self.code -= _QUARTER; self.low -= _QUARTER; self.high -= _QUARTER
            else:
                break
            self.low = (self.low << 1) & _MASK
            self.high = ((self.high << 1) | 1) & _MASK
            self.code = ((self.code << 1) | self.r.next_bit()) & _MASK
        return s


# ============================================================================
# 4. sequence codec (exact inverses)
# ============================================================================
def encode_sequence(z, model: TemporalPrior, L: int, zmax: float) -> bytes:
    """Code the latent sequence z (T, dim) against the learned conditional. The context
    is the QUANTIZED reconstruction (so encode/decode see identical context); the model
    predicts in one causal pass; coding is symbol-by-symbol (frame, then dim)."""
    z = np.asarray(z, dtype=np.float64)
    T, dim = z.shape
    edges = _grid(L, zmax)
    levels = quantize(z, L, zmax)
    zq = dequantize(levels, L, zmax)                 # the reconstruction context
    mu, log_s = model.predict_all(zq)                # (T, dim) one-shot causal
    enc = _RangeEncoder()
    for t in range(T):
        for d in range(dim):
            enc.encode(int(levels[t, d]), _cond_cumfreq(mu[t, d], log_s[t, d], edges))
    return enc.finish()


def decode_sequence(data: bytes, T: int, dim: int, model: TemporalPrior,
                    L: int, zmax: float) -> np.ndarray:
    """Inverse of encode_sequence: decode frame-by-frame, feeding the decoded
    (quantized) latents back as context. Returns the reconstructed levels (T, dim)."""
    edges = _grid(L, zmax)
    dec = _RangeDecoder(data)
    zq = np.zeros((T, dim), dtype=np.float64)
    levels = np.zeros((T, dim), dtype=np.int64)
    for t in range(T):
        mu_t, log_s_t = model.predict_step(zq, t)    # from decoded rows < t
        for d in range(dim):
            s = dec.decode(_cond_cumfreq(mu_t[d], log_s_t[d], edges))
            levels[t, d] = s
            zq[t, d] = dequantize(np.int64(s), L, zmax)
    return levels


# ============================================================================
# 5. rate accounting (with vs without the temporal model)
# ============================================================================
def gaussian_rate_bits_per_value(z, L, zmax):
    """Bits/value coding the SAME quantized levels against the fixed N(0,I) base
    (the i.i.d. baseline -- what EPC's coder does today)."""
    from .codec import gaussian_cumfreq, encode_iid
    levels = quantize(np.asarray(z), L, zmax).ravel()
    coded = encode_iid(levels, gaussian_cumfreq(L, zmax))
    return 8.0 * len(coded) / levels.size


def temporal_rate_bits_per_value(z, model, L, zmax):
    """Bits/value coding the SAME quantized levels against the learned conditional."""
    coded = encode_sequence(z, model, L, zmax)
    return 8.0 * len(coded) / (np.asarray(z).shape[0] * np.asarray(z).shape[1])
