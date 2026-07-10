"""
Rate-distortion training of the KATE compressor against the kinetic path-bound.

The compressor is fit to the rate-distortion Lagrangian rate + lambda * distortion, in the
neural-compression setting of a learned analysis transform, a learned synthesis transform,
and a rate-distortion objective (Balle et al., ICLR 2017; Minnen et al., NeurIPS 2018),
with one substitution: the distortion is the path-space transition term h(P||Q) of the
differentiable kinetic path-bound (kate.bound_loss), not coordinate mean-squared error.

The analysis transform is the normalizing flow x -> z, quantization acts in the Gaussian
base space, and the synthesis transform is the exact inverse flow z -> x. For a
per-dimension quantization step Delta_d the expected code length is the flow-based rate

    rate(bits/frame) = E[ -log2 N(z) ] - sum_d log2 Delta_d ,

differentiable through the flow. A learnable per-dimension log-width therefore performs a
rate-distortion bit allocation across the latent: under the kinetic distortion it spends
bits on the slow, kinetically decisive directions, and under coordinate mean-squared error
it water-fills by amplitude. Sweeping the Lagrange multiplier lambda traces the
rate-versus-kinetic-distortion curve.

The differentiable transition term is a surrogate evaluated on soft VAMPnet states. The
certified kinetics come separately, on hard states, from the deeptime reversible-maximum-
likelihood MSM and the path bound (kate.pathbound), and the hard-state folding-timescale
error returned by hard_state_error is that check. Rate distortion with a certificate is
thus two objects, a trained surrogate and a post-hoc certificate, and whether training on
the bound beats training on mean-squared error is measured rather than assumed (see
examples/demo_rd_train.py).
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

from .flow import RealNVP
from .bound_loss import SoftStateEncoder, kinetic_distortion
from .kinetic_codec import (count_matrix, estimate_reversible_T, implied_timescales,
                            largest_connected_set)

_LOG2 = math.log(2.0)


def latent_rate_bits(z: torch.Tensor, log_width: torch.Tensor) -> torch.Tensor:
    """Expected code length in bits per frame of base-space latents z quantized at the
    per-dimension step exp(log_width).

    Under the flow's standard-normal base density the per-dimension code length at bin
    width Delta is -log2 N(z) - log2 Delta; summed over dimensions and averaged over the
    batch this is the flow-based rate. Per-dimension rates are clamped to be
    non-negative, since a bin cannot cost fewer than zero bits."""
    neg_log2_p = 0.5 * (z ** 2 + math.log(2 * math.pi)) / _LOG2       # -log2 N(z), per (frame, dim)
    per_dim = neg_log2_p - log_width / _LOG2                          # subtract log2 Delta_d
    return torch.clamp(per_dim, min=0.0).sum(dim=-1).mean()           # bits/frame


class KineticRDCompressor(nn.Module):
    """A latent bit-allocation trained by a rate-distortion objective.

    The flow (analysis/synthesis transform) and the soft-state encoder are held fixed;
    only the per-dimension quantization log-widths are learned, so the trained object is
    the allocation of bits across the flow latent. With the kinetic distortion the
    allocation protects the slow directions that carry the transition term; with
    coordinate mean-squared error it protects the high-amplitude directions instead."""

    def __init__(self, flow: RealNVP, encoder: SoftStateEncoder, lag: int):
        super().__init__()
        self.flow = flow
        self.encoder = encoder
        self.lag = lag
        for p in self.flow.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.log_width = nn.Parameter(torch.zeros(flow.dim))

    def reconstruct_soft(self, CV: torch.Tensor):
        """Differentiable forward pass: flow to the base space, apply the additive-uniform
        quantization proxy (Balle et al., ICLR 2017), invert, and read out soft states.

        Returns the mean rate, the reference and compressed soft-state trajectories, and
        the reconstructed collective variables."""
        z, _ = self.flow.forward(CV)
        width = torch.exp(self.log_width)
        z_q = z + (torch.rand_like(z) - 0.5) * width                 # dithered quantization proxy
        CV_hat = self.flow.inverse(z_q)
        rate = latent_rate_bits(z, self.log_width)
        return rate, self.encoder(CV), self.encoder(CV_hat), CV_hat

    @torch.no_grad()
    def compress_hard(self, CV: torch.Tensor):
        """Round the latents at the learned step (true quantization) and return the
        reconstruction together with the achieved rate, the summed per-dimension empirical
        entropy of the integer codes in bits per frame."""
        z, _ = self.flow.forward(CV)
        width = torch.exp(self.log_width)
        q = torch.round(z / width)
        CV_hat = self.flow.inverse(q * width)
        bits = 0.0
        for d in range(q.shape[1]):
            _, cnt = torch.unique(q[:, d], return_counts=True)
            p = cnt.to(torch.float64) / cnt.sum()
            bits += float(-(p * torch.log2(p)).sum())
        return CV_hat, bits


def train_rd(compressor: KineticRDCompressor, CV, lam: float, kind: str = "kinetic",
             steps: int = 400, lr: float = 0.05, seed: int = 0, verbose: bool = False):
    """Train the bit allocation by minimizing rate + lambda * distortion.

    ``kind`` selects the distortion: 'kinetic' for the differentiable transition term
    (kate.bound_loss), or 'mse' for coordinate mean-squared error. Only
    ``compressor.log_width`` is updated."""
    torch.manual_seed(seed)
    CV = torch.as_tensor(np.asarray(CV), dtype=torch.float32)
    opt = torch.optim.Adam([compressor.log_width], lr=lr)
    for st in range(steps):
        rate, chi_ref, chi_cmp, CV_hat = compressor.reconstruct_soft(CV)
        if kind == "kinetic":
            dist = kinetic_distortion(chi_ref, chi_cmp, compressor.lag)
        elif kind == "mse":
            dist = ((CV - CV_hat) ** 2).mean()
        else:
            raise ValueError("kind must be 'kinetic' or 'mse'")
        loss = rate + lam * dist
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (st % max(1, steps // 8) == 0 or st == steps - 1):
            print("   step %4d  rate=%.3f b/frame  dist=%.4e  loss=%.4f"
                  % (st, float(rate), float(dist), float(loss)))
    return compressor


def hard_state_error(compressor: KineticRDCompressor, CV, n_states: int = 100,
                     lag: int = None, dt: float = 1.0, seed: int = 0) -> dict:
    """Certificate check on HARD states: discretize the reference and the decoded
    reconstruction on one common k-means, estimate reversible MSMs, and return the
    slowest-implied-timescale (folding) error.

    This is the honest, non-differentiable evaluation that mirrors the reported paper
    numbers; ``dt`` scales the timescale to physical units."""
    lag = compressor.lag if lag is None else lag
    CVr = np.asarray(CV, dtype=np.float32)
    CV_hat, rate_bits = compressor.compress_hard(torch.as_tensor(CVr))
    CV_hat = CV_hat.numpy()
    km = MiniBatchKMeans(n_clusters=n_states, random_state=seed, n_init=3,
                         batch_size=max(256, 3 * n_states)).fit(CVr)

    def t1(feat):
        lab = km.predict(feat).astype(np.int64)
        C = count_matrix([lab], n_states, lag)
        act = largest_connected_set(C)
        T, _ = estimate_reversible_T(C[np.ix_(act, act)])
        return float(implied_timescales(T, lag, 1)[0]) * dt

    t1_ref, t1_hat = t1(CVr), t1(CV_hat)
    err = abs(t1_hat - t1_ref) / t1_ref * 100.0 if t1_ref > 0 else float("nan")
    return {"rate_bits": rate_bits, "t1_ref": t1_ref, "t1_hat": t1_hat, "folding_err_pct": err}


def rate_distortion_curve(CV, lag: int, lambdas: Sequence[float], kind: str = "kinetic",
                          flow: RealNVP = None, encoder: SoftStateEncoder = None,
                          n_soft: int = 4, n_states: int = 100, dt: float = 1.0,
                          flow_epochs: int = 200, vamp_epochs: int = 200, steps: int = 400,
                          seed: int = 0, verbose: bool = False) -> List[dict]:
    """Trace a rate-versus-kinetic-error curve by training one bit allocation per lambda.

    A shared flow and soft-state encoder are fit once (or supplied); each lambda then
    yields an operating point recording the achieved rate (bits/frame), the soft kinetic
    distortion at the learned allocation, and the hard-state folding-timescale error."""
    CV = np.asarray(CV, dtype=np.float32)
    dim = CV.shape[1]
    if flow is None:
        flow = RealNVP(dim, hidden=64, n_layers=8).fit(CV, epochs=flow_epochs, verbose=False, seed=seed)
    if encoder is None:
        encoder = SoftStateEncoder(dim, n_soft, hidden=32).fit_vamp(CV, lag=lag, epochs=vamp_epochs, seed=seed)
        for p in encoder.parameters():
            p.requires_grad_(False)
    out = []
    for lam in lambdas:
        comp = KineticRDCompressor(flow, encoder, lag)
        train_rd(comp, CV, lam, kind=kind, steps=steps, seed=seed)
        ev = hard_state_error(comp, CV, n_states=n_states, lag=lag, dt=dt, seed=seed)
        with torch.no_grad():
            _, chi_ref, chi_cmp, _ = comp.reconstruct_soft(torch.as_tensor(CV))
            ev["soft_kinetic_distortion"] = float(kinetic_distortion(chi_ref, chi_cmp, lag))
        ev["lam"] = lam
        ev["kind"] = kind
        out.append(ev)
        if verbose:
            print("  %-7s lambda=%8.1f  rate=%5.2f b/frame  t1 %7.2f->%7.2f  err=%6.1f%%"
                  % (kind, lam, ev["rate_bits"], ev["t1_ref"], ev["t1_hat"], ev["folding_err_pct"]))
    return out
