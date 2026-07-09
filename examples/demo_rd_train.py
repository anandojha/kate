"""
demo_rd_train.py  --  T11: the kinetic path-bound as a rate-distortion TRAINING objective.
==========================================================================================
This is the sanity check for `kate.rd_train`, the latent-space rate-distortion form of the
T10 idea (examples/demo_bound_loss.py). The flow is the analysis transform, quantization
acts in the Gaussian base space, and a per-dimension bit allocation is trained by

    loss = rate(bits/frame) + lambda * distortion ,

with the distortion either the differentiable transition term h(P||Q) (kate.bound_loss)
or coordinate mean-squared error. Reported for each operating point is the achieved rate
and the HARD-STATE folding-timescale error (common k-means + reversible MSM,
kate.kinetic_codec), the honest certificate check rather than the soft training surrogate.

THE CONTROLLED SYSTEM: dim0 is a slow, low-amplitude bistable folding coordinate carrying
all the slow kinetics; dims 1-7 are fast, high-amplitude, kinetically irrelevant noise.

WHAT THIS DEMONSTRATES, AND THE HONEST NUANCE:
  1. The kinetic objective does what it is meant to: it drives the slow latent dimension's
     quantization width down (spends bits on it) and starves the fast dimensions. That
     mechanism is unambiguous in the bit allocation printed below.
  2. BUT it does NOT translate into a folding-error advantage on this system, because the
     flow has already whitened every mode to unit variance, so even a UNIFORM latent
     quantization preserves the kinetics well (about one percent error near ten bits per
     frame). The folding-timescale errors of uniform, MSE, and kinetic at matched rate all
     sit within the MSM-estimation noise of a forty-thousand-frame trajectory. The dramatic
     raw-space contrast of T10 (where MSE water-fills by amplitude and starves the
     low-amplitude slow mode) is therefore ABSENT in the latent space where KATE operates:
     the invertible transform, not the choice of distortion, does the work here.

This is a result to know before betting a paper on "kinetics-aware loss beats MSE": once
the transform is a good normalizing flow, the marginal value of the kinetic distortion
over plain uniform latent quantization is small on a well-sampled, cleanly-whitened
system. Where the kinetic loss should matter is when the transform is NOT amplitude
equalizing (a weak or fixed transform, or a coarse rate), which is the experiment to run
next on real data. The certified kinetics in the paper still come from the deeptime
reversible-MLE MSM and the path bound on hard states (kate.pathbound); this is a
controlled synthetic that measures the mechanism, not a real-data claim.
"""
import numpy as np
import torch

from kate.flow import RealNVP
from kate.bound_loss import SoftStateEncoder
from kate.rd_train import KineticRDCompressor, train_rd, hard_state_error, rate_distortion_curve

LAG = 20


def make_system(T=40000, switch=0.006, seed=0):
    """A well-sampled bistable slow coordinate (dim0, low amplitude) hidden among fast
    high-amplitude noise (dims 1-7, kinetically irrelevant)."""
    rng = np.random.default_rng(seed)
    s = np.ones(T)
    for t in range(1, T):
        s[t] = -s[t - 1] if rng.random() < switch else s[t - 1]
    n_switch = int(np.abs(np.diff(s)).sum() / 2)
    z = s + 0.25 * rng.standard_normal(T)
    Y = np.zeros((T, 8), np.float32)
    Y[:, 0] = 0.30 * z
    Y[:, 1:] = 1.0 * rng.standard_normal((T, 7))
    return Y, n_switch


def main():
    Y, n_switch = make_system()
    print("T11 -- kinetic path-bound as a rate-distortion training objective (synthetic)")
    print("system: %d frames, ~%d folding transitions (well sampled)\n" % (len(Y), n_switch))

    # One shared analysis transform and one shared soft-state readout, so the ONLY variable
    # between the trained runs is the distortion objective.
    flow = RealNVP(8, hidden=64, n_layers=8).fit(Y, epochs=150, verbose=False)
    enc = SoftStateEncoder(8, 3, hidden=32).fit_vamp(Y, lag=LAG, epochs=250)
    for p in enc.parameters():
        p.requires_grad_(False)

    # (0) Uniform latent quantization baseline: the flow alone, no learned allocation.
    print("  method    setting     rate(b/frame)   hard-state folding error")
    for w in (3.0, 2.0, 1.0):
        comp = KineticRDCompressor(flow, enc, LAG)
        with torch.no_grad():
            comp.log_width.copy_(torch.full((8,), float(np.log(w))))
        ev = hard_state_error(comp, Y, n_states=100, lag=LAG)
        print("   uniform   w=%.1f        %6.2f            %6.1f%%" % (w, ev["rate_bits"], ev["folding_err_pct"]))

    # (1) Trained allocations, lambda ranges chosen so both objectives span similar rates.
    for kind, lambdas in (("mse", [3.0, 10.0, 30.0, 100.0]),
                          ("kinetic", [3e2, 1e3, 3e3, 1e4])):
        pts = rate_distortion_curve(Y, lag=LAG, lambdas=lambdas, kind=kind,
                                    flow=flow, encoder=enc, n_states=100, steps=250)
        for p in pts:
            print("   %-8s lam=%-6.0f    %6.2f            %6.1f%%"
                  % (kind, p["lam"], p["rate_bits"], p["folding_err_pct"]))

    # (2) The mechanism: the kinetic allocation protects the slow latent direction.
    z, _ = flow.forward(torch.as_tensor(Y))
    slow = int(np.argmax([abs(np.corrcoef(z.detach().numpy()[:, d], Y[:, 0])[0, 1]) for d in range(8)]))
    comp = KineticRDCompressor(flow, enc, LAG)
    train_rd(comp, Y, 1e3, kind="kinetic", steps=250)
    w = torch.exp(comp.log_width).detach().numpy()
    print("\nBit allocation (kinetic, lambda=1000): slow-dim width=%.2f  fast-dim mean width=%.2f"
          % (w[slow], w[[d for d in range(8) if d != slow]].mean()))
    print("(a small width means many bits: the kinetic objective spends them on the slow mode.)")
    print("\nReading: the kinetic objective demonstrably spends its bits on the slow direction")
    print("(the small slow-dim width above), but the flow's whitening already lets uniform latent")
    print("quantization preserve the kinetics -- so on this system the three folding errors sit")
    print("within noise, and the transform, not the distortion choice, does the work.")


if __name__ == "__main__":
    main()
