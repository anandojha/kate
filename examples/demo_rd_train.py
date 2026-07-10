"""
Rate-distortion training of the kinetic path-bound in the flow's latent space.

This is the sanity check for kate.rd_train, the latent-space form of the raw-space objective
in examples/demo_bound_loss.py. The normalizing flow is the analysis transform, quantization
acts in the Gaussian base space z, and a per-dimension bit allocation is trained by
minimizing

    L = rate + lambda * distortion,

where rate is the coded bits per frame, lambda > 0 sets the operating point along the
rate-distortion curve, and the distortion is either the differentiable transition term
h(P||Q) of kate.bound_loss or the coordinate mean-squared error. Each operating point
reports the achieved rate together with the folding-timescale error of the hard-state
certificate, a common k-means clustering plus reversible MSM (kate.kinetic_codec), rather
than the soft training surrogate.

The synthetic system isolates the mechanism. Coordinate dim0 is a slow, low-amplitude
bistable folding coordinate carrying all the slow kinetics, while dims 1-7 are fast,
high-amplitude, and kinetically irrelevant. The kinetic objective drives the slow latent
dimension's quantization width down, spending bits there, and starves the fast dimensions,
an allocation that is unambiguous in the printed widths.

That allocation does not translate into a folding-error advantage here, because the flow
already whitens every mode to unit variance. A uniform latent quantization then preserves
the kinetics to about one percent near ten bits per frame, and the folding-timescale errors
of uniform, MSE, and kinetic quantization at matched rate all lie within the MSM-estimation
noise of a forty-thousand-frame trajectory. The raw-space contrast of demo_bound_loss.py,
where MSE water-fills by amplitude and starves the low-amplitude slow mode, is absent in the
latent space: the invertible transform, not the choice of distortion, does the work. The
kinetic distortion is expected to matter when the transform is not amplitude equalizing, a
weak or fixed transform or a coarse rate, which is the next experiment on real data. The
certified kinetics reported in the paper come from the deeptime reversible-MLE MSM and the
path bound on hard states (kate.pathbound); this is a controlled synthetic that measures the
mechanism.
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

    # One shared analysis transform and one shared soft-state readout, so the only variable
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
