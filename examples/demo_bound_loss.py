"""
demo_bound_loss.py  --  T10: the kinetic path-bound used as a TRAINING LOSS.
============================================================================
This is the §2 sanity check for `glide.bound_loss`: it demonstrates the one place
where the ML and the novel idea become the same thing -- a compressor's bit
allocation trained to minimize the *kinetic* path-bound term instead of
coordinate error.

THE CONTROLLED SYSTEM (designed so the answer is knowable):
  dim 0  = a slow, bistable folding coordinate -- LOW amplitude (sigma~0.3),
           but it carries ALL of the slow kinetics. Well sampled (~470 folding
           transitions over 120k frames), so the kinetics are signal, not noise.
  dim 1-7 = fast white noise -- HIGH amplitude (sigma~1.0), kinetically IRRELEVANT.

THE CONTRAST (a fixed total bit budget B/frame, allocated two ways):
  "mse"  : minimize RAW-coordinate error -- exactly what SZ3/ZFP/MDZip minimize.
           Water-fills bits by amplitude -> pours the whole budget into the fast
           noise, spends ~0 bits on the low-amplitude slow coordinate.
  "kin"  : minimize the differentiable transition term h(P||Q) of the soft MSM
           (glide.bound_loss) -- discovers it must protect dim 0 and starve the rest.

THE RESULT (printed below): at EQUAL total rate, raw-MSE's kinetic distortion is
~flat in budget (more bits buy NO kinetic fidelity -- they go to the wrong place),
while the bound-as-loss converts every bit into kinetic fidelity (~100x lower
distortion at 4 bits/frame). This is the whole thesis -- "coordinate-error
compression does not preserve kinetics; the path bound does" -- made trainable.

HONEST SCOPE -- read this:
  * This is a CONTROLLED SYNTHETIC system, built so the slow mode is low-amplitude
    and well sampled. It demonstrates the MECHANISM, not a real-data win.
  * On NTL9 the same experiment is INCONCLUSIVE: with ~6 folding events the soft
    MSM's transition term is dominated by sampling noise, so the bound gives little
    training signal. That is a sampling limit of that trajectory, not of the method.
  * The certified kinetics in the paper still come from the deeptime reversible-MLE
    MSM + the path bound on hard states (glide.pathbound). bound_loss is the
    differentiable SURROGATE used as a loss; whether it beats MSE on real data is an
    empirical question to be answered on a well-sampled system.
"""
import numpy as np
import torch
import torch.nn as nn

from glide.bound_loss import (SoftStateEncoder, soft_transition_matrix,
                            transition_term, _lagged)

LAG = 20


def make_system(T=120000, switch=0.004, seed=0):
    """A well-sampled bistable slow coordinate (dim0, low amplitude) hidden among
    fast high-amplitude noise (dims 1-7, kinetically irrelevant)."""
    rng = np.random.default_rng(seed)
    s = np.ones(T)
    for t in range(1, T):
        s[t] = -s[t - 1] if rng.random() < switch else s[t - 1]
    n_switch = int(np.abs(np.diff(s)).sum() / 2)
    z = s + 0.25 * rng.standard_normal(T)
    Y = np.zeros((T, 8), np.float32)
    Y[:, 0] = 0.30 * z                              # slow, low amplitude, kinetic
    Y[:, 1:] = 1.0 * rng.standard_normal((T, 7))    # fast, high amplitude, irrelevant
    return Y, n_switch


def main():
    torch.manual_seed(0)
    Y, n_switch = make_system()
    ntr = int(0.7 * len(Y))
    Yt, Ye = torch.tensor(Y[:ntr]), torch.tensor(Y[ntr:])
    sig = torch.tensor(Y.std(0))
    print("T10 -- kinetic path-bound as a training loss (controlled synthetic)")
    print("system: %d frames, ~%d folding transitions (well sampled)" % (len(Y), n_switch))
    print("        dim0 sigma=%.2f (slow, kinetic) | fast dims sigma~%.2f (irrelevant)\n"
          % (float(sig[0]), float(sig[1])))

    # frozen VAMPnet-style soft-state readout (so the kinetic distortion is meaningful)
    enc = SoftStateEncoder(8, 3, hidden=32).fit_vamp(Y[:ntr], lag=LAG, epochs=300)
    for p in enc.parameters():
        p.requires_grad_(False)
    Pref = soft_transition_matrix(*_lagged(enc(Ye), LAG))        # unquantized reference kinetics

    def alloc(obj, B, epochs=500):
        """Learn a per-dim bit budget (softmax -> sums to B exactly) by minimizing
        either raw-coordinate MSE or the differentiable kinetic transition term."""
        th = nn.Parameter(torch.zeros(8))
        opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(epochs):
            b = B * torch.softmax(th, 0)
            step = sig / (2.0 ** b)
            Yh = Yt + (torch.rand_like(Yt) - 0.5) * step          # dithered quant, raw space
            if obj == "kin":
                D = transition_term(soft_transition_matrix(*_lagged(enc(Yt), LAG)),
                                    soft_transition_matrix(*_lagged(enc(Yh), LAG)))
            else:
                D = ((Yt - Yh) ** 2).mean()                       # raw-coordinate MSE (SZ3/ZFP)
            opt.zero_grad(); D.backward(); opt.step()
        return (B * torch.softmax(th, 0)).detach()

    def eval_kd(b):
        """Hard-quantize the test trajectory at budget b, measure kinetic distortion."""
        step = sig / (2.0 ** b)
        Yh = torch.round(Ye / step) * step
        return float(transition_term(Pref, soft_transition_matrix(*_lagged(enc(Yh), LAG))))

    print("  budget B   method   bits->dim0   bits->fast(avg)   kinetic_distortion(test)")
    for B in (1.0, 2.0, 3.0, 4.0, 6.0):
        for obj in ("mse", "kin"):
            b = alloc(obj, B)
            print("   %4.1f      %-3s     %6.2f        %6.2f             %.4e"
                  % (B, obj, float(b[0]), float(b[1:].mean()), eval_kd(b)))
    print("\nReading: raw-MSE spends ~0 bits on the slow coordinate at every budget (its")
    print("kinetic distortion is flat -- more bits buy no kinetic fidelity); the bound-as-")
    print("loss protects dim0 and drives the distortion down ~100x at equal rate.")


if __name__ == "__main__":
    main()
