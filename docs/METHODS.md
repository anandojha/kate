# KATE - Methods and Results

*KATE - **K**inetic-**A**ware **T**rajectory **E**ncoder:
kinetics-preserving compression of molecular-dynamics trajectories, with a
kinetic (path-distribution) fidelity bound. This document consolidates the method
and the measurements obtained on the NTL9 fast-folder. Each quantitative claim is
empirical and is reported with its limitations.*

---

## 1. Problem and thesis

A molecular-dynamics (MD) trajectory is most often compressed to save storage. The
quantities of primary scientific interest, however, are not the raw coordinates but the
observables computed from them: the equilibrium ensemble (free energies, contact
populations) and, more demandingly, the kinetics (the slow timescales and the transition
rates between metastable states).

The central observation motivating KATE is the following.

> Preserving the ensemble does not preserve the kinetics. A compressor can
> reproduce every equilibrium average and still destroy the slow dynamics, and a
> coordinate-error-bounded compressor (SZ3, ZFP, MDZip) can keep every atom within a
> tight tolerance and still collapse the implied timescales, because the slow modes
> reside in a low-variance subspace that uniform error bounds do not protect.

The contribution of KATE is a compressor built around an observable-space bound: a
KL/Pinsker guarantee in the space of the *path distribution*, which is the quantity
that covers the kinetics. The bound is the organizing principle, and the neural-network
components are motivated by it rather than constituting the primary result.

---

## 2. The kinetic bound (the organizing principle)

Treat the trajectory as a sample from a path distribution. The Kullback-Leibler
divergence between the reference path distribution `P` and the compressed one `Q`
factorizes into

```
KL_path(P || Q)  =  KL_ensemble(π_P || π_Q)        (the static / ensemble term)
                  +  Σ_i π_P(i) · KL( P(·|i) || Q(·|i) )   (the transition term)
```

- The ensemble term bounds errors in *static* observables (populations, free
  energies). This is the term controlled by ensemble-preserving and coordinate-bounded
  compressors.
- The transition term `h(P‖Q) = Σ_i π_i Σ_j P_ij log(P_ij/Q_ij)` bounds errors in
  *kinetic* observables. It is a separate term, and a compressor can drive the
  ensemble term to zero while leaving this term large.

By Pinsker's inequality, each KL term bounds the total-variation distance of the
corresponding observables, providing certifiable error bars. `kate bound` reports both
terms and labels which observable each covers. The static Pinsker bound does not cover
kinetics; only the path-distribution (transition) term does.

---

## 3. The method (pipeline)

```
trajectory -> align (Kabsch, protein-only) -> collective variables (TICA / VAMPnet)
           -> normalizing flow (RealNVP / spline)  -> information-gain frame selection
           -> entropy coding (Gaussian / temporal / predictive)
           -> retained MSM  ------------------------------> the .kate artifact
```

The compressed object also serves as the analysis substrate: the `.kate` artifact stores the
run-aware all-frame discrete trajectory (`dtraj`) + k-means centers + the flow, so that
`analyze` / `bound` / `benchmark` re-estimate the MSM at any lag without the
original trajectory; the file is itself the kinetic model. The pure-numpy `bound` path
loads without torch or deeptime.

Components are reused from prior art where credibility is a concern (deeptime's reversible-MLE
MSM, BayesianMSM, VAMPnets, streaming covariance) and reimplemented from the
published algorithm where ownership and reproducibility are a concern (the flow, the
Witten-Neal-Cleary range coder, the path bound, IGFS, the artifact format, the CLI).
No third-party compressor source enters the repository; SZ3/ZFP/MDZip are external
subprocess baselines.

---

## 4. The ML track (T6-T10) and its honest status

| target | what | status |
|---|---|---|
| T6 | learned nonlinear slow CVs via VAMPnets | implemented (`--cv vampnet`) |
| T7 | rational-quadratic neural-spline flow | implemented (`--flow spline`) |
| T8 | temporal learned-entropy coding (lossless) | implemented (`--entropy temporal`) |
| T9 | predictive learned-entropy coding (lossy, causal GRU/DPCM) | both gate halves measured on NTL9 (§6) |
| T10 | the path-bound made differentiable as a training loss | mechanism shown (synthetic); measured negative on NTL9 (§6) |

The following invariants hold across the track: the flow remains invertible, the bound
remains intact, and no lossy CNN autoencoder is used (that is MDZip's design, and it breaks the
thesis). Among the ML components only T8/T10 are positioned as novel; VAMPnets, spline
flows, and context-model entropy coding are cited prior art.

---

## 5. Kinetic-resolution accounting (the honesty tool)

A comparison of compressors on a trajectory first requires knowledge of what that trajectory can
validate. A compressor cannot preserve a kinetic observable that the source never
sampled. `kate analyze --resolution` reports, per dynamical process, the Bayesian
timescale, its 95 % confidence interval, the relative uncertainty, and the number of
independent events the trajectory contains (≈ T_total / t_i). A process is flagged
*resolved* only if its Bayesian error is small and it contains sufficient events.

On 25 µs NTL9 this correctly reports the slow folding mode (t₁ ≈ 30-40 µs, ≈ 0.6
events, > 30 % error) as not resolved, and the faster band (≲ 2 µs) as resolved.
All kinetic claims below are scored only on the resolved band; the slow folding
timescale is sampling-limited and is excluded explicitly rather than reported as if it
were trustworthy. This step is usually omitted in the MD-compression literature.

---

## 6. Results on NTL9 (measured)

System: NTL9 fast-folder, 39 residues, 25 µs at 10 ps. Featurization: CA-CA distances
(|i−j| ≥ 3) → TICA slow CVs. Kinetics: deeptime reversible-MLE MSM on a common k-means
discretization; error = mean relative error of the resolved-band implied timescales.

### 6.1 The contrast - KATE vs SZ3/ZFP (`ntl9_contrast_resolved.png`)

Real SZ3 and ZFP 1.0.1 binaries were run in fixed-accuracy mode, swept over the error bound;
each reconstruction was re-featurized, re-projected on the same TICA, discretized on
the same centers, and then scored on the resolved band.

| to keep resolved kinetics < 1 % error | rate needed |
|---|---|
| KATE (slow CVs + kinetic model) | ~12 bits/frame |
| SZ3 (all-atom, error-bounded) | ~840 bits/frame |
| ZFP (all-atom, error-bounded) | ~1400 bits/frame |

This corresponds to a ~70-120× rate gap. Under aggressive compression the general compressors
collapse the kinetics: SZ3 reaches 95 % timescale error at 331 bits/frame. They
spend bits on bounded *all-atom* error, without regard to which coordinate combinations carry
the slow dynamics, whereas KATE's bound targets the kinetics directly.

KATE and SZ3/ZFP optimize different objectives: SZ3/ZFP provide bounded
all-atom reconstruction, while KATE provides the kinetic model (slow CVs + MSM; all-atom
reconstruction would add the T4 residual bits). The comparable axis is the *rate
needed for a given kinetic fidelity*, which is the quantity plotted in the figure.

### 6.2 The T9 gate - predictive entropy coding (`ntl9_temporal_redundancy.png`, `ntl9_t9_gate.png`)

*Rate half.* KATE compresses *slow* CVs, whose µs timescales keep them strongly
autocorrelated (ρ ≈ 0.98) even at the 0.5 ns storage stride. At equal distortion,
predictive coding saves ~15 bits/frame (~43 %) over static per-frame coding (35 →
20 bits/frame, 8 CVs), and ~14 bits even at 1 ns. This corrects an earlier conservative
hedge ("gain may be modest, frames decorrelated at ~100 ps"), which holds for fast
Cartesian modes but not for the slow CVs that KATE stores.

*Observable-error half.* With closed-loop DPCM reconstruction, scored on the
resolved band, the predictive rate-vs-kinetics-error frontier lies far to the left of static:
0.8 % timescale error at 10 bits/frame versus static's 1.5 % at 26, i.e. the same kinetic
fidelity at ~half to a third the rate (~9× lower error at matched rate). Both halves
of the gate pass on real data.

*Scope:* a 1 µs block only marginally resolves these sub-µs processes, but the
static-vs-predictive comparison is a *paired* test on identical processes, so the
relative result is robust; the predictor used here is linear AR(1)/DPCM, a
conservative stand-in for the GRU.

### 6.3 The bound-as-loss (T10) - a win and an honest negative (`examples/demo_bound_loss.py`)

*Synthetic (mechanism).* On a controlled, well-sampled system (a low-amplitude slow
folding coordinate hidden among high-amplitude fast noise), at equal bit budget,
raw-coordinate MSE, which is what SZ3/ZFP minimize, spends 0 bits on the slow coordinate
and its kinetic distortion is flat in budget, while the differentiable bound-as-loss
protects the slow mode and drives kinetic distortion ~100× lower at 4 bits/frame.
The mechanism is confirmed.

*Real NTL9 (measured negative).* On real TICA CVs (all unit variance after
normalization), the bound-loss correctly concentrates bits on the slow modes and
starves the fast ones, but it does not beat MSE on the real kinetic eval, and MSE
wins at higher budget (3.4 % vs 13.8 % timescale error). The cause is that the real MSM
discretizes with k-means over all 8 CVs, so zeroing the fast modes corrupts the
clustering, whereas the differentiable soft-MSM surrogate treats those modes as
irrelevant; the surrogate and the estimator disagree. Aligning them (soft states matched to
the discretization) is an open problem. This result is reported rather than omitted.

---

## 7. Limitations and honest negatives

- Sampling, not compression, limits NTL9's slow kinetics. The folding timescale
  (~tens of µs) has < 1 independent event in 25 µs, and no featurization or compressor
  fixes a sampling shortage. All kinetic claims are restricted to the resolved band.
- T10 does not yet transfer to real data (§6.3): the differentiable surrogate and
  the k-means/MLE estimator disagree.
- The contrast compares different objectives (§6.1); the appropriate axis is
  rate-for-kinetic-fidelity.
- The "exact/invertible" claim is qualified: the flow is an exact diffeomorphism and kept
  frames are exact up to quantization, but IGFS frame selection is lossy.
- Compression ratio is not the primary metric; kinetic fidelity at a given rate is.
- Validation here is restricted to NTL9. A well-sampled ligand-binding system (e.g.
  trypsin-benzamidine) would test binding kinetics and the slow band that the present data
  cannot resolve.

---

## 8. Reproducibility

```bash
pip install -e ".[kinetics,test]"
python -m pytest -q                                   # 118 tests, torch/deeptime auto-skip
python examples/demo_pathbound.py                     # the kinetic bound (pure numpy)
python examples/demo_bound_loss.py                    # T10 mechanism (synthetic)
kate analyze ART --resolution                          # kinetic-resolution report
kate benchmark TOP DCD --methods kate,sz3,zfp           # the contrast (needs SZ3/ZFP binaries)
```

The NTL9 measurement scripts (figures in `docs/`) require the 25 µs trajectory, which
is not included in the repository; the synthetic mechanism demos and the full test suite run anywhere.
Baseline binaries are located via `KATE_SZ3_BIN` / `KATE_ZFP_BIN` / `KATE_MDZIP_DIR`.
