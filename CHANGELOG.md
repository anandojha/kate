# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.1.0] - 2026-06-10

First public release: an end-to-end, analysis-native MD-trajectory compressor with a
kinetic (path-distribution) fidelity bound.

### Added
- **Core codec** - Kabsch alignment, TICA collective variables, a from-scratch RealNVP
  normalizing flow, information-gain frame selection (IGFS), a Witten-Neal-Cleary range
  coder, and a run-aware MSM, packaged as the `glide` CLI
  (`compress` / `decompress` / `analyze` / `bound` / `benchmark`).
- **The kinetic bound** (`glide.pathbound`) - the path-distribution KL factorized into an
  ensemble term + a transition term, with Pinsker bounds on kinetic observables. The
  organizing principle of the project.
- **T1** path bound wired into the runner + `glide bound` (pure-numpy contrast scorer).
- **T2** production kinetics via deeptime (`glide analyze`: lag scan, Bayesian error bars).
- **T3** baseline-comparison harness (`glide benchmark`) with SZ3/ZFP/MDZip subprocess
  wrappers (located via `GLIDE_SZ3_BIN` / `GLIDE_ZFP_BIN` / `GLIDE_MDZIP_DIR`) + local
  pseudo-baselines, and the contrast figure.
- **T4** full-atom reconstruction (`decompress --full-atom`, per-state dithered residual).
- **T5** out-of-core scaling (`compress --streaming`, streaming TICA).
- **T6** learned nonlinear slow CVs via VAMPnets (`--cv vampnet`).
- **T7** rational-quadratic neural-spline flow (`--flow spline`).
- **T8** temporal learned-entropy coding (`--entropy temporal`, lossless).
- **T9** predictive learned-entropy coding (`--entropy predictive`, lossy; causal GRU,
  bound-as-loss, standardized-innovation coding) - predictor built and unit-tested; the
  rate-vs-observable-error gate against T8 runs on the validation trajectory.
- **T10** the kinetic path-bound made **differentiable as a training loss**
  (`glide.bound_loss`): a soft-MSM transition term `h(P‖Q)` with autograd, so
  `loss = rate + λ·h(P‖Q)` trains a compressor to spend bits on *kinetics*, not
  coordinate error. `examples/demo_bound_loss.py` shows the mechanism on a controlled,
  well-sampled synthetic - at equal bit budget, raw-MSE (the SZ3/ZFP objective) spends
  0 bits on the slow coordinate and its kinetic distortion is flat in budget, while the
  bound-as-loss drives it ~100× lower. **Measured negative on real NTL9**: the allocation
  correctly concentrates on the slow TICA modes but does not beat MSE on the real
  k-means/MLE kinetics (MSE wins at higher budget) - the soft-MSM surrogate and the
  estimator disagree; aligning them is the open problem. Reported, not hidden. The
  certified kinetics still come from the deeptime MSM + hard-state path bound.
- **Kinetic-resolution accounting** (`kinetics_deeptime.kinetic_resolution`,
  `glide analyze --resolution`): per dynamical process, the Bayesian timescale + 95% CI,
  relative uncertainty, and independent-event count (~ T_total / t_i); flags a process
  `resolved` only if its error is small AND it has enough events. Makes explicit which
  kinetic observables a trajectory can validate - on 25 µs NTL9 the slow folding modes
  read as not-resolved, the fast band as resolved.
- **The contrast, measured on real NTL9, resolution-aware** (`docs/ntl9_contrast_resolved.png`):
  to preserve the resolved-band kinetics to <1%, GLIDE needs ~12 bits/frame vs SZ3 ~840 /
  ZFP ~1400 (~70-120x), and SZ3/ZFP collapse the kinetics under aggressive compression
  (SZ3 -> 95% timescale error at 331 bits/frame). Real SZ3 1.x / ZFP 1.0.1 binaries via
  the baselines wrappers; kinetics scored only on the band `analyze --resolution` flags.
- **T9 gate measured on real NTL9 (both halves)** - rate
  (`docs/ntl9_temporal_redundancy.png`): at equal distortion, predictive coding saves
  ~15 bits/frame (~43%) over static per-frame coding at the 0.5 ns storage stride,
  because GLIDE compresses slow CVs that stay strongly autocorrelated (ρ≈0.98) even at
  storage spacing (corrects the earlier "may be modest" hedge). Observable-error
  (`docs/ntl9_t9_gate.png`): on the resolved sub-µs band, with closed-loop DPCM
  reconstruction, predictive preserves the kinetics at ~half to a third the rate of
  static (0.8% timescale error at 10 bits/frame vs static 1.5% at 26). Empirical.
- **Code review (physics + competitive) and its fixes** (`docs/REVIEW.md`): audited the
  core against bgflow/bgmol, deeptime, mdzip, sz3/zfp. Resulting changes:
  - **Certified kinetics now use deeptime's reversible MLE** (`estimate_reversible_T`,
    prefers deeptime, falls back to `(C+Cᵀ)/2`); artifact records `msm_estimator`. The
    portable `glide bound` stays torch/deeptime-free on the symmetrized estimator.
  - **`glide analyze --mfpt N`**: PCCA+ metastable coarse-graining + mean first-passage
    times - the k_on/k_off rate observables the bound covers, now actually computed.
  - **`glide compress --features contacts`**: invariant inter-atomic-distance TICA
    featurization (removes spurious rigid-body slow modes; reconstruction unaffected).
  - **Steric-validity check** on decoded frames flags reconstruction-introduced overlaps.
  - **Path bound honors support**: `kinetic_bound_valid` / Pinsker = inf when the true
    divergence is +∞ (a missed transition); `glide bound` warns.
  - Honest competitive positioning in README (GLIDE is not a pure-compression winner).
- **~99% test coverage** (124 tests); torch/deeptime tests auto-skip when absent.

### Notes
- `deeptime` is an optional `[kinetics]` extra; `compress` / `decompress` / `bound` run
  on the core dependencies alone, and `import glide` pulls in neither torch nor deeptime.
- Components (flow, MSM, VAMPnet, spline flow, context-model coding) are reused prior art
  and cited; the contribution is the kinetic bound + the end-to-end integration. T9's
  rate gain is reported empirically against T8, never assumed.
