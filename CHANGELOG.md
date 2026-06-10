# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.1.0] - 2026-06-10

First public release: an end-to-end, analysis-native MD-trajectory compressor with a
kinetic (path-distribution) fidelity bound.

### Added
- **Core codec** — Kabsch alignment, TICA collective variables, a from-scratch RealNVP
  normalizing flow, information-gain frame selection (IGFS), a Witten–Neal–Cleary range
  coder, and a run-aware MSM, packaged as the `epc` CLI
  (`compress` / `decompress` / `analyze` / `bound` / `benchmark`).
- **The kinetic bound** (`epc.pathbound`) — the path-distribution KL factorized into an
  ensemble term + a transition term, with Pinsker bounds on kinetic observables. The
  organizing principle of the project.
- **T1** path bound wired into the runner + `epc bound` (pure-numpy contrast scorer).
- **T2** production kinetics via deeptime (`epc analyze`: lag scan, Bayesian error bars).
- **T3** baseline-comparison harness (`epc benchmark`) with SZ3/ZFP/MDZip subprocess
  wrappers (located via `EPC_SZ3_BIN` / `EPC_ZFP_BIN` / `EPC_MDZIP_DIR`) + local
  pseudo-baselines, and the contrast figure.
- **T4** full-atom reconstruction (`decompress --full-atom`, per-state dithered residual).
- **T5** out-of-core scaling (`compress --streaming`, streaming TICA).
- **T6** learned nonlinear slow CVs via VAMPnets (`--cv vampnet`).
- **T7** rational-quadratic neural-spline flow (`--flow spline`).
- **T8** temporal learned-entropy coding (`--entropy temporal`, lossless).
- **T9** predictive learned-entropy coding (`--entropy predictive`, lossy; causal GRU,
  bound-as-loss, standardized-innovation coding) — predictor built and unit-tested; the
  rate-vs-observable-error gate against T8 runs on the validation trajectory.
- **T10** the kinetic path-bound made **differentiable as a training loss**
  (`epc.bound_loss`): a soft-MSM transition term `h(P‖Q)` with autograd, so
  `loss = rate + λ·h(P‖Q)` trains a compressor to spend bits on *kinetics*, not
  coordinate error. `examples/demo_bound_loss.py` shows the mechanism on a controlled,
  well-sampled synthetic — at equal bit budget, raw-MSE (the SZ3/ZFP objective) spends
  0 bits on the slow coordinate and its kinetic distortion is flat in budget, while the
  bound-as-loss drives it ~100× lower. Mechanism only; inconclusive on sampling-limited
  NTL9; the certified kinetics still come from the deeptime MSM + hard-state path bound.
- **Kinetic-resolution accounting** (`kinetics_deeptime.kinetic_resolution`,
  `epc analyze --resolution`): per dynamical process, the Bayesian timescale + 95% CI,
  relative uncertainty, and independent-event count (~ T_total / t_i); flags a process
  `resolved` only if its error is small AND it has enough events. Makes explicit which
  kinetic observables a trajectory can validate — on 25 µs NTL9 the slow folding modes
  read as not-resolved, the fast band as resolved.
- **T9 rate gain measured on real NTL9 latents** (`docs/ntl9_temporal_redundancy.png`):
  at equal distortion, predictive coding saves ~15 bits/frame (~43%) over static
  per-frame coding at the 0.5 ns storage stride. Corrects the earlier conservative hedge
  — EPC compresses slow CVs, which stay strongly autocorrelated (ρ≈0.98) even at storage
  spacing, so the temporal redundancy is large and real. Rate side of the gate, empirical.
- **~99% test coverage** (118 tests); torch/deeptime tests auto-skip when absent.

### Notes
- `deeptime` is an optional `[kinetics]` extra; `compress` / `decompress` / `bound` run
  on the core dependencies alone, and `import epc` pulls in neither torch nor deeptime.
- Components (flow, MSM, VAMPnet, spline flow, context-model coding) are reused prior art
  and cited; the contribution is the kinetic bound + the end-to-end integration. T9's
  rate gain is reported empirically against T8, never assumed.
