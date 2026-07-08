# kate - kinetics-preserving compression of MD trajectories

**KATE** = **K**inetic-**A**ware **T**rajectory **E**ncoder. It is a normalizing-flow codec whose fidelity bound covers the kinetics in addition to the ensemble.

[![CI](https://github.com/anandojha/kate/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/kate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![coverage 99%](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)](#tests)

> **Thesis.** Ensemble-preserving compression does not preserve kinetics. Two
> ensembles with identical stationary distributions can have arbitrarily different
> rates. KATE adds a path-distribution (trajectory) bound,
> `KL(path) = ensemble term + transition term`, so that kinetic observables
> (timescales, MFPTs, k_on/k_off) are covered in addition to static ones. The
> path-space bound itself is established prior art; the contribution is adopting it
> as a compressor's fidelity objective, not the bound and not the architecture.

This repository packages a tested research codebase as an installable library and CLI: a
classical analysis-native codec, a from-scratch RealNVP normalizing flow, the
flow-based KATE codec, the kinetic path bound, and a
[deeptime](https://github.com/deeptime-ml/deeptime) MSM wrapper. See
[`docs/REVIEW.md`](docs/REVIEW.md) for the prior work and baselines this builds on and
how it differs.

```bash
pip install git+https://github.com/anandojha/kate.git          # core
pip install "kate[kinetics] @ git+https://github.com/anandojha/kate.git"  # + deeptime
```

## What is new

KATE is, to our knowledge, the first MD-trajectory compressor whose declared fidelity
target is the kinetics. It adopts a path-distribution (relative-entropy-rate) bound as
the operating objective of a codec that unifies a normalizing-flow density model, a
Markov state model, and entropy coding, and shows that the slow timescales are
preserved where ensemble- and coordinate-bounded compressors collapse them.

The bound itself is not new. Path-space relative-entropy-rate and goal-oriented
information bounds for stochastic dynamics are established prior art (see
[Relation to prior work](#relation-to-prior-work)); KATE re-derives the discrete-Markov
special case and applies it. The contribution is the application and the measured
contrast, not a new theorem and not any individual component: normalizing flows
(Boltzmann Generators), MSM-as-entropy-coder, learned entropy coding (MDZip, JCIM AE,
SZ3/ZFP/MDZ), and VAMPnets are all reused prior art.

## Relation to prior work

The kinetic bound KATE uses is a special case of the path-space information bounds for
stochastic dynamics developed in the uncertainty-quantification literature:

- Pantazis & Katsoulakis, J. Chem. Phys. 138, 054115 (2013), arXiv:1210.7264 -
  relative-entropy-rate (RER) path-space sensitivity for stationary stochastic dynamics.
- Dupuis, Katsoulakis, Pantazis & Plechac, SIAM/ASA JUQ (2016), arXiv:1503.05136 -
  goal-oriented path-space information bounds, tighter than plain RER/Pinsker.
- Birrell, Katsoulakis & Rey-Bellet, arXiv:1906.09282 (2019) - path-space UQ bounds for
  hitting times / mean first-passage times specifically.

KATE's stationary-plus-transition factorization is the discrete-time finite-state Markov
specialization of that RER object, and the observables it covers (implied timescales,
MFPTs) are the ones this line of work already bounds. (These references were verified at
abstract level; confirm the full text before the paper relies on exact theorem statements.)

Observable-preserving compression is itself an established paradigm: QPET (Liu et al.,
VLDB 2025, arXiv:2412.02799) and physics-aware rate-distortion (arXiv:2606.03279)
constrain distortion in the space of physical quantities. KATE differs in the observable:
the slow kinetics are a non-local functional of the whole path (implied timescales / MSM
eigenvalues), which point-wise differentiable-QoI error bounds do not reach.

The T10 idea of using the path bound as a training loss is developed concurrently and
independently by Zou, Lie & Marzouk (arXiv:2603.20467, March 2026) for surrogate-SDE
drift learning, which slightly predates this repository's public release; T10 is therefore
positioned as concurrent work, not as first.

## Results on NTL9 (measured, real trajectory)

> Full narrative: [`docs/METHODS.md`](docs/METHODS.md), covering the bound, the method, the
> resolution accounting, and every measured result with its limitations.

Validated on the 25 µs NTL9 fast-folder, scored only on the kinetically resolved band
(`kate analyze --resolution`; the slow folding mode is sampling-limited and is not scored;
see honesty constraints). All numbers are empirical, with the limits stated.

- The contrast ([`docs/ntl9_contrast_resolved.png`](docs/ntl9_contrast_resolved.png)).
  To preserve the resolved kinetics to <1 % timescale error, KATE needs ~12 bits/frame,
  whereas general-purpose error-bounded compressors need ~840 (SZ3) - 1400 (ZFP), a ~70-120×
  rate gap. When pushed to aggressive compression, SZ3/ZFP collapse the kinetics
  (SZ3 reaches 95 % timescale error at 331 bits/frame). These methods spend bits on bounded all-atom
  error, blind to which coordinates carry the slow dynamics, whereas KATE's bound targets the
  kinetics. The objectives differ; the axis reported is the rate needed for a given kinetic
  fidelity, which is what is plotted.
- Predictive (T9) entropy coding ([`docs/ntl9_temporal_redundancy.png`](docs/ntl9_temporal_redundancy.png),
  [`docs/ntl9_t9_gate.png`](docs/ntl9_t9_gate.png)). Slow CVs remain autocorrelated (ρ≈0.98)
  even at the 0.5 ns storage stride, so predictive coding saves ~15 bits/frame (~43 %)
  over static coding at equal distortion, and on the resolved band preserves the kinetics at
  ~half to a third the rate of static coding (~9× lower timescale error at matched rate).
- Bit allocation that respects kinetics ([`examples/demo_bound_loss.py`](examples/demo_bound_loss.py)).
  The differentiable path-bound (T10), trained as a loss, spends bits on the slow
  coordinate where raw-MSE spends none (the mechanism is shown on a controlled synthetic). On
  real NTL9 it is a measured negative: the allocation correctly targets the slow
  modes but does not beat MSE on the real k-means/MLE kinetics, because the soft-MSM surrogate
  and the estimator disagree. This is reported, not hidden (see the T10 entry below).

## Honesty constraints (do not regress)

- Dropped: "first error-bounded MD compressor", which is false, since SZ/ZFP/MDZ bound
  coordinates/QoI already. The genuine novelty is the observable-space (KL/Pinsker)
  bound, specifically the kinetic (path) bound.
- The ensemble (static) Pinsker bound does not cover kinetic observables.
  Only the path-distribution bound (`kate.pathbound`) does. The `bound` report
  labels which term covers what.
- "Exact / invertible" is qualified: the flow is an exact diffeomorphism and kept
  frames reconstruct exactly up to quantization, but information-gain frame
  selection (IGFS) is a lossy step, so the three together are not "exact" unqualified.
- The validation system is trypsin-benzamidine (not the kinase runs from the
  abstract); it is used for the kinetics claims.
- The compression ratio (~8× at 4-bit) is not the contribution; it is comparable to
  plain quantization. The differentiators are analysis-nativeness and the bound.
- The flow, MSM, coder, and IGFS are machinery and are not claimed as invented.
  The contribution is the framing, the path/kinetic bound, the application, and the contrast.
- ML track: the architecture is not the contribution. Among the neural pieces only
  the temporal learned-entropy model (T8) is positioned as novel here; VAMPnets,
  spline flows, MAF, equivariant flows, and learned entropy models are reused prior
  art. All method names and citations are to be verified before the paper.
- Not a pure-compression winner. On rate, all-atom RMSD, speed, and maturity, the
  dedicated compressors win: SZ3 has an MD-specific spatio-temporal predictor, a hard
  per-atom error bound, and zstd; ZFP has O(1) random access; MDZip's autoencoder stores
  ~one small latent per whole frame and reconstructs all atoms (KATE reconstructs
  coordinates only for the ~10% kept frames). KATE's narrow advantage is that it is the
  only one that retains kinetics (the MSM), ships a path-distribution bound on
  kinetic observables, and is analysis-native (the file is the kinetic model).
- MDZip is a coordinate-RMSD method, not an ensemble-preserving one; its loss is
  `sqrt(mean((recon−x)²))`. That such a method can damage kinetics is a correctly
  diagnosed hypothesis, but it is empirical and not yet measured here, so it is not asserted.
- The certified kinetics currently run through the hand-rolled `(C+Cᵀ)/2` estimator,
  not deeptime's reversible MLE (`runner.py`, `benchmark.py`, `kate bound`). That
  estimator is reversible but statistically biased; routing the reported numbers through
  `deeptime` MLE is the top open fix. See [`docs/REVIEW.md`](docs/REVIEW.md) for the full
  internal physics/competitive audit and the prioritized roadmap.

## Citation ≠ license compliance

Citing a paper is an academic courtesy. Copying code is a separate obligation
governed by that project's software license (attribution, and possibly copyleft that
could relicense the repository). This project imports mature libraries and
reimplements algorithms from their papers; the algorithms are cited, and no relicensing occurs.
No third-party source (deeptime, MDZip, SZ3, ZFP) is vendored into the repository.

Provenance boundary:
- Reimplemented from the method: flow, entropy coder, path bound, IGFS,
  artifact format, CLI, benchmark, and the T7/T8 ML pieces.
- Imported, never copied: `deeptime` (reversible-MLE MSM, BayesianMSM, VAMPnets,
  streaming covariance) and `numpy/scipy/scikit-learn/torch/mdtraj`.
- External baselines, run as subprocesses: MDZip / SZ3 / ZFP (never vendored).

## Installation

**One-line install (conda):**

```bash
git clone https://github.com/anandojha/kate.git
cd kate
bash install_kate.sh
```

This creates a fresh conda environment named `kate`, installs all dependencies,
builds and installs KATE, and runs the test suite to verify.

**Manual (conda):**

```bash
git clone https://github.com/anandojha/kate.git
cd kate
conda create -n kate python=3.11 -y
conda activate kate
conda install -c conda-forge mdtraj deeptime matplotlib -y
pip install torch
pip install ".[kinetics,test]"
```

**pip only:**

```bash
pip install git+https://github.com/anandojha/kate.git                       # core
pip install "kate[kinetics] @ git+https://github.com/anandojha/kate.git"    # + deeptime
```

`torch` is a core dependency (the flow requires it). `deeptime` is an optional
`[kinetics]` extra: `kate compress`, `decompress`, and `bound` run without it; only
`analyze`, `benchmark`, and the VAMPnet CV path import it. Importing `kate` pulls in
neither torch nor deeptime eagerly, as enforced by `tests/test_no_eager_torch.py`.

## Testing

```bash
python -m pytest tests/ -v
```

## Quick start

```bash
conda activate kate
kate compress topology.pdb trajectory.dcd -o run.kate   # compress a trajectory
kate analyze run.kate --resolution                      # timescales + what is resolved
kate bound run.kate reference.kate                     # kinetic-fidelity report
```

## The target: one end-to-end tool

```
kate compress   TOP DCD  -> artifact    align -> CV/flow -> IGFS -> entropy code + retained MSM
kate decompress artifact -> trajectory  flow inverse for kept frames; full-atom residual stage
kate analyze    artifact -> kinetics    deeptime MSM: timescales, lag scan, Bayesian bars, --resolution
kate bound      artifact ref -> report  ensemble term, transition term, Pinsker pair/path bounds
kate benchmark  TOP DCD  -> table+plot  KATE vs MDZip vs SZ3 vs ZFP, each scored by the path bound
```

Module map: `compress = runner.py/codec.py`, `decompress = codec.py (+T4 residual)`,
`analyze = kinetics_deeptime.py`, `bound = pathbound.py`, `benchmark = benchmark.py`.
The artifact stores the run-aware all-frame dtraj and k-means centers (not just one
count matrix), so `analyze`/`benchmark` can re-estimate the MSM at any lag.

`kate analyze --resolution` adds a kinetic-resolution report: per dynamical process,
the Bayesian timescale, its 95% confidence interval, the relative uncertainty, and the
number of independent events the trajectory contains (~ T_total / t_i). A process is
flagged resolved only if its Bayesian error is small and it has enough events,
because no compressor can preserve a kinetic observable the source trajectory never
sampled. On the 25 µs NTL9 set this correctly reports the slow folding modes (> ~3 µs,
< a handful of events) as not resolved and the faster band as resolved, so any
kinetic claim is held to what the data actually supports rather than to the slowest eigenvalue
the estimator happens to return. This step is usually omitted in the
MD-compression literature.

`kate analyze` also carries the MSM-community validation tooling reviewers expect:
`--lag-scan` for implied-timescale convergence (Prinz et al. 2011), `--cktest` for the
Chapman-Kolmogorov test on PCCA+ metastable sets (a retained MSM passes it while a
kinetics-corrupting reconstruction fails, which states the thesis in the field's own
validation language), `--bootstrap N` for block-bootstrap timescale confidence
intervals, and `--mfpt N` for PCCA+ mean-first-passage-time rates. Every reported
timescale, MFPT, and rate can therefore carry an error bar.

## Sanity checks (all pass on CPU)

| original script           | here (packaged)                       | checks |
|---------------------------|---------------------------------------|--------|
| `python kate_flow.py`      | `python -m kate.flow`                  | invertibility ~1e-6, density ~1, wells recovered |
| `python demo_pathbound.py`| `python examples/demo_pathbound.py`   | ensemble term ~0 for both chains; transition term large for the ensemble-only chain |
| `python demo_kinetic_codec.py` | `python examples/demo_kinetic_codec.py` | range coder hits the MSM entropy-rate floor; kinetics recovered |
| `python kinetics_deeptime.py` | `python -m kate.kinetics_deeptime`  | reversible MLE MSM, lag scan, Bayesian error bars (needs `[kinetics]`) |
| `python demo_kate.py`      | `python examples/demo_kate.py`         | full flow-based pipeline + measured bound |

Run the test suite with `pytest` (torch/deeptime-dependent tests auto-skip if those
libraries are absent).

### Reproducibility note (kinetics)

The crude classical estimator in `kinetic_codec` (a single-lag `(C+C^T)/2` MSM on
TICA of aligned Cartesian coordinates) is featurization-limited and its recovered
timescales are library-version sensitive: on the newest numpy/scipy/sklearn the
synthetic demo's slow timescales are under-resolved (the leading TICA mode on raw
Cartesian is spurious). This is expected and is the reason deeptime is
the published path: implied timescales are a lower bound that converges upward with
lag (Prinz et al.), so the rigorous kinetics come from a deeptime reversible-MLE
MSM and lag scan (`kate analyze`, version-stable) and, for nonlinear slow CVs, from
VAMPnets [T6] on ligand-pocket contacts (not raw Cartesian). The bound, the
flow, the KATE pipeline, and the thermodynamics (state populations) are unaffected.

## Build targets (all implemented)

Classical / scaling track:
- **T1 [x]** path bound wired into the runner + `kate bound` (pure-numpy contrast scorer)
- **T2 [x]** production kinetics via deeptime (`kate analyze`: lag scan, Bayesian bars)
- **T3 [x]** baseline-comparison harness (`kate benchmark`, the contrast figure)
- **T4 [x]** full-atom reconstruction (`decompress --full-atom`, per-state dithered residual)
- **T5 [x]** scale to 419k→1M frames (`compress --streaming`, streaming TICA, multi-pass)

Neural-ML track (built in order T8 → T6 → T7; the flow stays
invertible and the bound intact, with no lossy CNN autoencoder):
- **T8 [x]** temporal and learned-entropy model (`--entropy temporal`). Codes latents
  against a causal learned conditional instead of the fixed N(0,I) base; this changes only
  the code length, not the flow or the bound (exactly lossless). This is the novel ML piece.
- **T6 [x]** learned nonlinear slow CVs via VAMPnets (`--cv vampnet`, deeptime; TICA drop-in)
- **T7 [x]** more expressive flow (`--flow spline`, rational-quadratic neural-spline
  coupling; tighter density, same invertibility)
- **T9 [x] (both halves measured on NTL9)** learned
  predictive entropy coding (`--entropy predictive`, `--predictor {gru,tcn}`), a
  lossy rate-distortion mode:
  a causal GRU predicts a conditional Gaussian for the next latent (bound-as-loss:
  conditional NLL is the transition-kernel surrogate, not MSE), and the standardized
  innovation `(z−μ)/σ` is quantized (subtractive dither) against a unit Gaussian; the
  bit-width is the rate knob. It is streaming-compatible (online GRU state). T8 is kept intact
  as the lossless head-to-head.
  Rate gain measured on real NTL9 latents ([`docs/ntl9_temporal_redundancy.png`](docs/ntl9_temporal_redundancy.png)): at equal
  distortion (fixed quantizer step), predictive coding cuts ~15 bits/frame (~43%) off
  static per-frame coding at the 0.5 ns storage stride (35→20 bits/frame, 8 CVs), and
  still ~14 bits at 1 ns. This corrects an earlier conservative hedge ("gain may be
  modest, frames are decorrelated at ~100 ps"): that holds for fast Cartesian modes, but
  KATE compresses slow CVs, whose µs timescales keep them strongly autocorrelated
  (ρ≈0.98) even at storage spacing, so the temporal redundancy the predictor exploits is
  large and real.
  Observable-error half ([`docs/ntl9_t9_gate.png`](docs/ntl9_t9_gate.png)): scored on
  the resolved band (the best-sampled sub-µs processes of the block, via
  `analyze --resolution`), with proper closed-loop DPCM reconstruction. The rate-vs-
  kinetics-error frontier for predictive sits far to the left of static; it preserves the
  resolved kinetics at ~half to a third the rate (e.g. 0.8% timescale error at 10
  bits/frame vs static's 1.5% at 26 bits/frame; ~9× lower error at matched rate). Thus
  both halves of the gate pass on real NTL9: predictive coding is both cheaper in rate
  and at least as faithful in resolved kinetics. Scope: a 1 µs block only
  marginally resolves these processes, but the static-vs-predictive comparison is a
  paired test on identical processes, so the relative result is robust; the predictor
  here is linear AR(1)/DPCM (a faithful, conservative stand-in for the GRU). This is empirical,
  not assumed.
  Prior art (cite, verify before paper): DPCM (Cutler 1952); learned hyperprior and
  autoregressive context models, Ballé et al., ICLR 2018 (arXiv:1802.01436); Minnen
  et al., NeurIPS 2018 (arXiv:1809.02736).
- **T10 [x] (mechanism shown on synthetic)** the kinetic path-bound made
  differentiable so it can serve as a training loss (`kate.bound_loss`): a VAMPnet-style
  soft state assignment makes the soft MSM, and the path-bound transition term
  `h(P‖Q)`, a smooth function of the network, so `loss = rate + λ·h(P‖Q)` trains a
  compressor to spend bits where they matter for kinetics rather than for coordinate error.
  This is the one place where the ML and the central idea become the same object. It is demonstrated
  in `examples/demo_bound_loss.py` on a controlled, well-sampled system (a low-amplitude
  slow folding coordinate hidden among high-amplitude fast noise): at equal total bit
  budget, raw-coordinate MSE (what SZ3/ZFP/MDZip minimize) spends 0 bits on the slow
  coordinate and its kinetic distortion is flat in budget, while the bound-as-loss
  protects the slow mode and drives kinetic distortion down ~100× at 4 bits/frame.
  Scope: this shows the mechanism on synthetic data. On real NTL9 CVs the
  bound-loss does not beat MSE, a measured negative: it does correctly concentrate
  bits on the slow TICA modes and starve the fast ones (the allocation mechanism works),
  but at equal budget its resolved-kinetics error is no better than uniform MSE, and at
  higher budget MSE wins (3.4% vs 13.8% timescale error). The cause is concrete and worth
  reporting: the real MSM discretizes with k-means over all 8 CVs, so zeroing the fast
  modes corrupts the clustering, whereas the differentiable soft-MSM surrogate treats
  those modes as irrelevant; surrogate and estimator disagree. Thus `bound_loss` is a
  differentiable surrogate whose training signal does not yet transfer to the k-means/MLE
  estimator on this system; aligning the two (soft states matched to the discretization)
  is the open problem. The certified kinetics still come from the deeptime reversible-MLE
  MSM and the path bound on hard states. Whether the surrogate beats MSE on real data is
  empirical, and here it did not. Concurrent, independent work (Zou, Lie & Marzouk,
  arXiv:2603.20467, March 2026) develops the same path-bound-as-loss idea for surrogate-SDE
  drift learning; T10 is positioned as concurrent, not first (see Relation to prior work).

Defaults remain `--cv tica --flow realnvp --entropy gaussian` so the tested baseline and
the kinetic bound are unchanged; the ML pieces are opt-ins.
The full test suite (`pytest`) passes; torch/deeptime-dependent tests auto-skip when
those libraries are absent.

## Repository layout

```
src/kate/        flow.py codec.py kinetic_codec.py pathbound.py kinetics_deeptime.py
                inspect_traj.py runner.py  (+ artifact.py cli.py __main__.py
                benchmark.py baselines.py temporal_prior.py vampnet_cv.py spline_flow.py
                predictive_coder.py bound_loss.py)
tests/          pytest suite (torch/deeptime tests use importorskip)
examples/       demo_pathbound.py demo_kinetic_codec.py demo_kate.py demo_bound_loss.py
                (the §2 checks)
docs/           REVIEW.md (physics + competitive audit), METHODS.md, NTL9 figures
```

The reference clones (`MDZip/`, `SZ3/`, `zfp/`, `deeptime/`, `bgflow/`, `bgmol/`,
`awesome-AI4MolConformation-MD/`) are not part of this repository; they are
references and baselines kept on disk and git-ignored.

## Baselines & data (cluster-side)

MDZip / SZ3 / ZFP build and run in their own environments and are invoked as
subprocesses for the `benchmark` contrast. The
trypsin-benzamidine trajectory (`~419,213` frames, 100 ps/frame, solvent-stripped,
nm) resides on the cluster, not in this repository.

The wrappers (`kate.baselines`) are pointed at the tools via environment variables:

| env var | points to |
|---|---|
| `KATE_SZ3_BIN` | the SZ3 compressor executable (e.g. `.../SZ3/build/tools/sz3/sz3`) |
| `KATE_ZFP_BIN` | the ZFP compressor executable (e.g. `.../zfp/build/bin/zfp`) |
| `KATE_MDZIP_DIR` | the MDZip repo directory (its own torch/lightning env) |

If unset (or the tool is not on `PATH`), the wrapper raises a clear
`BaselineUnavailable` and the method is skipped; the local pseudo-baselines
(`shuffle`, `quantize`) can be used to exercise the contrast mechanics anywhere.

The coordinate-baseline role is covered by SZ3 / ZFP / MDZip (the published tools
benchmarked against); an in-house coordinate codec
(`coord_quant`) is intentionally not vendored, as the baselines to beat in the paper are the published tools rather than an
in-house reimplementation.
