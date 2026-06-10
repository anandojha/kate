# epc — Ensemble-Preserving Compression of MD trajectories, with a kinetic bound

[![CI](https://github.com/anandojha/epc/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/epc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![coverage 99%](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)](#tests)

> **Thesis.** Ensemble-preserving compression does **not** preserve kinetics. Two
> ensembles with identical stationary distributions can have arbitrarily different
> rates. EPC adds a **path-distribution (trajectory) bound** —
> `KL(path) = ensemble term + transition term` — so that **kinetic** observables
> (timescales, MFPTs, k_on/k_off) are covered, not just static ones. The kinetic
> bound is the headline, **not** the architecture.

This repo packages a tested research codebase as an installable library + CLI: a
classical analysis-native codec, a from-scratch RealNVP normalizing flow, the
flow-based EPC codec, the **kinetic path bound** (the novel piece), and a
[deeptime](https://github.com/deeptime-ml/deeptime) MSM wrapper. `RELATED_WORK.txt`
lists the prior work and baselines this builds on and differentiates against.

```bash
pip install git+https://github.com/anandojha/epc.git          # core
pip install "epc[kinetics] @ git+https://github.com/anandojha/epc.git"  # + deeptime
```

## What is honestly new

**Defensible headline:** *the first MD-trajectory compressor with a provable
**kinetic-observable** bound — unifying a generative density model (a normalizing
flow), a Markov dynamics model (an MSM), and entropy coding under one
path-distribution (KL/Pinsker) guarantee, and shown to preserve kinetics where
ensemble- and coordinate-bounded compressors do not.*

The novelty is the **end-to-end integration + the kinetic bound as the organizing
principle and contrast result** — *not* a claim that any single component is new.
Neural latent compression (MDZip, JCIM AE), error-bounded compression (SZ3/ZFP/MDZ),
MSM-as-entropy-coder, and flow-as-density (Boltzmann Generators) are all prior art.

## Results on NTL9 (measured, real trajectory)

Validated on the 25 µs NTL9 fast-folder, scored **only on the kinetically resolved band**
(`epc analyze --resolution`; the slow folding mode is sampling-limited and is *not* scored
— see honesty constraints). All numbers are empirical, with the limits stated.

- **The contrast** ([`docs/ntl9_contrast_resolved.png`](docs/ntl9_contrast_resolved.png)) —
  to preserve the resolved kinetics to <1 % timescale error, **EPC needs ~12 bits/frame;
  general-purpose error-bounded compressors need ~840 (SZ3) – 1400 (ZFP)** — a **~70–120×**
  rate gap — and when pushed to aggressive compression **SZ3/ZFP collapse the kinetics**
  (SZ3 → 95 % timescale error at 331 bits/frame). They spend bits on bounded *all-atom*
  error, blind to which coordinates carry the slow dynamics; EPC's bound targets the
  kinetics. (Different objectives — the honest axis is *rate needed for a given kinetic
  fidelity*, which is what is plotted.)
- **Predictive (T9) entropy coding** ([`docs/ntl9_temporal_redundancy.png`](docs/ntl9_temporal_redundancy.png),
  [`docs/ntl9_t9_gate.png`](docs/ntl9_t9_gate.png)) — slow CVs stay autocorrelated (ρ≈0.98)
  even at the 0.5 ns storage stride, so predictive coding saves **~15 bits/frame (~43 %)**
  over static at equal distortion, and on the resolved band preserves the kinetics at
  **~half to a third the rate** of static coding (~9× lower timescale error at matched rate).
- **Bit allocation that respects kinetics** ([`examples/demo_bound_loss.py`](examples/demo_bound_loss.py))
  — the differentiable path-bound (T10), trained as a loss, spends bits on the slow
  coordinate where raw-MSE spends none (mechanism shown on a controlled synthetic;
  inconclusive on sampling-limited NTL9, stated as such).

## Honesty constraints (do not regress)

- **Dropped:** "first error-bounded MD compressor" — false; SZ/ZFP/MDZ bound
  coordinates/QoI already. The genuine novelty is the **observable-space (KL/Pinsker)
  bound, specifically the kinetic (path) bound.**
- The **ensemble (static)** Pinsker bound does **not** cover kinetic observables.
  **Only the path-distribution bound** (`epc.pathbound`) does. The `bound` report
  labels which term covers what.
- **"Exact / invertible" is qualified:** the flow is an exact diffeomorphism and kept
  frames reconstruct exactly up to quantization, **but information-gain frame
  selection (IGFS) is a lossy step** — the three together are not "exact" unqualified.
- Validation system is **trypsin–benzamidine** (not the kinase runs from the
  abstract); use it for the kinetics claims.
- Compression ratio (~8× at 4-bit) is **not** the headline — it is comparable to
  plain quantization. The differentiators are analysis-nativeness + the bound.
- The flow / MSM / coder / IGFS are **machinery, not claimed as invented**.
  Contribution = framing + the path/kinetic bound + the application + the contrast.
- **ML track:** the architecture is not the headline. Among the neural pieces only
  the **temporal learned-entropy model (T8)** is positioned as novel here; VAMPnets,
  spline flows, MAF, equivariant flows, and learned entropy models are reused prior
  art. **Verify all method names/citations before the paper.**

## Citation ≠ license compliance

Citing a paper is an academic courtesy. **Copying code** is a separate obligation
governed by that project's *software license* (attribution, possibly copyleft that
could relicense your repo). This project **imports** mature libraries and
**reimplements** algorithms from their papers — so **we cite, we do not relicense.**
No third-party source (deeptime, MDZip, SZ3, ZFP) is vendored into the repo.

Provenance boundary:
- **Reimplemented from the method (ours):** flow, entropy coder, path bound, IGFS,
  artifact format, CLI, benchmark, and the T7/T8 ML pieces.
- **Imported, never copied:** `deeptime` (reversible-MLE MSM, BayesianMSM, VAMPnets,
  streaming covariance) and `numpy/scipy/scikit-learn/torch/mdtraj`.
- **External baselines, run as subprocesses:** MDZip / SZ3 / ZFP (never vendored).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .              # core: numpy, scipy, scikit-learn, torch, mdtraj
pip install -e ".[kinetics]"  # adds deeptime + matplotlib (analyze / benchmark / VAMPnet)
pip install -e ".[test]"      # pytest
```

`torch` is a **core** dependency (the flow needs it). `deeptime` is an **optional
`[kinetics]` extra**: `epc compress` / `decompress` / `bound` run without it; only
`analyze` / `benchmark` / the VAMPnet CV path import it (and raise a clear
`pip install epc[kinetics]` if absent). Importing `epc` pulls in **neither** torch
nor deeptime eagerly — enforced by `tests/test_no_eager_torch.py`.

> **macOS note.** Use a *fully isolated* venv (as above). A `--system-site-packages`
> venv that mixes a conda **MKL** numpy with pip torch's libomp can **segfault** from
> duplicate OpenMP runtimes. A clean venv pulls a wheel-based numpy (Apple
> Accelerate / OpenBLAS) and avoids it.

## The target: one end-to-end tool

```
epc compress   TOP DCD  -> artifact    align -> CV/flow -> IGFS -> entropy code + retained MSM
epc decompress artifact -> trajectory  flow inverse for kept frames; full-atom residual stage
epc analyze    artifact -> kinetics    deeptime MSM: timescales, lag scan, Bayesian bars, --resolution
epc bound      artifact ref -> report  ensemble term, transition term, Pinsker pair/path bounds
epc benchmark  TOP DCD  -> table+plot  EPC vs MDZip vs SZ3 vs ZFP, each scored by the path bound
```

Module map: `compress = runner.py/codec.py`, `decompress = codec.py (+T4 residual)`,
`analyze = kinetics_deeptime.py`, `bound = pathbound.py`, `benchmark = benchmark.py`.
The artifact stores the run-aware **all-frame dtraj + k-means centers** (not just one
count matrix), so `analyze`/`benchmark` can re-estimate the MSM at **any** lag.

`epc analyze --resolution` adds a **kinetic-resolution report**: per dynamical process,
the Bayesian timescale, its 95% confidence interval, the relative uncertainty, and the
number of *independent events* the trajectory contains (~ T_total / t_i). A process is
flagged *resolved* only if its Bayesian error is small **and** it has enough events —
because no compressor can preserve a kinetic observable the **source** trajectory never
sampled. On the 25 µs NTL9 set this correctly reports the slow folding modes (> ~3 µs,
< a handful of events) as **not resolved** and the faster band as resolved — so any
kinetic claim is held to what the data actually supports, not to the slowest eigenvalue
the estimator happens to return. This honesty step is usually omitted in the
MD-compression literature.

## Sanity checks (all pass on CPU)

| original script           | here (packaged)                       | checks |
|---------------------------|---------------------------------------|--------|
| `python epc_flow.py`      | `python -m epc.flow`                  | invertibility ~1e-6, density ~1, wells recovered |
| `python demo_pathbound.py`| `python examples/demo_pathbound.py`   | ensemble term ~0 for both chains; transition term large for the ensemble-only chain |
| `python demo_kinetic_codec.py` | `python examples/demo_kinetic_codec.py` | range coder hits the MSM entropy-rate floor; kinetics recovered |
| `python kinetics_deeptime.py` | `python -m epc.kinetics_deeptime`  | reversible MLE MSM, lag scan, Bayesian error bars (needs `[kinetics]`) |
| `python demo_epc.py`      | `python examples/demo_epc.py`         | full flow-based pipeline + measured bound |

Run the test suite with `pytest` (torch/deeptime-dependent tests auto-skip if those
libraries are absent).

### Reproducibility note (kinetics)

The **crude classical estimator** in `kinetic_codec` (a single-lag `(C+C^T)/2` MSM on
TICA of aligned **Cartesian** coordinates) is featurization-limited and its recovered
timescales are **library-version sensitive** — on the newest numpy/scipy/sklearn the
synthetic demo's slow timescales are under-resolved (the leading TICA mode on raw
Cartesian is spurious). This is expected and is exactly why we make deeptime
the published path: implied timescales are a **lower bound that converges upward with
lag** (Prinz et al.), so the rigorous kinetics come from a **deeptime reversible-MLE
MSM + lag scan** (`epc analyze`, version-stable) and, for nonlinear slow CVs, from
**VAMPnets [T6]** on **ligand-pocket contacts** (not raw Cartesian). The bound, the
flow, the EPC pipeline, and the thermodynamics (state populations) are unaffected.

## Build targets (all implemented)

Classical / scaling track:
- **T1 ✓** path bound wired into the runner + `epc bound` (pure-numpy contrast scorer)
- **T2 ✓** production kinetics via deeptime (`epc analyze`: lag scan, Bayesian bars)
- **T3 ✓** baseline-comparison harness (`epc benchmark`, the contrast figure)
- **T4 ✓** full-atom reconstruction (`decompress --full-atom`, per-state dithered residual)
- **T5 ✓** scale to 419k→1M frames (`compress --streaming`, streaming TICA, multi-pass)

Neural-ML track (built in order **T8 → T6 → T7**, the flow stays
invertible and the bound intact; **no lossy CNN autoencoder**):
- **T8 ✓** temporal + learned-entropy model (`--entropy temporal`) — codes latents
  against a causal learned conditional instead of the fixed N(0,I) base; changes only
  the *code length*, not the flow or the bound (exactly lossless). *The novel-ML piece.*
- **T6 ✓** learned nonlinear slow CVs via VAMPnets (`--cv vampnet`, deeptime; TICA drop-in)
- **T7 ✓** more expressive flow (`--flow spline`, rational-quadratic neural-spline
  coupling; tighter density, same invertibility)
- **T9 ✓ (both halves measured on NTL9)** learned
  *predictive* entropy coding (`--entropy predictive`, `--predictor {gru,tcn}`) — a
  **lossy** rate-distortion mode:
  a causal GRU predicts a conditional Gaussian for the next latent (bound-as-loss:
  conditional NLL = the transition-kernel surrogate, not MSE), and the **standardized
  innovation** `(z−μ)/σ` is quantized (subtractive dither) against a unit Gaussian; the
  bit-width is the rate knob. Streaming-compatible (online GRU state). T8 is kept intact
  as the lossless head-to-head.
  **Rate gain measured on real NTL9 latents** ([`docs/ntl9_temporal_redundancy.png`](docs/ntl9_temporal_redundancy.png)): at equal
  distortion (fixed quantizer step), predictive coding cuts **~15 bits/frame (~43%)** off
  static per-frame coding at the 0.5 ns storage stride (35→20 bits/frame, 8 CVs), and
  still ~14 bits at 1 ns. This *corrects an earlier conservative hedge* ("gain may be
  modest, frames are decorrelated at ~100 ps"): that holds for fast Cartesian modes, but
  EPC compresses **slow** CVs, whose µs timescales keep them strongly autocorrelated
  (ρ≈0.98) even at storage spacing — so the temporal-redundancy the predictor exploits is
  large and real.
  **Observable-error half** ([`docs/ntl9_t9_gate.png`](docs/ntl9_t9_gate.png)): scored on
  the resolved band (the best-sampled sub-µs processes of the block, via
  `analyze --resolution`), with proper closed-loop DPCM reconstruction. The rate-vs-
  kinetics-error frontier for predictive sits far left of static — it preserves the
  resolved kinetics at **~half to a third the rate** (e.g. 0.8% timescale error at 10
  bits/frame vs static's 1.5% at 26 bits/frame; ~9× lower error at matched rate). So
  **both halves of the gate pass on real NTL9**: predictive coding is both cheaper (rate)
  and at least as faithful (resolved kinetics). Honest scope: a 1 µs block only
  marginally resolves these processes, but the static-vs-predictive comparison is a
  *paired* test on identical processes, so the relative result is robust; the predictor
  here is linear AR(1)/DPCM (a faithful, conservative stand-in for the GRU). Empirical,
  never assumed.
  Prior art (cite, verify before paper): DPCM (Cutler 1952); learned hyperprior /
  autoregressive context models — Ballé et al., ICLR 2018 (arXiv:1802.01436); Minnen
  et al., NeurIPS 2018 (arXiv:1809.02736).
- **T10 ✓ (mechanism shown on synthetic)** the kinetic path-bound made
  **differentiable so it can be a training loss** (`epc.bound_loss`): a VAMPnet-style
  *soft* state assignment makes the soft MSM — and the path-bound transition term
  `h(P‖Q)` — a smooth function of the network, so `loss = rate + λ·h(P‖Q)` trains a
  compressor to spend bits where they matter for *kinetics*, not for coordinate error.
  This is the one place the ML and the novel idea become the same object. Demonstrated
  in `examples/demo_bound_loss.py` on a controlled, well-sampled system (a low-amplitude
  slow folding coordinate hidden among high-amplitude fast noise): at equal total bit
  budget, raw-coordinate MSE (what SZ3/ZFP/MDZip minimize) spends **0 bits** on the slow
  coordinate and its kinetic distortion is **flat in budget**, while the bound-as-loss
  protects the slow mode and drives kinetic distortion down **~100× at 4 bits/frame**.
  *Honest scope:* this shows the **mechanism** on synthetic; on NTL9 it is **inconclusive**
  (sampling-limited — ~6 folding events make the transition term noise-dominated, so the
  bound gives little training signal). `bound_loss` is the differentiable **surrogate**
  used as a loss; the certified kinetics still come from the deeptime reversible-MLE MSM
  + the path bound on hard states. Whether it beats MSE on real data is **empirical** and
  must be shown on a well-sampled system — not assumed.

Defaults stay `--cv tica --flow realnvp --entropy gaussian` so the tested baseline and
the headline (the kinetic bound) are unchanged; the ML pieces are motivated opt-ins.
The full test suite (`pytest`) is green; torch/deeptime-dependent tests auto-skip when
those libraries are absent.

## Repository layout

```
src/epc/        flow.py codec.py kinetic_codec.py pathbound.py kinetics_deeptime.py
                inspect_traj.py runner.py  (+ artifact.py cli.py __main__.py
                benchmark.py baselines.py temporal_prior.py vampnet_cv.py spline_flow.py
                predictive_coder.py bound_loss.py)
tests/          pytest suite (torch/deeptime tests use importorskip)
examples/       demo_pathbound.py demo_kinetic_codec.py demo_epc.py demo_bound_loss.py
                (the §2 checks)
RELATED_WORK.txt  prior work + baselines this builds on and differentiates against
```

The reference clones (`MDZip/`, `SZ3/`, `zfp/`, `deeptime/`, `bgflow/`, `bgmol/`,
`awesome-AI4MolConformation-MD/`) are **not** part of this repo — they are
references/baselines kept on disk and git-ignored.

## Baselines & data (cluster-side)

MDZip / SZ3 / ZFP build and run in their own environments and are invoked as
**subprocesses** for the `benchmark` contrast; see `RELATED_WORK.txt`. The
trypsin–benzamidine trajectory (`~419,213` frames, 100 ps/frame, solvent-stripped,
nm) lives on the cluster, not in this repo.

Point the wrappers (`epc.baselines`) at the tools via environment variables:

| env var | points to |
|---|---|
| `EPC_SZ3_BIN` | the SZ3 compressor executable (e.g. `.../SZ3/build/tools/sz3/sz3`) |
| `EPC_ZFP_BIN` | the ZFP compressor executable (e.g. `.../zfp/build/bin/zfp`) |
| `EPC_MDZIP_DIR` | the MDZip repo directory (its own torch/lightning env) |

If unset (or the tool isn't on `PATH`), the wrapper raises a clear
`BaselineUnavailable` and the method is skipped — use the local pseudo-baselines
(`shuffle`, `quantize`) to exercise the contrast mechanics anywhere.

The **coordinate-baseline role is covered by SZ3 / ZFP / MDZip** (the published tools
we benchmark against); we intentionally do **not** vendor an in-house coordinate codec
(`coord_quant`) — the baselines to beat in the paper are the published tools, not our
own reimplementation.
