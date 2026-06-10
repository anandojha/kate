# epc — Ensemble-Preserving Compression of MD trajectories, with a kinetic bound

> **Thesis.** Ensemble-preserving compression does **not** preserve kinetics. Two
> ensembles with identical stationary distributions can have arbitrarily different
> rates. EPC adds a **path-distribution (trajectory) bound** —
> `KL(path) = ensemble term + transition term` — so that **kinetic** observables
> (timescales, MFPTs, k_on/k_off) are covered, not just static ones. The kinetic
> bound is the headline, **not** the architecture.

This repo packages a tested research codebase as an installable library + CLI: a
classical analysis-native codec, a from-scratch RealNVP normalizing flow, the
flow-based EPC codec, the **kinetic path bound** (the novel piece), and a
[deeptime](https://github.com/deeptime-ml/deeptime) MSM wrapper. `RECIPE.txt` is the
authoritative spec; `RELATED_WORK.txt` lists prior work and baselines.

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

## Honesty constraints (do not regress — see `RECIPE.txt` §5)

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
epc analyze    artifact -> kinetics    deeptime MSM: timescales, lag scan, Bayesian error bars
epc bound      artifact ref -> report  ensemble term, transition term, Pinsker pair/path bounds
epc benchmark  TOP DCD  -> table+plot  EPC vs MDZip vs SZ3 vs ZFP, each scored by the path bound
```

Module map: `compress = runner.py/codec.py`, `decompress = codec.py (+T4 residual)`,
`analyze = kinetics_deeptime.py`, `bound = pathbound.py`, `benchmark = benchmark.py`.
The artifact stores the run-aware **all-frame dtraj + k-means centers** (not just one
count matrix), so `analyze`/`benchmark` can re-estimate the MSM at **any** lag.

## Sanity checks (RECIPE §2 — all pass on CPU)

| RECIPE command            | here                                  | checks |
|---------------------------|---------------------------------------|--------|
| `python epc_flow.py`      | `python -m epc.flow`                  | invertibility ~1e-6, density ~1, wells recovered |
| `python demo_pathbound.py`| `python examples/demo_pathbound.py`   | ensemble term ~0 for both chains; transition term large for the ensemble-only chain |
| `python demo_kinetic_codec.py` | `python examples/demo_kinetic_codec.py` | range coder hits the MSM entropy-rate floor; kinetics recovered |
| `python kinetics_deeptime.py` | `python -m epc.kinetics_deeptime`  | reversible MLE MSM, lag scan, Bayesian error bars (needs `[kinetics]`) |
| `python demo_epc.py`      | `python examples/demo_epc.py`         | full flow-based pipeline + measured bound |

Run the test suite with `pytest` (torch/deeptime-dependent tests auto-skip if those
libraries are absent).

## Build targets (status)

Classical / scaling track — `RECIPE.txt` §4:
- **T1** wire the path bound into the runner + `epc bound`
- **T2** production kinetics via deeptime (`epc analyze`: lag scan, Bayesian bars)
- **T3** baseline-comparison harness (`epc benchmark`, the contrast figure)
- **T4** full-atom reconstruction (per-state dithered residual stage)
- **T5** scale to 419k→1M frames (streaming TICA, two-pass encode)

Neural-ML track — `RECIPE.txt` §4b (implemented in order **T8 → T6 → T7**, keeping
the flow invertible and the bound intact; **no lossy CNN autoencoder**):
- **T8** temporal + learned-entropy model — codes latents against a causal learned
  conditional instead of a fixed Gaussian base; changes only the *code length*, not
  the flow or the bound. *The defensibly-novel ML piece.*
- **T6** learned slow CVs via VAMPnets (nonlinear, TICA drop-in)
- **T7** more expressive flow (neural-spline / MAF coupling; tighter bound, same
  invertibility)

## Repository layout

```
src/epc/        flow.py codec.py kinetic_codec.py pathbound.py kinetics_deeptime.py
                inspect_traj.py runner.py  (+ artifact.py cli.py __main__.py
                benchmark.py baselines.py temporal_prior.py vampnet_cv.py spline_flow.py)
tests/          pytest suite (torch/deeptime tests use importorskip)
examples/       demo_pathbound.py demo_kinetic_codec.py demo_epc.py  (the §2 checks)
RECIPE.txt      authoritative build spec       RELATED_WORK.txt  prior work + baselines
```

The reference clones (`MDZip/`, `SZ3/`, `zfp/`, `deeptime/`, `bgflow/`, `bgmol/`,
`awesome-AI4MolConformation-MD/`) are **not** part of this repo — they are
references/baselines kept on disk and git-ignored.

## Baselines & data (cluster-side)

MDZip / SZ3 / ZFP build and run in their own environments and are invoked as
**subprocesses** for the `benchmark` contrast; see `RELATED_WORK.txt`. The
trypsin–benzamidine trajectory (`~419,213` frames, 100 ps/frame, solvent-stripped,
nm) lives on the cluster, not in this repo.
