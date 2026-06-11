# GLIDE - Full Code Review (physics validity + competitive comparison)

A four-axis review of GLIDE against the reference packages it builds on or competes with
(**bgflow/bgmol**, **deeptime**, **mdzip**, **sz3/zfp**), plus a physics-correctness audit
of the core. Every algorithm was examined and, where possible, checked numerically. This
document records what is correct, what is over-stated, and what is missing, so that the
project's claims correspond to its code.

---

## Bottom line

- **The core information theory and ML machinery are correct and carefully implemented**,
  verified analytically and numerically: the path-KL factorization, Pinsker, the Markov entropy rate,
  the Witten-Neal-Cleary arithmetic coder, the RealNVP/spline change-of-variables and exact
  invertibility, closed-loop DPCM, and VAMP-2. No sign errors and no broken math were found.
- **The principal weaknesses are over-stated guarantees and one biased estimator in the production path**,
  rather than bugs. Three convergent findings (below) are the most significant.
- **On pure compression (rate, all-atom RMSD, speed, maturity) GLIDE is the weakest of the
  four.** Its defensible advantage is narrow but genuine: it is the only one of the four that retains
  kinetics (the MSM), ships a path-distribution (KL→Pinsker) bound on kinetic observables, and
  whose artifact is analysis-native (the file is the kinetic model). This is novel, but it does
  not constitute a pure-compression advantage, and the repository should not imply that it does.

---

## A. What is correct & physically sound (the foundation holds)

- **Path-space KL decomposition** (`pathbound.py:81-124`): `D(ρ_P‖ρ_Q) = D(μ_P‖μ_Q) + Σ_i μ_P(i) D(P(i,·)‖Q(i,·))` is the exact lag-τ joint divergence, correctly weighted by the reference stationary μ_P. Verified to 1e-16.
- **Pinsker** (`pathbound.py:127-130`): `|E_P[g]−E_Q[g]| ≤ √(KL/2)` for g∈[0,1]; the constant and direction are correct.
- **Kabsch alignment** (`kinetic_codec.py:256-275`): centering + SVD + reflection fix `det=+1`. Standard and correct.
- **Reversible `(C+Cᵀ)/2` estimator** (`kinetic_codec.py:457-469`): genuinely detailed-balanced (DB violation ~1e-18). A valid reversible estimator, though not the MLE (see B1).
- **Markov entropy rate** (`kinetic_codec.py:506-512`): Ekroot-Cover `H=−Σ_i π_i Σ_j T_ij log₂T_ij`. Correct.
- **WNC range coder** (`kinetic_codec.py:152-249`, `codec.py:54-107`, `temporal_prior.py:156-208`): textbook E1/E2/E3 renormalization; integer cumfreq tables keep encode and decode in sync, and the losslessness logic is sound.
- **RealNVP & RQ-spline flows** (`flow.py:86-105`, `spline_flow.py`): log-det correct, inverse exact (max err 0), spline C1-continuous with identity tails, monotone⟹invertible. Durkan et al. implemented correctly.
- **Closed-loop DPCM** (`predictive_coder.py:190-236`): both sides feed the reconstructed latent to the predictor, so no drift occurs. Sound.
- **deeptime usage** (`kinetics_deeptime.py`): correct count modes (sliding/effective), connectivity, reversible MLE, ITS lag-scan, Bayesian errors. `vampnet_cv.py` uses deeptime's `VAMPNet` rather than a reimplementation.
- **Units/time** (`runner.py:59,109,336`): `dt_strided_ns`, frame↔ns↔lag are all dimensionally consistent. No spurious kT appears, which is correct, as the treatment is entirely empirical.

## B. Critical findings (prioritized)

**B1 - [HIGH] The "certified" kinetics are produced by the biased `(C+Cᵀ)/2` estimator rather than deeptime, contrary to the docstrings.**
The production path (`runner.py:104-110`), the benchmark contrast (`benchmark.py:62-84`), and `glide bound` (`cli.py:124-125`) all estimate the MSM with the in-repo `(C+Cᵀ)/2` symmetrization. That estimator is statistically biased: it is not the reversible MLE, and it pulls the stationary distribution toward uniform when state populations are unequal, which is precisely the metastable regime GLIDE targets. However, `kinetics_deeptime.py:4-7` and `bound_loss.py:23-27` state that the publishable numbers come from deeptime's reversible MLE. The deeptime path is built (`msm_for_pathbound`) but bypassed. → Route the reported timescales, stored artifact MSM, and path-bound matrices through `MaximumLikelihoodMSM(reversible=True)` when deeptime is available; retain `(C+Cᵀ)/2` only as the entropy-coding model, where only row-stochasticity matters. The core-vs-`[kinetics]`-extra split must be respected: fall back to `(C+Cᵀ)/2` when deeptime is absent, and record which estimator produced the numbers.

**B2 - [HIGH] IGFS frame selection produces a tail-biased ensemble presented under an ensemble-Pinsker label.**
`igfs_select` (`codec.py:125-139`) is farthest-point sampling in latent space; it over-represents tails (verified: kept std 1.70 vs true 1.00 on N(0,1)), and `decode_ensemble` (`codec.py:231-243`) returns that biased subset without reweighting back to the stationary measure. The stage-4 docstring (`codec.py:21-25`) advertises a Pinsker ensemble bound on it. A mitigation is present: `runner.py:93-100` runs the ensemble Pinsker check against flow samples, which is valid, but `codec.py` overstates what the kept subset guarantees. → Either reweight kept frames to the Boltzmann/stationary measure on decode, or cease describing the raw IGFS subset as bounding ensemble averages.

**B3 - [MED] The transition (kinetic) bound is reported as a finite number even when the true divergence is +∞.**
`transition_kl_rate` clips Q (`pathbound.py:90-93`); when Q has a structural zero where P>0 (the exact signature of a missed transition, where kinetics are broken), the true `h(P‖Q)=+∞` but a large finite value is returned, and `report_kinetic_fidelity` historically reported a finite Pinsker bound regardless of `support_ok`. Fixed in this pass (`pathbound.py`: added `kinetic_bound_valid`, `transition_kl_is_lower_bound`, and `pinsker_pair_bound=inf` on support failure; `cli.py` now prints an explicit warning). The remaining work is to make `benchmark.py` honor the flag as well.

**B4 - [HIGH-value gap] GLIDE promises k_on/k_off and MFPTs but computes neither.**
`pathbound.py:14-18` advertises preservation of "MFPTs, k_on/k_off," but no MFPT, committor, or reactive flux is computed. deeptime provides all of them (`MarkovStateModel.mfpt`, `committor_forward/backward`, `reactive_flux`). → Report MFPT(reactant→product) on P vs Q in `benchmark.py`/`cmd_bound`. This converts the abstract pairwise bound into the rate language the field cites, addressing the largest claim-vs-measurement gap.

**B5 - [MED] The in-repo benchmark contrast is a conceptual demonstration using non-representative baselines, not a compression comparison.**
`benchmark.py` hard-codes `Q=P` for GLIDE (`benchmark.py:70-73`) so its transition term is identically 0, and the default baselines are local stand-ins (`shuffle` = i.i.d. resample, `quantize` = round to 1 decimal); the real ones are not run (`run_mdzip` is a stub, `baselines.py:115-121`; SZ3/ZFP are unverified scaffolds). The table contains no rate, RMSD, or speed axis. The code documents this in comments, but a reader could mistake "GLIDE's transition term ≈ 0" for "GLIDE compresses better," which it does not establish. The real-data NTL9 contrast in `docs/ntl9_contrast_resolved.png` does run real SZ3/ZFP binaries and constitutes the credible comparison; the in-repo harness does not. → Add a rate-distortion-kinetics-speed table and run the real baselines.

## C. Honest competitive positioning (per package)

- **vs deeptime** - GLIDE uses it correctly for `analyze`, but reimplements TICA, the reversible MSM, stationary distribution, timescales, and connectivity (`kinetic_codec.py:326-503`) and then runs the in-repo implementations in the production path (B1). The in-repo TICA is competent (proper generalized eigenproblem, streaming, sign-canonicalization) but redundant. The `bound_loss.py` soft-MSM reimplementation is justified, since it must be differentiable. Missing and high-value: MFPT/committor/TPT, PCCA+ metastable coarse-graining, hidden MSMs (robust to the discretization non-convergence the lag-scan repeatedly encounters), and VAMP/Koopman to verify rather than assume reversibility.
- **vs bgflow/bgmol** - GLIDE's from-scratch flows are not an inferior bgflow; they are deliberately minimal flows on low-dim CVs (2-10D), with no heavy dependencies, deterministic, and self-tested; for that regime coupling RealNVP/spline is adequate. Genuine gaps: (1) no periodic/circular handling - `spline_flow.py:96-104` uses linear tails, so any angular CV is mishandled (bgflow's `is_circular` spline ties boundary slopes); (2) no energy/Boltzmann validation - `decode_ensemble` returns coordinates with no clash/energy check, so a reconstruction can satisfy the bound yet be sterically invalid; (3) no internal-coordinate (bond/angle/torsion) representation, so reconstructed Cartesians need not respect bond geometry; (4) no equivariant/CNF flows (correctly a non-goal for CV-space). bgmol offers OpenMM systems, Z-matrix factories, and reference datasets that GLIDE reimplements or lacks.
- **vs MDZip** - MDZip is a 2D-conv autoencoder trained on pure coordinate RMSD (`mdzip/AE.py:12-17`), storing one ~20-D latent for the whole frame → uint8 → LZMA. On storage ratio and all-atom RMSD over all frames, MDZip very likely outperforms GLIDE (GLIDE only reconstructs coordinates for the ~10% kept frames). GLIDE's assertion that "a lossy AE destroys kinetics" is a plausible, correctly-diagnosed hypothesis (MDZip has no dynamics term) but is unmeasured in this repository; furthermore, MDZip is a coordinate-RMSD method, not an ensemble-preserving one, and should be described as such.
- **vs SZ3/ZFP** - SZ3 ships an MD-specific spatio-temporal predictor (`SZBioMDDecomposition.hpp`), a hard per-atom L∞ error bound with an unpredictable-value escape, Huffman + zstd backends, and C++ throughput; ZFP adds O(1) random access (fixed-rate) and GPU/OpenMP codecs. On all-atom fidelity-at-rate and speed they win decisively. They have no notion of slow CVs/MSMs/kinetics, which is precisely GLIDE's purpose, but their coordinate machinery is considerably more sophisticated than GLIDE's flat-bit residual.

## D. Prioritized roadmap ("best of all worlds")

> **Resolution status (updated):** Tier 1 items 1, 3, 4 and Tier 2 items 5, 6, 7, 8 are
> DONE (commits `294feac`, `2dd91fa`, `4276d07`, `1ed16e6`, `f909185`). Remaining: B2
> (IGFS↔ensemble reconciliation) and Tier 3 (compression competitiveness).

**Tier 1 - correctness & honesty (address first; mostly bounded):**
1. ✅ **DONE** - **Routed certified kinetics through deeptime's reversible MLE** (B1,
   `294feac`): `estimate_reversible_T` prefers the deeptime MLE and falls back to `(C+Cᵀ)/2`
   when deeptime is absent; runner and benchmark use it; the artifact records `msm_estimator`.
   `glide bound` remains the portable (C+Cᵀ)/2 scorer, since deeptime pulls in torch.
2. ☐ **OPEN** - **Reconcile IGFS with the ensemble bound** (B2): reweight kept frames to
   the stationary measure on decode, or restate what the IGFS subset guarantees. (The
   runner already runs the valid ensemble check against flow samples; the gap is the
   `codec.py` decode_ensemble framing.)
3. ✅ **DONE** - **`kinetic_bound_valid`** gating (B3, `2dd91fa`): `pathbound`/`cli` report
   the bound as invalid (Pinsker = inf) on support failure. (Still to extend to `benchmark`.)
4. ✅ **DONE** - **Tightened claims** (`2dd91fa`): honest competitive positioning in README.

**Tier 2 - physics rigor (high value):**
5. ✅ **DONE** - **MFPT** reporting (B4, `4276d07`): `glide analyze --mfpt N` computes mean
   first-passage times between PCCA+ metastable sets (k_on/k_off ~ 1/MFPT).
6. ✅ **DONE** - **PCCA+ metastable coarse-graining** (`4276d07`): bundled with the MFPT report.
7. ✅ **DONE** - **Internal-coordinate (contacts) featurization** (`1ed16e6`):
   `glide compress --features contacts` (invariant inter-atomic distances; removes spurious
   rigid-body slow modes). Opt-in; the default remains cartesian to preserve the validated baseline.
8. ✅ **DONE** - **Steric-validity check** (`f909185`): force-field-free min-inter-atomic-distance
   check on decoded frames flags reconstruction-introduced atom overlaps. (A full OpenMM
   energy hook remains optional future work.)

**Tier 3 - compression competitiveness (if pure-ratio matters):**
9. **Entropy-code the residual** instead of charging a flat bit rate (`kinetic_codec.py:682` itself describes the flat rate as an "upper bound"); this represents the largest unrealized compression ratio.
10. **Spatio-temporal residual predictor + outlier escape** (adapting SZ3's space-time Lorenzo) for better all-atom fidelity-at-rate.
11. **zstd final pass + vectorized/compiled coders** - the pure-Python per-symbol coders are orders of magnitude slower than the C++ tools.
12. **Periodic/circular spline support** - only if angular CVs are used; otherwise document the non-periodic assumption.

**Net:** the foundation is solid and the novelty (kinetic bound + analysis-native artifact) is genuine. The work that most improves physical rigor and honesty is Tier 1-2: ensure the certified path uses the publishable estimator, measure the rate observables (MFPT) the project already promises, and scope the claims to what the code demonstrates.
