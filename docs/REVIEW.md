# GLIDE — Full Code Review (physics validity + competitive comparison)

A four-axis review of GLIDE against the reference packages it builds on or competes with
(**bgflow/bgmol**, **deeptime**, **mdzip**, **sz3/zfp**), plus a physics-correctness audit
of the core. Every algorithm was read and, where possible, checked numerically. This
document is intentionally blunt: it records what is *correct*, what is *over-stated*, and
what is *missing* — so the project's claims match its code.

---

## Bottom line

- **The core information theory and ML machinery are correct and carefully done** — verified
  analytically and numerically: the path-KL factorization, Pinsker, the Markov entropy rate,
  the Witten–Neal–Cleary arithmetic coder, the RealNVP/spline change-of-variables and exact
  invertibility, closed-loop DPCM, and VAMP-2. No sign errors, no broken math.
- **The real weaknesses are over-stated guarantees and one biased estimator in the hot path**,
  not bugs. Three convergent findings (below) matter most.
- **On *pure compression* (rate, all-atom RMSD, speed, maturity) GLIDE is the weakest of the
  four.** Its genuine, defensible win is narrow and real: it is the *only* one that retains
  kinetics (the MSM), ships a path-distribution (KL→Pinsker) bound on kinetic observables, and
  whose artifact is analysis-native (the file *is* the kinetic model). That is novel — it is
  just **not** a pure-compression win, and the repo should never imply it is.

---

## A. What is correct & physically sound (the foundation holds)

- **Path-space KL decomposition** (`pathbound.py:81-124`): `D(ρ_P‖ρ_Q) = D(μ_P‖μ_Q) + Σ_i μ_P(i) D(P(i,·)‖Q(i,·))` is the exact lag-τ joint divergence, correctly weighted by the *reference* stationary μ_P. Verified to 1e-16.
- **Pinsker** (`pathbound.py:127-130`): `|E_P[g]−E_Q[g]| ≤ √(KL/2)` for g∈[0,1] — constant and direction right.
- **Kabsch alignment** (`kinetic_codec.py:256-275`): centering + SVD + reflection fix `det=+1`. Standard and correct.
- **Reversible `(C+Cᵀ)/2` estimator** (`kinetic_codec.py:457-469`): genuinely detailed-balanced (DB violation ~1e-18). A *valid reversible* estimator — just not the MLE (see B1).
- **Markov entropy rate** (`kinetic_codec.py:506-512`): Ekroot–Cover `H=−Σ_i π_i Σ_j T_ij log₂T_ij`. Correct.
- **WNC range coder** (`kinetic_codec.py:152-249`, `codec.py:54-107`, `temporal_prior.py:156-208`): textbook E1/E2/E3 renormalization; integer cumfreq tables keep encode/decode in sync — losslessness logic sound.
- **RealNVP & RQ-spline flows** (`flow.py:86-105`, `spline_flow.py`): log-det correct, inverse exact (max err 0), spline C1-continuous with identity tails, monotone⟹invertible. Durkan et al. implemented correctly.
- **Closed-loop DPCM** (`predictive_coder.py:190-236`): both sides feed the *reconstructed* latent to the predictor — no drift. Sound.
- **deeptime usage** (`kinetics_deeptime.py`): correct count modes (sliding/effective), connectivity, reversible MLE, ITS lag-scan, Bayesian errors. `vampnet_cv.py` uses deeptime's real `VAMPNet`, not a hand-roll.
- **Units/time** (`runner.py:59,109,336`): `dt_strided_ns`, frame↔ns↔lag all dimensionally consistent. No spurious kT (correct — everything is empirical).

## B. Critical findings (prioritized — these are the ones that matter)

**B1 — [HIGH] The "certified" kinetics come from the biased `(C+Cᵀ)/2` estimator, not deeptime — the opposite of what the docstrings claim.**
The production path (`runner.py:104-110`), the benchmark contrast (`benchmark.py:62-84`), and `glide bound` (`cli.py:124-125`) all estimate the MSM with the hand-rolled `(C+Cᵀ)/2` symmetrization. That estimator is **statistically biased** (it is not the reversible MLE; it pulls the stationary distribution toward uniform when state populations are unequal — exactly the metastable regime GLIDE targets). Yet `kinetics_deeptime.py:4-7` and `bound_loss.py:23-27` say the publishable numbers come from deeptime's reversible MLE. **The deeptime path is built (`msm_for_pathbound`) but bypassed.** → Route the *reported* timescales / stored artifact MSM / path-bound matrices through `MaximumLikelihoodMSM(reversible=True)` when deeptime is available; keep `(C+Cᵀ)/2` only as the entropy-coding model (where only row-stochasticity matters). Mind the core-vs-`[kinetics]`-extra split: fall back to `(C+Cᵀ)/2` when deeptime is absent, and say which estimator produced the numbers.

**B2 — [HIGH] IGFS frame selection ships a deliberately tail-biased ensemble under an ensemble-Pinsker label.**
`igfs_select` (`codec.py:125-139`) is farthest-point sampling in latent space; it over-represents tails (verified: kept std 1.70 vs true 1.00 on N(0,1)), and `decode_ensemble` (`codec.py:231-243`) returns that biased subset **without reweighting** back to the stationary measure. The stage-4 docstring (`codec.py:21-25`) markets a Pinsker ensemble bound on it. (Mitigation that *is* present: `runner.py:93-100` runs the ensemble Pinsker check against **flow samples**, which is valid — but `codec.py` overstates what the kept subset guarantees.) → Either reweight kept frames to the Boltzmann/stationary measure on decode, or stop describing the raw IGFS subset as bounding ensemble averages.

**B3 — [MED] The transition (kinetic) bound is reported as a finite number even when the true divergence is +∞.**
`transition_kl_rate` clips Q (`pathbound.py:90-93`); when Q has a structural zero where P>0 (the exact signature of a *missed transition* — kinetics broken), the true `h(P‖Q)=+∞` but a large finite value is returned, and `report_kinetic_fidelity` historically reported a finite Pinsker bound regardless of `support_ok`. **Fixed in this pass** (`pathbound.py`: added `kinetic_bound_valid`, `transition_kl_is_lower_bound`, and `pinsker_pair_bound=inf` on support failure; `cli.py` now prints an explicit warning). The remaining work is to make `benchmark.py` honor the flag too.

**B4 — [HIGH-value gap] GLIDE promises k_on/k_off & MFPTs but never computes one.**
`pathbound.py:14-18` advertises preservation of "MFPTs, k_on/k_off," but nothing computes an MFPT, committor, or reactive flux. deeptime has all of them (`MarkovStateModel.mfpt`, `committor_forward/backward`, `reactive_flux`). → Report MFPT(reactant→product) on P vs Q in `benchmark.py`/`cmd_bound`. This converts the abstract pairwise bound into the rate language the field actually cites — the single biggest claim-vs-measurement gap.

**B5 — [MED] The in-repo benchmark contrast is a *conceptual* demo with strawmen, not a compression comparison.**
`benchmark.py` hard-codes `Q=P` for GLIDE (`benchmark.py:70-73`) so its transition term is identically 0, and the default baselines are local stand-ins (`shuffle` = i.i.d. resample, `quantize` = round to 1 decimal); the real ones aren't run (`run_mdzip` is a stub, `baselines.py:115-121`; SZ3/ZFP are unverified scaffolds). There is **no rate / RMSD / speed axis** in the table. The code is honest about this in comments, but a reader could mistake "GLIDE's transition term ≈ 0" for "GLIDE compresses better," which it does not establish. (The real-data NTL9 contrast in `docs/ntl9_contrast_resolved.png` *does* run real SZ3/ZFP binaries — that is the credible comparison; the in-repo harness is not.) → Add a rate–distortion–kinetics–speed table and run the real baselines.

## C. Honest competitive positioning (per package)

- **vs deeptime** — GLIDE *uses* it correctly for `analyze`, but **reimplements TICA, the reversible MSM, stationary dist, timescales, connectivity** (`kinetic_codec.py:326-503`) and then runs the *hand-rolled* ones in the hot path (B1). The hand-rolled TICA is competent (proper generalized eigenproblem, streaming, sign-canonicalization) but redundant. The `bound_loss.py` soft-MSM hand-roll *is* justified (must be differentiable). **Missing & high-value:** MFPT/committor/TPT, PCCA+ metastable coarse-graining, hidden MSMs (robust to the discretization non-convergence the lag-scan keeps hitting), VAMP/Koopman to *check* (not assume) reversibility.
- **vs bgflow/bgmol** — GLIDE's from-scratch flows are **not a worse bgflow** — they are deliberately minimal flows on *low-dim CVs* (2–10D), with no heavy deps, deterministic, self-tested; for that regime coupling RealNVP/spline is adequate. **Genuine gaps:** (1) **no periodic/circular handling** — `spline_flow.py:96-104` uses linear tails, so any *angular* CV is mishandled (bgflow's `is_circular` spline ties boundary slopes); (2) **no energy/Boltzmann validation** — `decode_ensemble` returns coordinates with no clash/energy check (a reconstruction can satisfy the bound yet be sterically broken); (3) no internal-coordinate (bond/angle/torsion) representation, so reconstructed Cartesians needn't respect bond geometry; (4) no equivariant/CNF flows (correctly a non-goal for CV-space). bgmol offers OpenMM systems, Z-matrix factories, reference datasets GLIDE hand-rolls or lacks.
- **vs MDZip** — MDZip is a **2D-conv autoencoder trained on pure coordinate RMSD** (`mdzip/AE.py:12-17`), storing one ~20-D latent for the *whole frame* → uint8 → LZMA. On **storage ratio + all-atom RMSD over all frames, MDZip very likely beats GLIDE** (GLIDE only reconstructs coordinates for the ~10% *kept* frames). GLIDE's "a lossy AE destroys kinetics" is a *plausible, correctly-diagnosed hypothesis* (MDZip has no dynamics term) but is **unmeasured in this repo** — and MDZip is a *coordinate*-RMSD method, not an "ensemble-preserving" one, so call it that.
- **vs SZ3/ZFP** — SZ3 ships an **MD-specific spatio-temporal predictor** (`SZBioMDDecomposition.hpp`), a **hard per-atom L∞ error bound** with an unpredictable-value escape, Huffman + zstd backends, and C++ throughput; ZFP adds **O(1) random access** (fixed-rate) and GPU/OpenMP codecs. On all-atom fidelity-at-rate and speed they win decisively. They have **no** notion of slow CVs/MSMs/kinetics — which is exactly GLIDE's point, but their *coordinate* machinery is far more sophisticated than GLIDE's flat-bit residual.

## D. Prioritized roadmap ("best of all worlds")

**Tier 1 — correctness & honesty (do first; mostly bounded):**
1. **Route certified kinetics through deeptime's reversible MLE** (B1) — the single highest-value change. Keep `(C+Cᵀ)/2` for entropy coding only; fall back when deeptime absent; label which estimator produced the numbers.
2. **Reconcile IGFS with the ensemble bound** (B2) — reweight on decode, or restate the guarantee.
3. **Honor `kinetic_bound_valid` everywhere** (B3) — done in `pathbound`/`cli`; extend to `benchmark`.
4. **Tighten claims** — MDZip = coordinate-RMSD (not ensemble); state plainly that on rate/RMSD/speed GLIDE is not the winner; the in-repo benchmark is a conceptual contrast (real numbers = the NTL9 figures). *(Done in README this pass.)*

**Tier 2 — physics rigor (high value):**
5. **Report MFPT / committor preservation** (B4) via deeptime — turns the abstract bound into rate language.
6. **PCCA+ metastable coarse-graining** — certify the bound on the few-state network; far more interpretable contrast.
7. **Internal-coordinate / contact featurization as the default kinetic path** (not aligned Cartesian) — removes TICA's spurious rigid-body slow modes (`runner.py:208-217`). The code already *knows* this (`vampnet_cv.py:14-16`).
8. **Energy/clash sanity check on decoded frames** (optional OpenMM hook) — certify reconstructions are physically valid, not just statistically close.

**Tier 3 — compression competitiveness (if pure-ratio matters):**
9. **Entropy-code the residual** instead of charging a flat bit rate (`kinetic_codec.py:682` even calls the flat rate an "upper bound") — the biggest honest ratio left on the table.
10. **Spatio-temporal residual predictor + outlier escape** (borrow SZ3's space-time Lorenzo) for better all-atom fidelity-at-rate.
11. **zstd final pass + vectorized/compiled coders** — the pure-Python per-symbol coders are orders of magnitude slower than the C++ tools.
12. **Periodic/circular spline support** — only if angular CVs are used; otherwise document the non-periodic assumption.

**Net:** the foundation is solid and the novelty (kinetic bound + analysis-native artifact) is real. The work that most improves *physical rigor and honesty* is Tier 1–2: make the certified path actually use the publishable estimator, measure the rate observables (MFPT) the project already promises, and scope the claims to what the code demonstrates.
