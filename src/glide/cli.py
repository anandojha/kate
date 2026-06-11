"""
cli.py -- the single ``glide`` entry point.

  glide compress   TOP DCD -o ART      align -> CV/flow -> IGFS -> entropy code + MSM
  glide decompress ART -o OUT          flow inverse for kept frames (+T4 full-atom)
  glide analyze    ART                 deeptime MSM: timescales, lag scan, Bayesian bars  [T2]
  glide bound      ART REF             ensemble term, transition term, Pinsker pair/path
  glide benchmark  TOP DCD             GLIDE vs MDZip/SZ3/ZFP, scored by the path bound      [T3]

Imports are LAZY PER SUBCOMMAND: the module top level imports only argparse, and each
handler imports only what it needs. So ``glide bound`` (pure numpy) never imports torch
or deeptime -- the kinetic guarantee runs on a box without either.
"""
from __future__ import annotations

import argparse
import os
import sys


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _is_artifact_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def _load_reference_counts(path: str):
    """Reference dynamics P for `bound`: an .glide artifact (use its counts) or an
    .npy/.npz holding a count/transition matrix."""
    import numpy as np
    if _is_artifact_dir(path):
        from .artifact import load_artifact
        return load_artifact(path, with_flow=False).counts
    obj = np.load(path, allow_pickle=False)
    if hasattr(obj, "files"):
        for key in ("counts", "C", "T_msm", "T"):
            if key in obj.files:
                return obj[key]
        return obj[obj.files[0]]
    return obj


# ----------------------------------------------------------------------------- #
# subcommands
# ----------------------------------------------------------------------------- #
def cmd_compress(args):
    from .runner import run_glide, print_report
    from .artifact import save_artifact
    art, report = run_glide(
        args.top, args.dcd, stride=args.stride, cv=args.cv, cv_dim=args.cv_dim,
        keep_frac=args.keep_frac, epochs=args.epochs, nstates=args.nstates,
        lag_ns=args.lag_ns, dt_ps=args.dt_ps, lat_bits=args.lat_bits,
        n_bits=args.n_bits, streaming=args.streaming, chunk=args.chunk,
        entropy=args.entropy, flow_kind=args.flow, predictive_kind=args.predictor)
    print_report(report)
    save_artifact(art, args.out)
    print("  artifact written      : %s" % args.out)


def cmd_decompress(args):
    import numpy as np
    import torch
    from .artifact import load_artifact
    from .codec import decode_iid, gaussian_cumfreq
    from .kinetic_codec import DitheredResidualCodec
    art = load_artifact(args.artifact, with_flow=True)
    flow = art.build_flow()
    cum = gaussian_cumfreq(art.L, art.zmax)
    dlev = decode_iid(art.coded_latents, art.n_keep * art.cv_dim, cum)
    dlev = dlev.reshape(art.n_keep, art.cv_dim)
    zrec = -art.zmax + (dlev + 0.5) * (2 * art.zmax / art.L)
    with torch.no_grad():
        cv = flow.inverse(torch.as_tensor(zrec, dtype=torch.float32)).numpy()

    print("=" * 70)
    print("DECOMPRESS  %s -> %s" % (args.artifact, args.out))
    print("  kept frames           : %d  (CV-space, %d-D)" % (art.n_keep, art.cv_dim))

    if args.full_atom and art.residual is not None:
        # T4: CV reconstructs the SLOW modes via a fitted linear decoder; the residual
        # stage adds back the fast modes -> full 3N coordinates. CV-agnostic (TICA/VAMPnet).
        res = art.residual
        X_approx = (cv - np.asarray(res["cmean"])) @ np.asarray(res["B"]) + np.asarray(res["xmean"])
        rcodec = DitheredResidualCodec(n_bits=int(res["n_bits"]), seed=int(res["seed"]))
        labels_kept = np.asarray(res["labels_kept"]).astype(int)
        resid = (rcodec.dequantize(np.asarray(res["q"]), float(res["step"]))
                 + np.asarray(res["state_mean_R"])[labels_kept])
        N = int(res["n_atoms"])
        out_arr = (X_approx + resid).reshape(art.n_keep, N, 3)
        print("  reconstruction        : FULL-ATOM (3N) via the T4 residual stage")
        print("  shape                 : %s  (n_keep, atoms, xyz)" % (out_arr.shape,))
    else:
        if args.full_atom:
            print("  --full-atom requested but this artifact has NO residual stage;")
            print("  writing CV-space kept frames instead.")
        out_arr = cv
        print("  reconstruction        : CV-space kept frames")

    np.save(args.out, out_arr)
    print("  wrote                 : %s  shape %s" % (args.out, out_arr.shape))
    print("=" * 70)


def cmd_bound(args):
    import numpy as np
    from .artifact import load_artifact
    from .pathbound import report_kinetic_fidelity
    from .kinetic_codec import largest_connected_set, transition_matrix

    Q_art = load_artifact(args.artifact, with_flow=False)     # no torch
    Cq = np.asarray(Q_art.counts, dtype=np.float64)
    Cp = np.asarray(_load_reference_counts(args.ref), dtype=np.float64)
    lag = args.lag if args.lag is not None else Q_art.lag
    L = args.L if args.L is not None else sum(Q_art.run_lengths)

    n = min(Cp.shape[0], Cq.shape[0])
    if Cp.shape[0] != Cq.shape[0]:
        print("  WARNING: reference (%d) and artifact (%d) state counts differ; comparing"
              " the first %d states. For a FAIR comparison discretize both on COMMON"
              " k-means centers (see `glide benchmark`)." % (Cp.shape[0], Cq.shape[0], n))
    Cp, Cq = Cp[:n, :n], Cq[:n, :n]
    act = largest_connected_set(Cp + Cq)        # ergodic under the combined counts
    P, _ = transition_matrix(Cp[np.ix_(act, act)], reversible=True)
    Q, _ = transition_matrix(Cq[np.ix_(act, act)], reversible=True)
    r = report_kinetic_fidelity(P, Q, lag=lag, L=L, k=4)

    dt = Q_art.dt_strided_ns
    print("=" * 72)
    print("KINETIC FIDELITY  (path-distribution bound; lag=%d frames, L=%d)" % (lag, L))
    print("  artifact (Q)          : %s" % args.artifact)
    print("  reference (P)         : %s" % args.ref)
    print("  active states         : %d" % len(act))
    print("-" * 72)
    print("  ensemble term   D(mu_P||mu_Q)     : %.4e nats   (STATIC bound sees only this)"
          % r["ensemble_kl_nats"])
    print("  transition term h(P||Q)           : %.4e nats/step   (the KINETIC signal)"
          % r["transition_kl_rate_nats_per_step"])
    print("  two-slice KL (pairs)              : %.4e nats" % r["two_slice_kl_nats"])
    print("  path KL over %d frames        : %.4e nats" % (L, r.get("path_kl_nats", 0.0)))
    print("-" * 72)
    print("  Pinsker ENSEMBLE bound            : %.4e   (covers STATIC observables only)"
          % r["pinsker_ensemble_bound"])
    print("  Pinsker PAIR bound                : %.4e   (covers KINETIC observables)"
          % r["pinsker_pair_bound"])
    if "pinsker_path_bound" in r:
        print("  Pinsker PATH bound (L frames)     : %.4e" % r["pinsker_path_bound"])
    print("-" * 72)
    print("  implied timescales ref (P)        : %s frames" % np.round(r["its_ref"], 1))
    print("                          (= %s ns)" % np.round(r["its_ref"] * dt, 1))
    print("  implied timescales cmp (Q)        : %s frames" % np.round(r["its_cmp"], 1))
    print("                          (= %s ns)" % np.round(r["its_cmp"] * dt, 1))
    print("  support_ok (path KL finite)       : %s" % r["support_ok"])
    if not r["kinetic_bound_valid"]:
        print("-" * 72)
        print("  *** WARNING: support FAILED -- Q has a transition with zero probability")
        print("      where P does not. The TRUE path divergence is +infinity, so the")
        print("      transition / pair / path numbers above are LOWER BOUNDS only and the")
        print("      kinetic Pinsker bound DOES NOT hold (reported as inf). This means the")
        print("      reconstruction misses a transition the reference has -- kinetics broken.")
    print("=" * 72)
    print("Reading: the STATIC (ensemble) Pinsker bound does NOT cover kinetics; only the")
    print("PAIR / PATH bound (which includes the transition term) does. A large transition")
    print("term with a ~0 ensemble term is exactly 'ensemble preserved, kinetics not'.")
    print("=" * 72)


def cmd_analyze(args):
    """Production kinetics (T2) from the artifact's stored run-aware dtraj: a
    reversible maximum-likelihood MSM, an implied-timescale lag scan (the honest lag
    choice), and Bayesian error bars -- all WITHOUT decoding any coordinates."""
    import numpy as np
    from .artifact import load_artifact
    from . import kinetics_deeptime as kd
    if not kd._HAVE_DEEPTIME:
        raise SystemExit("`glide analyze` needs the kinetics engine: "
                         "pip install glide[kinetics]  (deeptime). Original error: %r"
                         % (kd._IMPORT_ERR,))

    art = load_artifact(args.artifact, with_flow=False)
    dtrajs = [np.asarray(d, dtype=np.int64) for d in art.dtraj]
    dt = art.dt_strided_ns
    base_lag = args.lag if args.lag is not None else art.lag
    k = args.k
    min_run = min(len(d) for d in dtrajs)

    print("=" * 72)
    print("KINETICS  (deeptime reversible-MLE MSM; from the stored dtraj, no decode)")
    print("  artifact              : %s" % args.artifact)
    print("  runs / frames         : %d / %d   (%.4f ns/frame)"
          % (len(dtrajs), sum(len(d) for d in dtrajs), dt))
    print("  microstates           : %d   base lag : %d frames (%.2f ns)"
          % (art.n_states, base_lag, base_lag * dt))

    if args.lag_scan:
        if args.lags:
            lags = [int(x) for x in args.lags.split(",")]
        else:
            lags = sorted({max(1, int(base_lag * f))
                           for f in (0.25, 0.5, 1, 2, 4)})
        lags = [L for L in lags if L < min_run // 2]
        scan = kd.its_lag_scan(dtrajs, lags, k=k)         # NaN rows on non-convergence
        print("-" * 72)
        print("IMPLIED-TIMESCALE LAG SCAN  (pick the lag where these PLATEAU)")
        print("  lag(frames)  lag(ns)   t2..t%d (ns)" % (k + 1))
        for L, ts in zip(lags, scan):
            tag = "  [MLE did not converge]" if np.all(np.isnan(ts)) else ""
            print("  %9d  %7.2f   %s%s" % (L, L * dt, np.round(ts * dt, 1), tag))
        print("  (timescales are a LOWER BOUND; they rise with lag, then plateau --")
        print("   the plateau lag is the honest MSM lag, per Prinz et al. A lag whose")
        print("   reversible MLE did not converge -- too few transitions / disconnected")
        print("   states at that lag -- is flagged; use fewer microstates or more data.)")

    print("-" * 72)
    if args.bayes:
        try:
            mean, std = kd.bayes_timescales(dtrajs, base_lag, k=k, n_samples=args.n_samples)
            print("IMPLIED TIMESCALES with BAYESIAN error bars  (lag %d, %d samples)"
                  % (base_lag, args.n_samples))
            for i in range(len(mean)):
                print("  t%-2d : %8.1f +/- %6.1f ns   (%.1f +/- %.1f frames)"
                      % (i + 2, mean[i] * dt, std[i] * dt, mean[i], std[i]))
        except Exception as e:
            print("BAYESIAN error bars: the MSM did not converge at lag %d (%s)."
                  % (base_lag, type(e).__name__))
            print("  -> the discretization is too fine / poorly connected at this lag;")
            print("     try fewer microstates (--nstates), a different lag, or more data.")
    else:
        try:
            its = kd.implied_timescales(dtrajs, base_lag, k=k)
            print("IMPLIED TIMESCALES  (reversible MLE, lag %d frames)" % base_lag)
            print("  frames : %s" % np.round(its, 1))
            print("  ns     : %s" % np.round(its * dt, 1))
        except Exception as e:
            print("IMPLIED TIMESCALES: the reversible MLE did not converge at lag %d (%s)."
                  % (base_lag, type(e).__name__))
            print("  -> too few transitions / disconnected states; use fewer microstates,")
            print("     a different lag, or more sampling. (Not a tool error -- a data/")
            print("     discretization limit; see the lag scan above.)")
    if args.resolution:
        print("-" * 72)
        print("KINETIC RESOLUTION  (which processes this trajectory can actually validate)")
        try:
            rep = kd.kinetic_resolution(dtrajs, base_lag, dt, k=k,
                                        n_samples=args.n_samples)
            total_us = sum(len(d) for d in dtrajs) * dt / 1000.0
            print(kd.format_resolution(rep, total_us=total_us))
        except Exception as e:
            print("  resolution report unavailable: the Bayesian MSM did not converge "
                  "at lag %d (%s)." % (base_lag, type(e).__name__))
        print("  (A kinetic observable the SOURCE trajectory never sampled cannot be")
        print("   'preserved' by any compressor -- report this BEFORE comparing methods.)")
    print("-" * 72)
    print("Caveats (RECIPE T2): featurize on LIGAND-POCKET CONTACTS (not raw Cartesian)")
    print("for binding kinetics and align on protein only; report k_on/k_off ONLY if the")
    print("binding/unbinding event count supports it -- otherwise lean on the implied")
    print("timescales / MSM eigenvalues and say so.")
    print("=" * 72)


def cmd_benchmark(args):
    """T3 contrast: load the DCD, round-trip it through each method, score each
    method's kinetic fidelity against the original MSM, and plot the contrast."""
    import numpy as np
    import mdtraj as md
    from .benchmark import run_benchmark
    from .inspect_traj import heavy_indices

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    topo = md.load_topology(args.top)
    sel = heavy_indices(topo)
    chunks = [np.asarray(ch.xyz, dtype=np.float64) for ch in
              md.iterload(args.dcd, top=args.top, chunk=2000,
                          atom_indices=sel, stride=args.stride)]
    coords = np.concatenate(chunks, 0)
    dt = args.stride * args.dt_ps / 1000.0
    print("=" * 84)
    print("BENCHMARK  %s  (%d frames, %d heavy atoms, stride %d)"
          % (args.dcd, coords.shape[0], len(sel), args.stride))
    run_benchmark([coords], methods=methods, lag=args.lag, nstates=args.nstates,
                  cv_dim=args.cv_dim, dt_strided_ns=dt, out=args.out)


# ----------------------------------------------------------------------------- #
# parser
# ----------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glide",
        description="GLIDE (Generative Latent Invertible Dynamics-preserving Encoder): "
                    "kinetics-preserving compression of MD trajectories, with a "
                    "kinetic (path-distribution) fidelity bound.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compress", help="TOP DCD -> artifact (flow codec + retained MSM)")
    c.add_argument("top"); c.add_argument("dcd")
    c.add_argument("-o", "--out", required=True, help="output artifact path (NAME.glide)")
    c.add_argument("--stride", type=int, default=10)
    c.add_argument("--cv-dim", type=int, default=6)
    c.add_argument("--keep-frac", type=float, default=0.10)
    c.add_argument("--epochs", type=int, default=300)
    c.add_argument("--nstates", type=int, default=200)
    c.add_argument("--lag-ns", type=float, default=5.0)
    c.add_argument("--dt-ps", type=float, default=100.0)
    c.add_argument("--lat-bits", type=int, default=14)
    c.add_argument("--n-bits", type=int, default=4, help="residual quantizer bit depth (T4)")
    c.add_argument("--streaming", action="store_true", help="out-of-core compress (T5)")
    c.add_argument("--chunk", type=int, default=2000, help="streaming chunk size (frames)")
    # ML opt-ins (defaults = the tested baseline; T6-T8 enable the alternatives)
    c.add_argument("--cv", choices=["tica", "vampnet"], default="tica")
    c.add_argument("--flow", choices=["realnvp", "spline"], default="realnvp")
    c.add_argument("--entropy", choices=["gaussian", "temporal", "predictive"],
                   default="gaussian")
    c.add_argument("--predictor", choices=["gru", "tcn"], default="gru",
                   help="T9 predictive entropy model (default GRU; streaming-compatible)")
    c.set_defaults(func=cmd_compress)

    d = sub.add_parser("decompress", help="artifact -> trajectory (kept frames)")
    d.add_argument("artifact")
    d.add_argument("-o", "--out", required=True, help="output .npy")
    d.add_argument("--full-atom", action="store_true", help="full 3N reconstruction (T4)")
    d.set_defaults(func=cmd_decompress)

    a = sub.add_parser("analyze", help="artifact -> kinetics (deeptime MSM)")
    a.add_argument("artifact")
    a.add_argument("--lag", type=int, default=None, help="MSM lag in frames (default: artifact's)")
    a.add_argument("--lag-scan", action="store_true", help="implied-timescale lag scan")
    a.add_argument("--lags", default=None, help="comma-separated lags for the scan")
    a.add_argument("--bayes", action="store_true", help="Bayesian timescale error bars")
    a.add_argument("--resolution", action="store_true",
                   help="kinetic-resolution report: which processes are statistically "
                        "resolved (Bayesian CI + independent-event count)")
    a.add_argument("--k", type=int, default=4, help="number of slow timescales")
    a.add_argument("--n-samples", type=int, default=100, help="Bayesian MSM samples")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("bound", help="artifact ref -> kinetic-fidelity report")
    b.add_argument("artifact"); b.add_argument("ref")
    b.add_argument("--lag", type=int, default=None, help="override lag (frames)")
    b.add_argument("--L", type=int, default=None, help="trajectory length for the path KL")
    b.set_defaults(func=cmd_bound)

    k = sub.add_parser("benchmark", help="TOP DCD -> contrast table+plot (GLIDE vs baselines)")
    k.add_argument("top"); k.add_argument("dcd")
    k.add_argument("--methods", default="glide,sz3,zfp,mdzip",
                   help="comma list: glide, sz3, zfp, mdzip, shuffle, quantize")
    k.add_argument("--stride", type=int, default=10)
    k.add_argument("--lag", type=int, default=10)
    k.add_argument("--nstates", type=int, default=100)
    k.add_argument("--cv-dim", type=int, default=2)
    k.add_argument("--dt-ps", type=float, default=100.0)
    k.add_argument("--out", default="benchmark")
    k.set_defaults(func=cmd_benchmark)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
