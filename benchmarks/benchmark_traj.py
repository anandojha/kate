"""
Unified benchmark driver scoring trajectory compressors by kinetic fidelity.

A lossy coordinate compressor is conventionally judged by the pointwise error
of its round trip, but a molecular dynamics trajectory stores its most
expensive information in the ordering of frames rather than in any single
structure. The slowest implied timescale of a Markov state model, the number
that sets folding and binding rates, can drift by an order of magnitude while
every coordinate stays within a fraction of an Angstrom. This driver
therefore scores every tool on that observable. The original trajectory fixes
one reference ruler: CA pair distances on a capped evenly spaced atom subset,
TICA at the analysis lag, KMeans microstates, and a reversible maximum
likelihood MSM, all fit once on the original data. Every reconstruction is
pushed through the frozen ruler and given a fresh MSM, so the reported
timescale error measures the reconstruction alone. Refitting the
discretization on degraded coordinates would move the state boundaries along
with the degradation and hide it, which is why the ruler is frozen.

The comparison is charged honestly on both axes. bits_per_coord is the coded
payload of each operating point, model weights and side information included,
divided by the number of stored coordinate values. rmsd_A is computed after
optimal per frame superposition, since without it the number measures rigid
body drift rather than structural error. Trajectories are pooled run aware:
independent runs are concatenated for statistics, but no lagged pair,
transition count, or entropy estimate crosses the seam between two runs.

KATE enters twice. kate_stored scores the Markov model carried inside the
artifact, the object KATE actually certifies. kate_roundtrip rebuilds a full
length coordinate trajectory from the artifact and pushes it through the
frozen ruler like every other tool, the symmetric comparison a coordinate
compressor gets. External tools that are absent from this machine are
reported as skipped with the reason, and the run completes on whatever subset
is available.

NAMEDIR holds one directory per independent run (run*-ca), each containing a
single subdirectory of DCD chunks; a NAMEDIR with no run*-ca entries is
treated as one run. Frames are 0.2 ns apart before striding. The output is
NAME_benchmark.csv plus an aligned text table on stdout.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import glob
import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

DT_NS = 0.2            # DESRES fast folding set frame spacing before striding
RULER_ATOMS = 36       # cap on the evenly spaced CA subset of the ruler
RULER_SEP = 3          # minimum sequence separation of a ruler pair
RULER_DIM = 4          # TICA dimension of the ruler
PCA_QBITS = 14         # quantization depth of the PCA mode projections

TOOL_ORDER = ["sz3", "zfp", "fpzip", "sperr", "xtc", "pca",
              "mdc", "ct", "mdzip", "kate"]

ENV_HELP = """environment knobs for the externally built tools:
  KATE_SZ3_BIN         sz3 CLI (also --sz3-bin, else PATH)
  KATE_SPERR_BIN       sperr3d CLI (also --sperr-bin, else PATH)
  KATE_MDCOMPRESS_BIN  mdcompress CLI (also --mdcompress-bin, else PATH)
  KATE_CT_DIR          compressTraj scripts directory (compress.py, decompress.py)
  KATE_CT_PYTHON       python of the compressTraj environment (default: this one)
  KATE_MDZIP_MODEL     MDZip model type (default skipAE)
  KATE_MDZIP_EPOCHS    MDZip training epochs (default 40)"""


def kabsch_align(X, ref):
    """Superpose every frame of X onto ref by the optimal proper rotation.

    The rotation comes from the SVD of the frame covariance with the sign of
    the smallest singular direction fixed by the determinant, so a reflection
    is never applied. Returns the aligned copy of X, centred on ref."""
    X = np.asarray(X, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    refc = ref - ref.mean(axis=0)
    Xc = X - X.mean(axis=1, keepdims=True)
    H = np.einsum("tni,nj->tij", Xc, refc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    U[:, :, -1] *= d[:, None]
    R = np.einsum("tij,tjk->tik", U, Vt)
    return np.einsum("tni,tij->tnj", Xc, R) + ref.mean(axis=0)


def kabsch_rmsd(X, Y):
    """Per frame RMSD between X and Y after optimal superposition.

    Only the singular values of the frame covariance are needed: with the
    reflection corrected trace, the minimum squared deviation is
    |X|^2 + |Y|^2 - 2 (s1 + s2 + sign(det H) s3) per frame. Computing the
    RMSD without superposition instead measures rigid body drift, which every
    aligned or centred reconstruction would be punished for and a raw
    coordinate codec would not."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    H = np.einsum("tni,tnj->tij", Yc, Xc)
    S = np.linalg.svd(H, compute_uv=False)
    d = np.sign(np.linalg.det(H))
    tr = S[:, 0] + S[:, 1] + d * S[:, 2]
    msd = ((Xc ** 2).sum(axis=(1, 2)) + (Yc ** 2).sum(axis=(1, 2)) - 2.0 * tr) / X.shape[1]
    return np.sqrt(np.clip(msd, 0.0, None))


def rmsd_scalar(X, Y):
    return float(np.sqrt((kabsch_rmsd(X, Y) ** 2).mean()))


def markov_entropy_bits(dtrajs, n_states):
    """Order 1 Markov entropy rate of the microstate sequence, bits per symbol.

    Bigram counts are accumulated per run so no seam pair enters the
    estimate. This is the rate at which the dtraj itself is charged in the
    KATE payload accounting."""
    C = np.zeros((n_states, n_states))
    for d in dtrajs:
        d = np.asarray(d, dtype=np.int64)
        np.add.at(C, (d[:-1], d[1:]), 1.0)
    row = C.sum(axis=1)
    P = np.divide(C, np.clip(row[:, None], 1.0, None))
    pi = row / row.sum()
    return float(-np.nansum(pi[:, None] * P * np.log2(np.where(P > 0, P, 1.0))))


class Ruler:
    """The frozen reference discretization every reconstruction is pushed through.

    Fit once on the original trajectory and never refit: CA pair distances on
    an evenly spaced subset of at most RULER_ATOMS atoms with sequence
    separation at least RULER_SEP, TICA at the analysis lag in RULER_DIM
    dimensions, KMeans with a fixed seed, and a reversible maximum likelihood
    MSM on the largest connected count set. Pair distances are invariant to
    rotation and translation, so reconstructions returned in an aligned or
    centred frame are scored identically to raw ones. All estimators take the
    per run list, so no lagged pair crosses a run boundary."""

    def __init__(self, Xruns, lag, nstates, dt_ns):
        from deeptime.clustering import KMeans
        from deeptime.decomposition import TICA
        nA = Xruns[0].shape[1]
        self.lag = int(lag)
        self.nstates = int(nstates)
        self.dt_ns = float(dt_ns)
        self.keep = np.unique(np.linspace(0, nA - 1, min(RULER_ATOMS, nA)).astype(int))
        self.pairs = np.array([(i, j) for i, j in
                               itertools.combinations(range(len(self.keep)), 2)
                               if j - i >= RULER_SEP])
        feats = [self.dists(x) for x in Xruns]
        self.tica = TICA(lagtime=self.lag, dim=RULER_DIM).fit_fetch(feats)
        Y = [self.tica.transform(f) for f in feats]
        self.km = KMeans(n_clusters=self.nstates, fixed_seed=0,
                         progress=None).fit_fetch(np.concatenate(Y))
        self.t1_ref = self.t1([self.km.transform(y).astype(np.int64) for y in Y])

    def dists(self, A):
        S = np.asarray(A, dtype=np.float64)[:, self.keep, :]
        return np.sqrt(((S[:, self.pairs[:, 0], :] -
                         S[:, self.pairs[:, 1], :]) ** 2).sum(-1))

    def t1(self, dtrajs):
        from deeptime.markov import TransitionCountEstimator
        from deeptime.markov.msm import MaximumLikelihoodMSM
        cm = TransitionCountEstimator(lagtime=self.lag, count_mode="sliding") \
            .fit_fetch(dtrajs).submodel_largest()
        return float(MaximumLikelihoodMSM(reversible=True).fit_fetch(cm)
                     .timescales(k=1)[0]) * self.dt_ns

    def score(self, Xhat_runs):
        dtrajs = [self.km.transform(self.tica.transform(self.dists(r)))
                  .astype(np.int64) for r in Xhat_runs]
        t1 = self.t1(dtrajs)
        return t1, abs(t1 - self.t1_ref) / self.t1_ref * 100.0


class Tool:
    """One compressor behind a uniform operating point interface.

    ``run(param)`` returns the reconstructed coordinates, concatenated across
    runs with the shape of the input, together with the payload in bits.
    ``check()`` returns None when the tool can run here and otherwise the
    reason it cannot, so an absent binary or package downgrades the tool to a
    skipped line instead of aborting the sweep."""

    def __init__(self, name, params, run, check=None):
        self.name = name
        self.params = params
        self.run = run
        self.check = check if check is not None else (lambda: None)


def load_runs(namedir, stride):
    """Load the CA trajectory run aware: one array per independent run."""
    import mdtraj.formats as F
    rdirs = sorted(glob.glob(os.path.join(namedir, "run*-ca"))) or [namedir]
    Xruns = []
    for rdir in rdirs:
        subs = sorted(glob.glob(rdir.rstrip("/") + "/*/"))
        if not subs:
            raise SystemExit("no trajectory subdirectory under %s" % rdir)
        dcds = sorted(glob.glob(subs[0] + "*.dcd"))
        if not dcds:
            raise SystemExit("no DCD chunks under %s" % subs[0])
        chunks = []
        for d in dcds:
            with F.DCDTrajectoryFile(d) as h:
                xyz, _, _ = h.read()
            chunks.append(xyz[::stride].astype(np.float32))
        Xruns.append(np.concatenate(chunks))
    return Xruns


def split_runs(Xhat, run_lengths):
    off = np.cumsum([0] + list(run_lengths))
    return [Xhat[off[i]:off[i + 1]] for i in range(len(run_lengths))]


def _find_bin(cli_value, flag, env_var, exe):
    path = cli_value or os.environ.get(env_var) or shutil.which(exe)
    if path and os.path.isfile(path):
        return path, None
    if path:
        return None, "%s not found at %s" % (exe, path)
    return None, "%s binary not found (set %s or %s)" % (exe, flag, env_var)


def build_tools(X, args):
    """Construct the registry over the concatenated coordinates X (Angstrom)."""
    T, nA = X.shape[0], X.shape[1]
    X32 = np.ascontiguousarray(X.astype(np.float32))
    X2 = np.ascontiguousarray(X32.reshape(T, -1))

    sz3_bin, sz3_why = _find_bin(args.sz3_bin, "--sz3-bin", "KATE_SZ3_BIN", "sz3")
    sperr_bin, sperr_why = _find_bin(args.sperr_bin, "--sperr-bin", "KATE_SPERR_BIN", "sperr3d")
    mdc_bin, mdc_why = _find_bin(args.mdcompress_bin, "--mdcompress-bin",
                                 "KATE_MDCOMPRESS_BIN", "mdcompress")

    def sz3(eb):
        with tempfile.TemporaryDirectory() as d:
            raw, comp, dec = d + "/i.f32", d + "/c.sz", d + "/o.f32"
            X32.tofile(raw)
            n = X32.size
            subprocess.run([sz3_bin, "-f", "-z", comp, "-i", raw,
                            "-M", "ABS", str(eb), "-1", str(n)],
                           check=True, capture_output=True)
            subprocess.run([sz3_bin, "-f", "-x", dec, "-s", comp, "-1", str(n)],
                           check=True, capture_output=True)
            back = np.fromfile(dec, dtype=np.float32).reshape(X.shape)
            return back.astype(np.float64), os.path.getsize(comp) * 8

    def zfp(tol):
        import zfpy
        c = zfpy.compress_numpy(X2, tolerance=float(tol))
        return zfpy.decompress_numpy(c).reshape(X.shape).astype(np.float64), len(c) * 8

    def fpz(prec):
        import fpzip
        c = fpzip.compress(X32, precision=int(prec))
        back = np.asarray(fpzip.decompress(c)).reshape(X.shape)
        return back.astype(np.float64), len(c) * 8

    # SPERR runs as a 3D volume with dims fastest first (3, N, T) so the
    # temporal axis supplies real smoothness; --pwe is a maximum pointwise
    # error, directly comparable to the SZ3 ABS ladder.
    def sperr(eps):
        with tempfile.TemporaryDirectory() as d:
            raw, strm, dec = d + "/i.f32", d + "/c.stream", d + "/o.f32"
            X32.tofile(raw)
            subprocess.run([sperr_bin, "-c", "--ftype", "32",
                            "--dims", "3", str(nA), str(T),
                            "--pwe", str(eps), "--bitstream", strm, raw],
                           check=True, capture_output=True)
            subprocess.run([sperr_bin, "-d", "--decomp_f", dec, strm],
                           check=True, capture_output=True)
            back = np.fromfile(dec, dtype=np.float32).reshape(X.shape)
            return back.astype(np.float64), os.path.getsize(strm) * 8

    # XTC quantizes to a fixed 1/prec nm grid. For small CA systems the real
    # .xtc container is overhead dominated, which is not a compression signal,
    # so the rate is the entropy of the quantized per frame centred
    # coordinates: the coordinate information XTC actually codes, a
    # conservative prediction free upper bound on its true coordinate rate.
    def xtc(prec):
        Xnm = X.astype(np.float64) / 10.0
        cen = Xnm - Xnm.mean(axis=1, keepdims=True)
        q = np.round(cen * prec).astype(np.int64)
        back = ((q.astype(np.float64) / prec) + Xnm.mean(axis=1, keepdims=True)) * 10.0
        _, cnt = np.unique(q.ravel(), return_counts=True)
        p = cnt / cnt.sum()
        H = float(-(p * np.log2(p)).sum())
        return back, H * X.size

    # PCA essential dynamics compression: superpose, keep k variance ordered
    # modes, quantize and entropy code the projections. The closest linear
    # subspace cousin of KATE, with variance optimal modes in place of
    # kinetic ones and no bound.
    pca_cache = {}

    def pca(k):
        if not pca_cache:
            A = kabsch_align(X.astype(np.float64), X[0].astype(np.float64)).reshape(T, -1)
            mu = A.mean(axis=0)
            Ac = A - mu
            C = np.cov(Ac, rowvar=False)
            w, V = np.linalg.eigh(C)
            pca_cache.update(mu=mu, Ac=Ac, V=V[:, np.argsort(w)[::-1]])
        mu, Ac, V = pca_cache["mu"], pca_cache["Ac"], pca_cache["V"]
        Vk = V[:, :k]
        P = Ac @ Vk
        lo, hi = P.min(axis=0), P.max(axis=0)
        step = (hi - lo) / (2 ** PCA_QBITS - 1) + 1e-12
        Q = np.round((P - lo) / step).astype(np.int64)
        Prec = Q * step + lo
        ent = 0.0
        for j in range(k):
            c = np.bincount(Q[:, j] - Q[:, j].min())
            pj = c[c > 0] / len(Q)
            ent += float(-(pj * np.log2(pj)).sum())
        bits = ent * T + Vk.size * 32 + mu.size * 32 + k * 2 * 32
        Xh = (Prec @ Vk.T + mu).reshape(T, nA, 3)
        return Xh, bits

    # Shared CA workspace for the tools that need a real trajectory on disk.
    ws = {}

    def workspace():
        if not ws:
            import mdtraj as md
            d = tempfile.mkdtemp(prefix="benchmark_traj_")
            atexit.register(shutil.rmtree, d, ignore_errors=True)
            top = md.Topology()
            chain = top.add_chain()
            for _ in range(nA):
                res = top.add_residue("ALA", chain)
                top.add_atom("CA", md.element.carbon, res)
            tr = md.Trajectory((X.astype(np.float64) / 10.0).astype(np.float32), top)
            tr[0].save_pdb(d + "/ca.pdb")
            tr.save_dcd(d + "/full.dcd")
            with open(d + "/ca.desc", "w") as f:
                f.write("MOL\tprotein/DNA/RNA\t%d\n" % nA)
            ws.update(dir=d, pdb=d + "/ca.pdb", dcd=d + "/full.dcd",
                      desc=d + "/ca.desc", top=top)
        return ws

    def _check_frames(back, what):
        if back.shape[0] != T:
            raise RuntimeError("%s returned %d frames for %d input frames; run "
                               "boundaries cannot be restored" % (what, back.shape[0], T))
        return back

    def mdc(res_fm):
        import mdtraj as md
        w = workspace()
        mdcf, outp = w["dir"] + "/c.mdc", w["dir"] + "/o.dcd"
        subprocess.run([mdc_bin, "compress", "-i", w["dcd"], "-d", w["desc"],
                        "-o", mdcf, "--res", str(res_fm)],
                       check=True, capture_output=True)
        bits = os.path.getsize(mdcf) * 8
        subprocess.run([mdc_bin, "decompress", "-i", mdcf, "-o", outp],
                       check=True, capture_output=True)
        back = md.load_dcd(outp, top=w["top"]).xyz.astype(np.float64) * 10.0
        os.remove(mdcf)
        os.remove(outp)
        return _check_frames(back, "mdcompress"), bits

    ct_dir = os.environ.get("KATE_CT_DIR", "")
    ct_py = os.environ.get("KATE_CT_PYTHON", sys.executable)

    def check_ct():
        if not ct_dir:
            return "compressTraj scripts not configured (set KATE_CT_DIR)"
        if not os.path.isfile(os.path.join(ct_dir, "compress.py")):
            return "compress.py not found under KATE_CT_DIR=%s" % ct_dir
        return None

    def ct(lat):
        import mdtraj as md
        w = workspace()
        out = w["dir"] + "/ct_out"
        os.makedirs(out, exist_ok=True)
        tag = "L%d" % lat
        subprocess.run([ct_py, os.path.join(ct_dir, "compress.py"),
                        "-r", w["pdb"], "-t", w["dcd"], "-p", tag,
                        "-l", str(lat), "-e", "25", "-b", "1024",
                        "--layers", "128,64", "-o", out],
                       check=True, capture_output=True)
        subprocess.run([ct_py, os.path.join(ct_dir, "decompress.py"),
                        "-m", "%s/%s_model.pt" % (out, tag),
                        "-s", "%s/%s_scaler.pkl" % (out, tag),
                        "-r", w["pdb"],
                        "-c", "%s/%s_compressed.pkl" % (out, tag),
                        "-cog", "%s/%s_cog.npy" % (out, tag),
                        "-p", tag, "-o", out],
                       check=True, capture_output=True)
        back = md.load_xtc("%s/%s_decompressed.xtc" % (out, tag),
                           top=w["pdb"]).xyz.astype(np.float64) * 10.0
        bits = (os.path.getsize("%s/%s_compressed.pkl" % (out, tag)) +
                os.path.getsize("%s/%s_model.pt" % (out, tag))) * 8
        return _check_frames(back, "compressTraj"), bits

    def check_mdzip():
        if importlib.util.find_spec("mdzip") is None:
            return "mdzip package not importable in this environment"
        return None

    # MDZip decodes through its decoder directly on the int8 latents because
    # its own .xtc writer is unreliable; the payload is the real compressed
    # artifact, int8 xz model weights plus int8 lzma latents.
    def mdzip(lat):
        import io
        import lzma
        import pickle
        import torch
        from mdzip.mdzip_core import train, compress
        from mdzip.quantize import load_lightae_int8
        mtype = os.environ.get("KATE_MDZIP_MODEL", "skipAE")
        epochs = int(os.environ.get("KATE_MDZIP_EPOCHS", "40"))
        w = workspace()
        work = w["dir"] + "/mdzip"
        os.makedirs(work, exist_ok=True)
        tag = "L%d" % lat
        train(traj=w["dcd"], top=w["pdb"], out=work, fname=tag, epochs=epochs,
              batchSize=1024, lat=lat, model_type=mtype)
        mdir = os.path.join(work, "%s_compressed" % tag)
        mw = os.path.join(mdir, "%s_model_weights.pt.xz" % tag)
        compress(traj=w["dcd"], top=w["pdb"], model=mw, model_type=mtype,
                 out=mdir, fname=tag)
        clat = os.path.join(mdir, "%s_compressed_lat.pt.xz" % tag)
        m = load_lightae_int8(mw, model_type=mtype, device="cpu")
        dec = m.model.decoder
        dec.eval()
        raw = lzma.decompress(open(clat, "rb").read())
        pkg = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        Z = (pkg["q"].float() / 255.0) * pkg["scale"].view(1, -1) + pkg["min"].view(1, -1)
        scaler = pickle.load(open(os.path.join(mdir, "%s_scaler.pkl" % tag), "rb"))
        C = []
        with torch.no_grad():
            for i in range(0, Z.shape[0], 4096):
                fr = dec(Z[i:i + 4096]).cpu().numpy().reshape(-1, nA, 3)
                fr = scaler.inverse_transform(fr.reshape(-1, 3)).reshape(fr.shape)
                C.append(fr.astype(np.float64))
        back = np.concatenate(C) * 10.0
        bits = (os.path.getsize(mw) + os.path.getsize(clat)) * 8
        return _check_frames(back, "MDZip"), bits

    def _pkg_check(pkg, pip_name):
        def check():
            if importlib.util.find_spec(pkg) is None:
                return "%s not importable (pip install %s)" % (pkg, pip_name)
            return None
        return check

    kmax = min(3 * nA - 1, 16)
    pca_modes = [k for k in [1, 2, 3, 4, 6, 8, 10, 12, 16] if k <= kmax]

    return {t.name: t for t in [
        Tool("sz3", [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0], sz3, lambda: sz3_why),
        Tool("zfp", [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0], zfp, _pkg_check("zfpy", "zfpy")),
        Tool("fpzip", [17, 16, 15, 14, 13, 12, 11, 10], fpz, _pkg_check("fpzip", "fpzip")),
        Tool("sperr", [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0], sperr, lambda: sperr_why),
        Tool("xtc", [1000, 100, 30, 10, 5, 3, 2], xtc),
        Tool("pca", pca_modes, pca),
        Tool("mdc", [1000, 30000, 60000, 100000, 200000, 400000, 600000],
             mdc, lambda: mdc_why),
        Tool("ct", [2, 6, 12], ct, check_ct),
        Tool("mdzip", [4, 10, 20], mdzip, check_mdzip),
    ]}


def check_kate():
    try:
        import kate.runner  # noqa: F401
        return None
    except Exception:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo)
        try:
            import kate.runner  # noqa: F401
            return None
        except Exception as e:
            return "kate package not importable (%s)" % str(e)[:80]


def kate_rows(Xruns, ruler, args):
    """Compress with KATE once and score the artifact both ways.

    kate_stored reads the slowest implied timescale of the Markov model the
    artifact carries, at the rate of the kinetic payload alone: coded latents,
    flow weights, the dtraj at its order 1 Markov entropy rate, the KMeans
    centers, and the TICA parameters. kate_roundtrip rebuilds a full length
    trajectory from the artifact, adds the full atom side of the payload, the
    residual and the state means, and is then pushed through the frozen ruler
    and re estimated exactly like every coordinate compressor. The stored
    model is what KATE certifies; the round trip is the symmetric comparison."""
    from kate.runner import compress_trajectory, reconstruct_full_length
    X = np.concatenate(Xruns)
    T = X.shape[0]
    art, rep = compress_trajectory(
        [x.astype(np.float64) / 10.0 for x in Xruns], cv="tica",
        features="contacts", feature_atoms=RULER_ATOMS, feature_sep=RULER_SEP,
        cv_dim=RULER_DIM, keep_frac=0.05, epochs=60, nstates=ruler.nstates,
        lag=ruler.lag, stride=args.stride, dt_ps=DT_NS * 1000.0, lat_bits=14,
        n_bits=4, seed=args.seed, verbose=False)

    dtrajs = [np.asarray(d, dtype=np.int64) for d in art.dtraj]
    Hrate = markov_entropy_bits(dtrajs, int(art.n_states))
    ncoord = X.size
    coded = rep["coded_bytes"] * 8
    flow = rep["flow_bytes"] * 8
    centers = int(np.asarray(art.centers).size) * 32
    tica_bits = (ruler.pairs.shape[0] * RULER_DIM + RULER_DIM) * 32
    rate_kin = (coded + flow + Hrate * (T - 1) + centers + tica_bits) / ncoord
    rate_full = rate_kin + (rep["residual_bits"] + rep["state_mean_bits"]) / ncoord

    t1_stored = float(rep["implied_timescales_ns"][0])
    err_stored = abs(t1_stored - ruler.t1_ref) / ruler.t1_ref * 100.0

    X_nm, runs_nm = reconstruct_full_length(art)
    Xh = np.asarray(X_nm, dtype=np.float64) * 10.0
    t1_rt, err_rt = ruler.score([np.asarray(r, dtype=np.float64) * 10.0 for r in runs_nm])
    rmsd_rt = rmsd_scalar(X, Xh)

    return [("kate_stored", 0.05, rate_kin, float("nan"), t1_stored, err_stored),
            ("kate_roundtrip", 0.05, rate_full, rmsd_rt, t1_rt, err_rt)]


def _fmt(v, spec):
    return "-" if not np.isfinite(v) else spec % v


def print_table(rows):
    print("%-15s %-9s %12s %9s %11s %9s"
          % ("tool", "param", "bits/coord", "rmsd_A", "t1_ns", "err_pct"))
    for name, p, bpc, rmsd, t1, err in rows:
        print("%-15s %-9s %12s %9s %11s %9s"
              % (name, p, _fmt(bpc, "%.3f"), _fmt(rmsd, "%.3f"),
                 _fmt(t1, "%.1f"), _fmt(err, "%.2f")))


def main():
    ap = argparse.ArgumentParser(
        description="Score trajectory compressors by kinetic fidelity on one "
                    "frozen reference discretization.",
        epilog=ENV_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("namedir", help="directory of independent runs (run*-ca)")
    ap.add_argument("stride", type=int, help="frame stride applied per DCD chunk")
    ap.add_argument("lag", type=int, help="MSM lag in strided frames")
    ap.add_argument("nstates", type=int, help="KMeans microstates of the ruler")
    ap.add_argument("name", help="protein name; output is NAME_benchmark.csv")
    ap.add_argument("--tools", default=",".join(TOOL_ORDER),
                    help="comma separated subset of: %s" % ",".join(TOOL_ORDER))
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--sz3-bin", default=None, help="sz3 CLI path")
    ap.add_argument("--sperr-bin", default=None, help="sperr3d CLI path")
    ap.add_argument("--mdcompress-bin", default=None, help="mdcompress CLI path")
    ap.add_argument("--seed", type=int, default=0,
                    help="KATE seed; the ruler KMeans seed stays fixed at 0")
    args = ap.parse_args()

    requested = [t.strip() for t in args.tools.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TOOL_ORDER]
    if unknown:
        ap.error("unknown tools: %s (known: %s)" % (",".join(unknown), ",".join(TOOL_ORDER)))

    Xruns = load_runs(args.namedir, args.stride)
    run_lengths = [x.shape[0] for x in Xruns]
    X = np.concatenate(Xruns)
    dts = DT_NS * args.stride
    ruler = Ruler(Xruns, args.lag, args.nstates, dts)
    print("%s: %d runs T=%d N=%d | dt %.2f ns | lag %d (%.1f ns) | t1_ref=%.1f ns"
          % (args.name, len(Xruns), X.shape[0], X.shape[1], dts,
             args.lag, args.lag * dts, ruler.t1_ref), flush=True)

    registry = build_tools(X, args)
    rows = []
    for name in requested:
        if name == "kate":
            reason = check_kate()
            if reason:
                print("  %-14s SKIPPED: %s" % (name, reason), flush=True)
                continue
            t0 = time.time()
            try:
                for r in kate_rows(Xruns, ruler, args):
                    rows.append(r)
                    print("  %-14s p=%-7s %8.3f b/coord | RMSD %7s A | t1 %9.1f ns"
                          " | err %6.2f%% (%.0fs)"
                          % (r[0], r[1], r[2], _fmt(r[3], "%.3f"), r[4], r[5],
                             time.time() - t0), flush=True)
            except Exception as e:
                rows.append(("kate", 0.05, *([float("nan")] * 4)))
                print("  ! kate failed: %s" % str(e)[:150], flush=True)
            continue
        tool = registry[name]
        reason = tool.check()
        if reason:
            print("  %-14s SKIPPED: %s" % (name, reason), flush=True)
            continue
        for p in tool.params:
            t0 = time.time()
            try:
                Xh, bits = tool.run(p)
                bpc = bits / X.size
                rmsd = rmsd_scalar(X, Xh)
                t1, err = ruler.score(split_runs(Xh, run_lengths))
            except Exception as e:
                bpc = rmsd = t1 = err = float("nan")
                print("  ! %s p=%s failed: %s" % (name, p, str(e)[:150]), flush=True)
            rows.append((name, p, bpc, rmsd, t1, err))
            print("  %-14s p=%-7s %8s b/coord | RMSD %7s A | t1 %9s ns | err %7s%% (%.0fs)"
                  % (name, p, _fmt(bpc, "%.3f"), _fmt(rmsd, "%.3f"),
                     _fmt(t1, "%.1f"), _fmt(err, "%.2f"), time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, args.name + "_benchmark.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "param", "bits_per_coord", "rmsd_A", "t1_ns",
                    "folding_err_pct"])
        for r in rows:
            w.writerow(r)

    print()
    print_table(rows)
    print()
    print("wrote %s (t1_ref=%.1f ns)" % (out_csv, ruler.t1_ref), flush=True)


if __name__ == "__main__":
    main()
