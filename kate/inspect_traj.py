#!/usr/bin/env python
"""
Pre-compression inspection of a DCD trajectory and its topology.

kinetic_codec expects a clean coordinate set and a known sampling interval, so a
trajectory is characterized before it is compressed. Streaming over the file, so
the whole set is never held in memory, the frame count is tallied and the topology
is reduced to its atom, residue, and chain counts and to a residue composition, in
which any residue outside the standard amino acid, water, and ion tables is flagged
as a ligand or ion candidate. A handful of atom selections are then probed so the
stored set can be chosen.

Units are read from the geometry rather than trusted from the header. Among the
heavy atoms the nearest neighbor is a bonded pair, whose separation is ~0.10-0.15
nm (1.0-1.5 Angstrom), so a median nearest-neighbor distance near 0.1 marks
nanometer coordinates while one near 1.5 marks Angstrom. MDTraj reports
Trajectory.xyz in nanometers and .time in picoseconds, and the saved step dt is
taken as the median frame-to-frame gap.

The memory footprint follows from the stored set. Holding the N heavy protein and
ligand atoms costs 3N float32 = 12N bytes per frame, so a run of n_frames frames
costs 12*N*n_frames bytes, and extrapolation to 125 us at a fixed save rate scales
this by 125000 ns / total_ns. The whitening covariance is (3N)^2 entries and its
eigendecomposition is O((3N)^3), so once 3N passes a few thousand a low-rank or
strided fit is needed.

With --compress the optionally strided heavy atoms are handed to kinetic_codec as a
single continuous run, and the compression ratio, the reconstruction RMSD measured
against the per-atom RMSF (the thermal fluctuation scale), and the implied
timescales are reported.
"""
import argparse
import numpy as np

STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
    "MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HIE","HID","HIP","HSD","HSE","HSP","CYX","CYM","ASH","GLH","LYN","ARN",
}
WATER = {"HOH","WAT","SOL","TIP3","TIP","T3P","H2O","TIP4","SPC"}
IONS = {"NA","CL","K","MG","ZN","CA","FE","NA+","CL-","SOD","CLA","POT","MG2",
        "CAL","ZN2"}


def classify(name):
    n = name.upper()
    if n in STANDARD_AA:
        return "protein"
    if n in WATER:
        return "water"
    if n in IONS:
        return "ion"
    return "other"


def heavy_indices(topo, resnames=None):
    """Heavy (non-hydrogen) atom indices, optionally restricted to a set of
    residue names. The filter runs in Python rather than through the topology
    selection language, whose grammar shifts between MDTraj versions.
    """
    out = []
    for a in topo.atoms:
        if a.element is None or a.element.symbol == "H":
            continue
        if resnames is not None and a.residue.name.upper() not in resnames:
            continue
        out.append(a.index)
    return np.array(out, dtype=int)


def inspect(top_path, dcd_path):
    import mdtraj as md
    print("=" * 70)
    print("TOPOLOGY  (%s)" % top_path)
    print("=" * 70)
    topo = md.load_topology(top_path)
    print("  atoms / residues / chains : %d / %d / %d"
          % (topo.n_atoms, topo.n_residues, topo.n_chains))

    by_class = {"protein": [], "water": [], "ion": [], "other": []}
    name_counts = {}
    for r in topo.residues:
        by_class[classify(r.name)].append(r.name)
        name_counts[r.name] = name_counts.get(r.name, 0) + 1
    print("  residues by class         : protein=%d  water=%d  ion=%d  other=%d"
          % tuple(len(by_class[k]) for k in ("protein", "water", "ion", "other")))
    others = sorted(set(by_class["other"]))
    if others:
        print("  NON-STANDARD residues (ligand/cofactor candidates):")
        for nm in others:
            print("      %-6s x%d" % (nm, name_counts[nm]))
    else:
        print("  (no non-standard residues -- ligand may be named as standard?)")

    print("-" * 70)
    print("CANDIDATE SELECTIONS  (atom counts)")
    trials = [("protein", "protein"), ("protein & CA", "protein and name CA"),
              ("backbone", "backbone"), ("water", "water"),
              ("not water", "not water")]
    for nm in others:
        trials.append(("resname %s" % nm, "resname %s" % nm))
    for label, sel in trials:
        try:
            idx = topo.select(sel)
            print("  %-16s : %7d" % (label, len(idx)))
        except Exception as e:
            print("  %-16s : (select failed: %s)" % (label, e))

    # Frame scan; streaming, so the whole file is never held in memory.
    print("-" * 70)
    print("TRAJECTORY  (%s)" % dcd_path)
    n_frames = 0
    first_xyz = None
    times = []
    boxes = []
    for ch in md.iterload(dcd_path, top=top_path, chunk=2000):
        if first_xyz is None:
            first_xyz = np.asarray(ch.xyz[0], dtype=np.float64)
        n_frames += ch.n_frames
        if ch.time is not None:
            times.append(np.asarray(ch.time, dtype=np.float64))
        if ch.unitcell_lengths is not None:
            boxes.append(np.asarray(ch.unitcell_lengths, dtype=np.float64))
    print("  frames                    : %d" % n_frames)

    # Units sanity check: nearest-neighbor distance among heavy atoms in frame 0.
    hv = heavy_indices(topo)
    nn = float("nan")
    if first_xyz is not None and len(hv) > 1:
        from scipy.spatial import cKDTree
        pts = first_xyz[hv]
        d, _ = cKDTree(pts).query(pts, k=2)
        nn = float(np.median(d[:, 1]))
    units = "nm (expected for MDTraj)" if 0.05 < nn < 0.3 else \
            ("Angstrom?? (unexpected)" if 0.5 < nn < 3.0 else "unclear")
    print("  median nearest-neighbor   : %.4f  -> units look like %s" % (nn, units))
    if boxes:
        b0 = boxes[0][0]
        print("  unitcell lengths (frame0) : %s  (0 or absent => solvent stripped)"
              % np.round(b0, 3))

    dt_ns = None
    if times:
        t = np.concatenate(times)
        if t.size > 1:
            d = np.diff(t)
            dt = float(np.median(d))
            uniform = bool(np.allclose(d, dt, atol=1e-6))
            print("  time[:4] (ps)             : %s" % np.round(t[:4], 4))
            if dt <= 0:
                print("  save interval             : non-positive -- header unreliable; "
                      "use your DCDReporter interval")
            elif abs(dt - round(dt)) < 1e-6 and dt < 5:
                print("  save interval             : %.3f ps/frame (looks integer -- "
                      "could be frame index; confirm vs your reporter)" % dt)
            else:
                print("  save interval             : %.4f ps/frame%s"
                      % (dt, "" if uniform else "  (NON-uniform!)"))
                dt_ns = dt / 1000.0

    # Memory footprint; the stored set is the heavy protein and ligand atoms.
    print("-" * 70)
    print("FOOTPRINT")
    prot_lig = set(by_class["protein"]) | set(others)
    stored = heavy_indices(topo, resnames={s.upper() for s in prot_lig})
    Nh = len(stored)
    gb = n_frames * Nh * 3 * 4 / 1e9
    print("  stored set (heavy prot+lig): %d atoms  (3N = %d)" % (Nh, 3 * Nh))
    print("  this file in RAM (float32) : %.2f GB" % gb)
    print("  whitening cov is (3N)^2    : %.2f M entries  (eigh ~O((3N)^3))"
          % ((3 * Nh) ** 2 / 1e6))
    if 3 * Nh > 6000:
        print("  WARNING: 3N large -> use WhiteningTransform(rank=...) (low-rank) "
              "and/or stride for the fit.")
    if dt_ns:
        total_ns = n_frames * dt_ns
        print("  this file duration        : %.3f ns (%d frames @ %.4f ps)"
              % (total_ns, n_frames, dt_ns * 1000))
        if total_ns > 0:
            f125 = n_frames * (125000.0 / total_ns)
            print("  EXTRAPOLATION to 125 us   : ~%.2e frames  -> ~%.1f GB float32"
                  % (f125, f125 * Nh * 3 * 4 / 1e9))
    else:
        print("  (save interval unknown -> cannot extrapolate to 125 us; "
              "supply your reporter interval)")
    print("=" * 70)
    return dict(topo=topo, n_frames=n_frames, stored=stored, dt_ns=dt_ns)


def run_compress(top_path, dcd_path, stride, lag, nbits, nstates, facts):
    """Run kinetic_codec on the strided heavy atoms and report the results.

    The compression ratio, the reconstruction RMSD relative to the per-atom RMSF,
    and the implied timescales are printed. The ``facts`` dictionary is the output
    of inspect().
    """
    import mdtraj as md
    from .kinetic_codec import KineticCodec
    stored = facts["stored"]
    print("\n" + "#" * 70)
    print("COMPRESSION RUN on real data  (stride=%d, lag=%d, n_bits=%d, n_states=%d)"
          % (stride, lag, nbits, nstates))
    print("#" * 70)
    print("  loading %d stored atoms, stride %d ..." % (len(stored), stride))
    chunks = []
    for ch in md.iterload(dcd_path, top=top_path, chunk=2000,
                          atom_indices=stored, stride=stride):
        chunks.append(np.asarray(ch.xyz, dtype=np.float64))
    coords = np.concatenate(chunks, axis=0)
    print("  loaded coords             : %s (nm)" % (coords.shape,))
    runs = [coords]   # One continuous run; split here if this DCD holds segments.

    codec = KineticCodec(tica_lag=lag, tica_dim=2, n_states=nstates,
                         msm_lag=lag, n_bits=nbits, reversible=True, seed=0)
    ct = codec.fit_encode(runs)
    rec = codec.decode(ct)
    rep = codec.report(ct)

    print("  stream total              : %.4f bits/coord  (%.2fx vs float32)"
          % (rep["stream_bits_per_coord"], rep["ratio_vs_float32_stream_only"]))
    print("  state cost / entropy floor: %.4f / %.4f bits/frame"
          % (rep["state_bits_per_frame"], rep["msm_entropy_rate_bits_per_frame"]))
    print("  one-time side info        : %.2f MB" % (rep["side_info_bytes"] / 1e6))

    # Reconstruction RMSD relative to the per-atom RMSF (the thermal scale, in nm).
    orig = coords - coords.mean(1, keepdims=True)
    r = rec[0] - rec[0].mean(1, keepdims=True)
    rmsd = np.sqrt(((orig - r) ** 2).sum(2).mean(1))   # per-frame, pre-aligned
    rmsf = np.sqrt(((coords - coords.mean(0, keepdims=True)) ** 2).sum(2).mean(0))
    print("  mean reconstruction RMSD  : %.4f nm   (mean per-atom RMSF = %.4f nm)"
          % (rmsd.mean(), rmsf.mean()))

    kin = ct.kinetics(k=5)
    its = kin["implied_timescales"]
    print("  implied timescales (frames, x stride x dt = physical):")
    print("     ", np.round(its, 1))
    if facts["dt_ns"]:
        print("     slowest ~ %.2f ns (lag %d frames x stride %d x %.4f ps)"
              % (np.nanmax(its) * stride * facts["dt_ns"], lag, stride,
                 facts["dt_ns"] * 1000))
    print("  NOTE: TICA here is on Cartesian; for real binding kinetics featurize")
    print("        on ligand-pocket contacts and run a lag-convergence scan.")
    print("#" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("top")
    ap.add_argument("dcd")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--nstates", type=int, default=100)
    args = ap.parse_args()
    facts = inspect(args.top, args.dcd)
    if args.compress:
        run_compress(args.top, args.dcd, args.stride, args.lag,
                     args.nbits, args.nstates, facts)
