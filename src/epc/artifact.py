"""
artifact.py
===========
On-disk format for an EPC artifact -- the compressed object that IS the analysis
substrate. Designed to be METHOD-TAG aware up front (cv / flow / entropy) so the
T6-T8 ML variants slot in without reworking the schema, and to be loadable WITHOUT
torch so that `epc bound` (pure numpy) runs on a box with neither torch nor deeptime.

Layout (a directory ``NAME.epc/``):
  config.json   scalars + method tags + flow architecture + time metadata
  arrays.npz    coded latents, kept indices, k-means centers, counts, retained MSM,
                the run-aware ALL-FRAME dtraj (integer labels), TICA params
  flow.pt       torch state_dict of the flow (written only if a flow is present)

Why store the dtraj, not just one count matrix (the "file is the kinetic model"):
``analyze --lag-scan`` / ``--bayes`` and ``benchmark`` re-estimate the MSM at MANY
lags; a single count matrix supports only one. The per-frame dtraj + centers is tiny
and lets them re-estimate at ANY lag without the original trajectory.
``load_artifact(path, with_flow=False)`` touches neither torch nor flow.pt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


@dataclass
class Artifact:
    # --- geometry / coder grid (required) ---
    cv_dim: int
    L: int
    zmax: float
    n_keep: int
    coded_latents: bytes               # entropy-coded kept latents
    kept_idx: np.ndarray               # indices of retained (IGFS) frames
    # --- kinetics: the dynamics term ("the file is the kinetic model") ---
    run_lengths: List[int]
    dtraj: List[np.ndarray]            # run-aware per-frame integer microstate labels
    centers: np.ndarray                # k-means centers in CV space
    counts: np.ndarray                 # run-aware count matrix at `lag`
    T_msm: np.ndarray                  # retained reversible MSM (full n_states)
    n_states: int
    lag: int                           # MSM/TICA lag in (strided) frames
    # --- time metadata ---
    stride: int
    dt_ps: float
    dt_strided_ns: float
    # --- flow architecture (so the decoder can be rebuilt) ---
    flow_arch: dict                    # {"dim", "hidden", "n_layers"}
    # --- method tags (extensible for T6-T8) ---
    cv: str = "tica"                   # 'tica' | 'vampnet'
    flow_kind: str = "realnvp"         # 'realnvp' | 'spline'
    entropy: str = "gaussian"          # 'gaussian' | 'temporal'
    # --- featurizer params (CV transform; small, kept for T4/recon) ---
    tica_mean: Optional[np.ndarray] = None
    tica_eigvecs: Optional[np.ndarray] = None
    tica_timescales: Optional[np.ndarray] = None
    align_ref: Optional[np.ndarray] = None     # (N,3) alignment reference (full-atom recon)
    x_mean: Optional[np.ndarray] = None        # (3N,) mean config in aligned space
    # --- T4 residual stage (added by T4; None until then) ---
    residual: Optional[dict] = None
    # --- flow weights (a torch state_dict; None when loaded with_flow=False) ---
    flow_state: Optional[dict] = field(default=None, repr=False)

    # ---- convenience ----
    def build_flow(self):
        """Reconstruct the live RealNVP/spline flow from arch + state. Imports torch
        lazily -- only call this on the compress/decompress path, never for `bound`."""
        if self.flow_state is None:
            raise ValueError("artifact has no flow weights (loaded with_flow=False?)")
        import torch
        if self.flow_kind == "realnvp":
            from .flow import RealNVP
            flow = RealNVP(self.flow_arch["dim"], hidden=self.flow_arch["hidden"],
                           n_layers=self.flow_arch["n_layers"])
        elif self.flow_kind == "spline":                         # T7
            from .spline_flow import SplineFlow
            flow = SplineFlow(self.flow_arch["dim"], hidden=self.flow_arch["hidden"],
                              n_layers=self.flow_arch["n_layers"],
                              n_bins=self.flow_arch.get("n_bins", 8))
        else:
            raise ValueError(f"unknown flow_kind {self.flow_kind!r}")
        flow.load_state_dict(self.flow_state)
        flow.eval()
        return flow


# config.json holds only JSON-friendly scalars/lists/strings.
_CONFIG_KEYS = ("cv_dim", "L", "zmax", "n_keep", "run_lengths", "n_states", "lag",
                "stride", "dt_ps", "dt_strided_ns", "flow_arch", "cv", "flow_kind",
                "entropy")


def save_artifact(art: Artifact, path: str) -> str:
    """Write the artifact to a directory ``path`` (created if absent). Returns path."""
    os.makedirs(path, exist_ok=True)
    cfg = {k: getattr(art, k) for k in _CONFIG_KEYS}
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    arrays = {
        "coded_latents": np.frombuffer(art.coded_latents, dtype=np.uint8),
        "kept_idx": np.asarray(art.kept_idx, dtype=np.int64),
        "centers": np.asarray(art.centers, dtype=np.float64),
        "counts": np.asarray(art.counts, dtype=np.float64),
        "T_msm": np.asarray(art.T_msm, dtype=np.float64),
        "dtraj_concat": np.concatenate([np.asarray(d, dtype=np.int64)
                                        for d in art.dtraj]) if art.dtraj
                        else np.zeros(0, dtype=np.int64),
    }
    for name in ("tica_mean", "tica_eigvecs", "tica_timescales", "align_ref", "x_mean"):
        v = getattr(art, name)
        if v is not None:
            arrays[name] = np.asarray(v, dtype=np.float64)
    if art.residual is not None:
        # residual is a small dict of arrays/scalars (T4); flatten with a prefix.
        for k, v in art.residual.items():
            arrays[f"residual__{k}"] = np.asarray(v)
    np.savez(os.path.join(path, "arrays.npz"), **arrays)

    if art.flow_state is not None:
        import torch
        torch.save(art.flow_state, os.path.join(path, "flow.pt"))
    return path


def load_artifact(path: str, with_flow: bool = True) -> Artifact:
    """Load an artifact. ``with_flow=False`` reads NO torch and NO flow.pt -- used by
    `epc bound` so the kinetic bound runs without torch/deeptime installed."""
    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    npz = np.load(os.path.join(path, "arrays.npz"), allow_pickle=False)

    run_lengths = list(cfg["run_lengths"])
    dconcat = npz["dtraj_concat"]
    dtraj, off = [], 0
    for n in run_lengths:
        dtraj.append(dconcat[off:off + n]); off += n

    residual = None
    res_keys = [k for k in npz.files if k.startswith("residual__")]
    if res_keys:
        residual = {k[len("residual__"):]: npz[k] for k in res_keys}

    flow_state = None
    if with_flow and os.path.exists(os.path.join(path, "flow.pt")):
        import torch
        flow_state = torch.load(os.path.join(path, "flow.pt"), weights_only=True)

    return Artifact(
        cv_dim=int(cfg["cv_dim"]), L=int(cfg["L"]), zmax=float(cfg["zmax"]),
        n_keep=int(cfg["n_keep"]),
        coded_latents=npz["coded_latents"].tobytes(),
        kept_idx=npz["kept_idx"],
        run_lengths=run_lengths, dtraj=dtraj,
        centers=npz["centers"], counts=npz["counts"], T_msm=npz["T_msm"],
        n_states=int(cfg["n_states"]), lag=int(cfg["lag"]),
        stride=int(cfg["stride"]), dt_ps=float(cfg["dt_ps"]),
        dt_strided_ns=float(cfg["dt_strided_ns"]),
        flow_arch=dict(cfg["flow_arch"]),
        cv=cfg.get("cv", "tica"), flow_kind=cfg.get("flow_kind", "realnvp"),
        entropy=cfg.get("entropy", "gaussian"),
        tica_mean=npz["tica_mean"] if "tica_mean" in npz.files else None,
        tica_eigvecs=npz["tica_eigvecs"] if "tica_eigvecs" in npz.files else None,
        tica_timescales=npz["tica_timescales"] if "tica_timescales" in npz.files else None,
        align_ref=npz["align_ref"] if "align_ref" in npz.files else None,
        x_mean=npz["x_mean"] if "x_mean" in npz.files else None,
        residual=residual, flow_state=flow_state,
    )
