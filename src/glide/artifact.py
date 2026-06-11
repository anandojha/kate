"""
On-Disk Format for a GLIDE Artifact
===================================
Background
----------
This module defines the persistent format of a GLIDE artifact, the compressed
object that also serves as the substrate for downstream analysis. The schema is
method-tag aware (cv / flow / entropy) so that the T6-T8 machine-learning variants
can be incorporated without revision, and it is loadable without torch so that
``glide bound`` (pure numpy) runs on a host with neither torch nor deeptime.

On-disk layout
--------------
An artifact is a directory ``NAME.glide/`` containing three files:

  config.json   scalars, method tags, flow architecture, and time metadata
  arrays.npz    coded latents, kept indices, k-means centers, counts, the
                retained MSM, the run-aware all-frame dtraj (integer labels),
                and TICA parameters
  flow.pt       torch state_dict of the flow, written only when a flow is present

Rationale for storing the discrete trajectory
----------------------------------------------
The artifact stores the run-aware all-frame dtraj together with the k-means
centers rather than a single count matrix. The subcommands ``analyze --lag-scan``,
``analyze --bayes``, and ``benchmark`` re-estimate the MSM at many lag times,
whereas a single count matrix supports only one. The per-frame dtraj and centers
are compact and permit re-estimation at any lag without the original trajectory.
The call ``load_artifact(path, with_flow=False)`` reads neither torch nor flow.pt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


@dataclass
class Artifact:
    # Geometry and coder grid (required).
    cv_dim: int
    L: int
    zmax: float
    n_keep: int
    coded_latents: bytes               # entropy-coded kept latents
    kept_idx: np.ndarray               # indices of retained (IGFS) frames
    # Kinetics: the dynamics term of the artifact.
    run_lengths: List[int]
    dtraj: List[np.ndarray]            # run-aware per-frame integer microstate labels
    centers: np.ndarray                # k-means centers in CV space
    counts: np.ndarray                 # run-aware count matrix at `lag`
    T_msm: np.ndarray                  # retained reversible MSM (full n_states)
    n_states: int
    lag: int                           # MSM/TICA lag in strided frames
    # Time metadata.
    stride: int
    dt_ps: float
    dt_strided_ns: float
    # Flow architecture, retained so the decoder can be rebuilt.
    flow_arch: dict                    # {"dim", "hidden", "n_layers"}
    # Method tags; extensible for the T6-T8 variants.
    cv: str = "tica"                   # 'tica' | 'vampnet'
    flow_kind: str = "realnvp"         # 'realnvp' | 'spline'
    entropy: str = "gaussian"          # 'gaussian' | 'temporal'
    msm_estimator: str = "symmetrized-cc"   # estimator backing the reported kinetics
                                            # ('deeptime-mle' | 'symmetrized-cc')
    # Featurizer parameters of the CV transform; compact, retained for T4/recon.
    tica_mean: Optional[np.ndarray] = None
    tica_eigvecs: Optional[np.ndarray] = None
    tica_timescales: Optional[np.ndarray] = None
    align_ref: Optional[np.ndarray] = None     # (N,3) alignment reference (full-atom recon)
    x_mean: Optional[np.ndarray] = None        # (3N,) mean config in aligned space
    # T4 residual stage; None until added by T4.
    residual: Optional[dict] = None
    # T8 temporal and T9 predictive learned-entropy models; None unless used.
    temporal_arch: Optional[dict] = None
    predictive_arch: Optional[dict] = None
    # Flow and entropy-model weights (torch state_dicts); None when with_flow=False.
    flow_state: Optional[dict] = field(default=None, repr=False)
    temporal_state: Optional[dict] = field(default=None, repr=False)
    predictive_state: Optional[dict] = field(default=None, repr=False)

    def build_temporal(self):
        """Reconstruct the TemporalPrior (T8) from its architecture and state.

        Imports torch on the compress/decompress path only.
        """
        if self.temporal_state is None or self.temporal_arch is None:
            return None
        from .temporal_prior import TemporalPrior
        m = TemporalPrior(**self.temporal_arch)
        m.load_state_dict(self.temporal_state)
        m.eval()
        return m

    def build_predictor(self):
        """Reconstruct the T9 predictor (GRU or TCN) from its architecture and state."""
        if self.predictive_state is None or self.predictive_arch is None:
            return None
        from .predictive_coder import make_predictor
        arch = dict(self.predictive_arch)
        m = make_predictor(arch["dim"], kind=arch.get("kind", "gru"),
                           hidden=arch.get("hidden", 64))
        m.load_state_dict(self.predictive_state)
        m.eval()
        return m

    def build_flow(self):
        """Reconstruct the RealNVP or spline flow from its architecture and state.

        Imports torch lazily. This method is intended for the compress/decompress
        path and is not used by the ``bound`` subcommand.
        """
        if self.flow_state is None:
            raise ValueError("artifact has no flow weights (loaded with_flow=False?)")
        import torch
        if self.flow_kind == "realnvp":
            from .flow import RealNVP
            flow = RealNVP(self.flow_arch["dim"], hidden=self.flow_arch["hidden"],
                           n_layers=self.flow_arch["n_layers"])
        elif self.flow_kind == "spline":                         # T7 variant
            from .spline_flow import SplineFlow
            flow = SplineFlow(self.flow_arch["dim"], hidden=self.flow_arch["hidden"],
                              n_layers=self.flow_arch["n_layers"],
                              n_bins=self.flow_arch.get("n_bins", 8))
        else:
            raise ValueError(f"unknown flow_kind {self.flow_kind!r}")
        flow.load_state_dict(self.flow_state)
        flow.eval()
        return flow


# config.json holds only JSON-serializable scalars, lists, and strings.
_CONFIG_KEYS = ("cv_dim", "L", "zmax", "n_keep", "run_lengths", "n_states", "lag",
                "stride", "dt_ps", "dt_strided_ns", "flow_arch", "cv", "flow_kind",
                "entropy", "msm_estimator", "temporal_arch", "predictive_arch")


def save_artifact(art: Artifact, path: str) -> str:
    """Write the artifact to the directory ``path``, creating it if absent.

    Returns
    -------
    str
        The path that was written.
    """
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
        # The residual (T4) is a small dict of arrays and scalars; it is flattened
        # under a prefix. Quantized levels are stored in the smallest safe integer
        # dtype: 16-bit covers the dithered-quantizer range for typical n_bits,
        # giving an eightfold reduction relative to int64 before compression.
        for k, v in art.residual.items():
            a = np.asarray(v)
            if k == "q" and np.issubdtype(a.dtype, np.integer):
                a = a.astype(np.int32 if np.abs(a).max(initial=0) > 32000 else np.int16)
            arrays[f"residual__{k}"] = a
    # The npz is zlib-compressed: the residual levels are small integers that
    # compress well, bringing the on-disk artifact close to the bit-accounting size.
    np.savez_compressed(os.path.join(path, "arrays.npz"), **arrays)

    if art.flow_state is not None:
        import torch
        torch.save(art.flow_state, os.path.join(path, "flow.pt"))
    if art.temporal_state is not None:
        import torch
        torch.save(art.temporal_state, os.path.join(path, "temporal.pt"))
    if art.predictive_state is not None:
        import torch
        torch.save(art.predictive_state, os.path.join(path, "predictive.pt"))
    return path


def load_artifact(path: str, with_flow: bool = True) -> Artifact:
    """Load an artifact from disk.

    When ``with_flow=False`` the loader reads neither torch nor flow.pt, which the
    ``bound`` subcommand relies on so that the kinetic bound runs without torch or
    deeptime installed.
    """
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

    flow_state = temporal_state = None
    if with_flow and os.path.exists(os.path.join(path, "flow.pt")):
        import torch
        flow_state = torch.load(os.path.join(path, "flow.pt"), weights_only=True)
    if with_flow and os.path.exists(os.path.join(path, "temporal.pt")):
        import torch
        temporal_state = torch.load(os.path.join(path, "temporal.pt"), weights_only=True)
    predictive_state = None
    if with_flow and os.path.exists(os.path.join(path, "predictive.pt")):
        import torch
        predictive_state = torch.load(os.path.join(path, "predictive.pt"), weights_only=True)

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
        msm_estimator=cfg.get("msm_estimator", "symmetrized-cc"),
        tica_mean=npz["tica_mean"] if "tica_mean" in npz.files else None,
        tica_eigvecs=npz["tica_eigvecs"] if "tica_eigvecs" in npz.files else None,
        tica_timescales=npz["tica_timescales"] if "tica_timescales" in npz.files else None,
        align_ref=npz["align_ref"] if "align_ref" in npz.files else None,
        x_mean=npz["x_mean"] if "x_mean" in npz.files else None,
        residual=residual,
        temporal_arch=cfg.get("temporal_arch"), temporal_state=temporal_state,
        predictive_arch=cfg.get("predictive_arch"), predictive_state=predictive_state,
        flow_state=flow_state,
    )
