"""
baselines.py
============
The external compressors EPC is benchmarked against (the T3 contrast), plus local
"pseudo-baselines" so the contrast harness runs end-to-end ANYWHERE.

External baselines (MDZip / SZ3 / ZFP) build and run in their OWN environments and on
the cluster, where the trypsin-benzamidine data lives (see RELATED_WORK.txt). We do
NOT vendor their source -- we shell out to them as subprocesses. Each wrapper locates
the tool via an env var (or PATH) and raises `BaselineUnavailable` with a clear
message if it is not configured here. The subprocess command structure is scaffolded;
verify each tool's exact CLI/API before a real run (their flags vary by version).

Local pseudo-baselines (run anywhere, no external tools) DEMONSTRATE the contrast:
  * 'shuffle'  : i.i.d. resample of frames -> the ENSEMBLE is preserved EXACTLY while
                 all temporal correlation is destroyed. The extreme "ensemble
                 preserved, kinetics not" -- what an ensemble-only method approaches.
  * 'quantize' : round coordinates to a coarse grid -> a pointwise-bounded round-trip
                 (the SZ/ZFP family idea) that blurs state boundaries -> kinetics drift
                 while the ensemble is ~preserved.

These are stand-ins for the figure mechanics, NOT claims about the real baselines'
numbers -- those come from the actual MDZip/SZ3/ZFP runs on the cluster.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np


class BaselineUnavailable(RuntimeError):
    """Raised when an external baseline tool is not configured in this environment."""


# env var that points at each external tool (binary or repo dir)
_ENV = {"sz3": "EPC_SZ3_BIN", "zfp": "EPC_ZFP_BIN", "mdzip": "EPC_MDZIP_DIR"}
_LOCAL = {"epc", "shuffle", "quantize"}


def available(method: str) -> bool:
    m = method.lower()
    if m in _LOCAL:
        return True
    env = _ENV.get(m)
    if env and os.environ.get(env):
        return True
    return shutil.which(m) is not None


def _require_external(method: str) -> str:
    m = method.lower()
    path = os.environ.get(_ENV.get(m, "")) or shutil.which(m)
    if not path:
        raise BaselineUnavailable(
            f"baseline '{m}' is not available here. The real {m.upper()} runs in its "
            f"own environment on the cluster; set ${_ENV.get(m, '?')} to its binary/dir "
            f"(see RELATED_WORK.txt). This harness scaffolds the subprocess call; use a "
            f"local pseudo-baseline ('shuffle' / 'quantize') to demo the contrast.")
    return path


# --------------------------------------------------------------------------- #
# local pseudo-baselines
# --------------------------------------------------------------------------- #
def pseudo_shuffle(coords: np.ndarray, seed: int = 0) -> np.ndarray:
    """i.i.d. frame resample: identical ensemble, destroyed kinetics."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, coords.shape[0], size=coords.shape[0])
    return np.asarray(coords)[idx]


def pseudo_quantize(coords: np.ndarray, decimals: int = 1) -> np.ndarray:
    """Coordinate-bounded round-trip (mimics SZ/ZFP pointwise error): coarse rounding
    blurs state boundaries -> kinetics drift while the ensemble is ~preserved."""
    return np.round(np.asarray(coords), decimals=decimals)


# --------------------------------------------------------------------------- #
# external baselines (subprocess; cluster-side). Scaffolds -- VERIFY each CLI.
# --------------------------------------------------------------------------- #
def run_sz3(coords: np.ndarray, abs_err: float = 1e-2) -> np.ndarray:
    """SZ3 pointwise error-bounded round-trip on float32 coords (ABS error mode)."""
    binp = _require_external("sz3")
    arr = np.ascontiguousarray(coords, dtype=np.float32)
    n = arr.size
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "in.f32"); comp = raw + ".sz"; dec = os.path.join(d, "out.f32")
        arr.tofile(raw)
        # NB: verify SZ3's current CLI flags before a real run.
        subprocess.run([binp, "-f", "-z", comp, "-i", raw, "-M", "ABS",
                        str(abs_err), "-1", str(n)], check=True)
        subprocess.run([binp, "-f", "-x", dec, "-s", comp, "-1", str(n)], check=True)
        out = np.fromfile(dec, dtype=np.float32).reshape(arr.shape)
    return out.astype(np.float64)


def run_zfp(coords: np.ndarray, abs_err: float = 1e-2) -> np.ndarray:
    """ZFP fixed-accuracy round-trip on float32 coords."""
    binp = _require_external("zfp")
    arr = np.ascontiguousarray(coords, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "in.f32"); comp = os.path.join(d, "in.zfp"); dec = os.path.join(d, "out.f32")
        arr.tofile(raw)
        dims = ["-1", str(arr.size)]
        subprocess.run([binp, "-i", raw, "-z", comp, "-f", *dims, "-a", str(abs_err)], check=True)
        subprocess.run([binp, "-z", comp, "-o", dec, "-f", *dims, "-a", str(abs_err)], check=True)
        out = np.fromfile(dec, dtype=np.float32).reshape(arr.shape)
    return out.astype(np.float64)


def run_mdzip(coords: np.ndarray, top: str = None, **kw) -> np.ndarray:
    """MDZip autoencoder round-trip (its own torch/lightning env on the cluster)."""
    _require_external("mdzip")
    raise BaselineUnavailable(
        "MDZip runs in its own env on the cluster (compress(traj,top,...) / decompress"
        "(...)); wire $EPC_MDZIP_DIR and its python there. Not run locally.")


def reconstruct(method: str, coords: np.ndarray, **kw) -> np.ndarray:
    """Round-trip `coords` (T, N, 3) through a baseline; returns reconstructed coords.
    Local pseudo-baselines run anywhere; external ones require their tool."""
    m = method.lower()
    if m == "shuffle":
        return pseudo_shuffle(coords, seed=kw.get("seed", 0))
    if m == "quantize":
        return pseudo_quantize(coords, decimals=kw.get("decimals", 1))
    if m == "sz3":
        return run_sz3(coords, abs_err=kw.get("abs_err", 1e-2))
    if m == "zfp":
        return run_zfp(coords, abs_err=kw.get("abs_err", 1e-2))
    if m == "mdzip":
        return run_mdzip(coords, **kw)
    raise ValueError(f"unknown baseline method {method!r}")
