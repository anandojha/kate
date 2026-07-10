"""
Baseline compressors for the kinetic-fidelity contrast.

KATE is benchmarked (the T3 contrast) against external MD-trajectory compressors
and against two local pseudo-baselines that make the harness runnable end-to-end
in any environment. The contrast asks whether a method preserves the kinetics,
the state-to-state transition times that set the rates, or only the static
ensemble, since a compressor can reproduce the equilibrium distribution of
coordinates while scrambling the dynamics that ride on top of it.

The external baselines MDZip, SZ3 and ZFP build and run in their own environments
on the cluster, alongside the trypsin-benzamidine data. Their source is not
vendored, so each is invoked as a subprocess. A wrapper locates the tool through
an environment variable (or PATH) and raises BaselineUnavailable when the tool is
not configured. The subprocess flags vary by tool version and are verified
against the installed CLI before a production run.

The two local pseudo-baselines bracket the contrast. 'shuffle' resamples frames
independently, preserving the ensemble exactly while destroying all temporal
correlation, the limiting case that an ensemble-only method approaches.
'quantize' rounds coordinates onto a coarse grid, the pointwise-bounded
round-trip of the SZ/ZFP family, which blurs state boundaries so the kinetics
drift while the ensemble stays approximately preserved. These stand in for the
figure mechanics and are not claims about the real baselines' numbers, which come
from the MDZip/SZ3/ZFP runs on the cluster.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np


class BaselineUnavailable(RuntimeError):
    """Raised when an external baseline tool is not configured in this environment."""


# Environment variable pointing at each external tool (binary or repository dir).
_ENV = {"sz3": "KATE_SZ3_BIN", "zfp": "KATE_ZFP_BIN", "mdzip": "KATE_MDZIP_DIR"}
_LOCAL = {"kate", "shuffle", "quantize"}


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
            f"own environment on the cluster; set ${_ENV.get(m, '?')} to its binary/dir. "
            f"This harness scaffolds the subprocess call; use a "
            f"local pseudo-baseline ('shuffle' / 'quantize') to demo the contrast.")
    return path


def pseudo_shuffle(coords: np.ndarray, seed: int = 0) -> np.ndarray:
    """Resample frames independently, preserving the ensemble but destroying kinetics."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, coords.shape[0], size=coords.shape[0])
    return np.asarray(coords)[idx]


def pseudo_quantize(coords: np.ndarray, decimals: int = 1) -> np.ndarray:
    """Apply a coordinate-bounded round-trip mimicking SZ/ZFP pointwise error.

    Coarse rounding blurs state boundaries so the kinetics drift while the
    ensemble is approximately preserved.
    """
    return np.round(np.asarray(coords), decimals=decimals)


# External baselines run as subprocesses on the cluster; verify each tool's CLI before use.
def run_sz3(coords: np.ndarray, abs_err: float = 1e-2) -> np.ndarray:
    """Run an SZ3 pointwise error-bounded round-trip on float32 coords (ABS mode)."""
    binp = _require_external("sz3")
    arr = np.ascontiguousarray(coords, dtype=np.float32)
    n = arr.size
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "in.f32"); comp = raw + ".sz"; dec = os.path.join(d, "out.f32")
        arr.tofile(raw)
        # Verify SZ3's current command-line flags before a production run.
        subprocess.run([binp, "-f", "-z", comp, "-i", raw, "-M", "ABS",
                        str(abs_err), "-1", str(n)], check=True)
        subprocess.run([binp, "-f", "-x", dec, "-s", comp, "-1", str(n)], check=True)
        out = np.fromfile(dec, dtype=np.float32).reshape(arr.shape)
    return out.astype(np.float64)


def run_zfp(coords: np.ndarray, abs_err: float = 1e-2) -> np.ndarray:
    """Run a ZFP fixed-accuracy round-trip on float32 coords."""
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
    """Run an MDZip autoencoder round-trip in its own torch/lightning env on the cluster."""
    _require_external("mdzip")
    raise BaselineUnavailable(
        "MDZip runs in its own env on the cluster (compress(traj,top,...) / decompress"
        "(...)); wire $KATE_MDZIP_DIR and its python there. Not run locally.")


def reconstruct(method: str, coords: np.ndarray, **kw) -> np.ndarray:
    """Round-trip coords (T, N, 3) through a baseline and return the result.

    Local pseudo-baselines run in any environment; external baselines require
    their configured tool.
    """
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
