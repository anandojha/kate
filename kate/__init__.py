"""
Kinetic-Aware Trajectory Encoder
========================================================
Background
----------
KATE provides kinetics-preserving compression of molecular dynamics trajectories
together with a kinetic, path-distribution fidelity bound. Ensemble-preserving
compression does not in general preserve kinetics: two ensembles with identical
stationary distributions may exhibit different transition rates. KATE addresses
this by bounding the path-distribution divergence, which decomposes as

    KL(path) = ensemble term + transition term,

so that kinetic observables, and not only static ensemble averages, are covered.
The kinetic bound is the central contribution of the package.

Import hygiene
--------------
Importing ``kate``, including ``from kate import pathbound``, pulls in neither
torch nor deeptime. The pure-numpy path that evaluates the kinetic ``bound``
therefore runs on a host with neither dependency installed. The torch-backed
components (flow, codec, runner, spline_flow, temporal_prior) and the
deeptime-backed components (kinetics_deeptime, vampnet_cv) are imported lazily,
on first access or when the corresponding CLI subcommand executes. This property
is enforced by tests/test_no_eager_torch.py.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Eager, pure-numpy public API for the kinetic bound; no torch, no deeptime.
from . import pathbound  # noqa: E402  (numpy only)
from .pathbound import (  # noqa: E402
    report_kinetic_fidelity,
    two_slice_kl,
    path_kl,
    ensemble_kl,
    transition_kl_rate,
    pinsker,
    stationary_distribution,
)

# The following attributes are resolved lazily so that importing ``kate`` never
# imports torch or deeptime. Each entry maps name -> (submodule, attribute).
_LAZY = {
    "KateCodec": ("codec", "KateCodec"),
    "KateArtifact": ("codec", "KateArtifact"),
    "igfs_select": ("codec", "igfs_select"),
    "RealNVP": ("flow", "RealNVP"),
    "KineticCodec": ("kinetic_codec", "KineticCodec"),
    "CompressedTrajectory": ("kinetic_codec", "CompressedTrajectory"),
    "run_kate": ("runner", "run_kate"),
    "save_artifact": ("artifact", "save_artifact"),
    "load_artifact": ("artifact", "load_artifact"),
}

__all__ = [
    "__version__",
    "pathbound",
    "report_kinetic_fidelity", "two_slice_kl", "path_kl", "ensemble_kl",
    "transition_kl_rate", "pinsker", "stationary_distribution",
    *list(_LAZY.keys()),
]


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    if name in _LAZY:
        import importlib
        modname, attr = _LAZY[name]
        mod = importlib.import_module(f"{__name__}.{modname}")
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY.keys())))
