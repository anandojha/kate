"""
glide -- Generative Latent Invertible Dynamics-preserving Encoder (GLIDE):
kinetics-preserving compression of MD trajectories, with a kinetic
(path-distribution) fidelity bound.

Thesis: ensemble-preserving compression does NOT preserve kinetics. Two ensembles
with identical stationary distributions can have different rates. GLIDE adds a
path-distribution bound -- KL(path) = ensemble term + transition term -- so KINETIC
observables are covered. The kinetic bound is the headline, not the architecture.

Import hygiene (deliberate): importing ``glide`` -- or ``from glide import pathbound``
-- pulls in NEITHER torch NOR deeptime, so the pure-numpy path (the kinetic
``bound``) runs on a box without either installed. torch-backed pieces (flow,
codec, runner, spline_flow, temporal_prior) and deeptime-backed pieces
(kinetics_deeptime, vampnet_cv) are imported lazily -- only when first accessed, or
when the CLI subcommand that needs them runs. This is enforced by
tests/test_no_eager_torch.py.
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- eager, pure-numpy public API (the kinetic bound; no torch, no deeptime) ---
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

# Everything below is served lazily so that importing ``glide`` never drags in torch
# or deeptime. name -> (submodule, attribute).
_LAZY = {
    "GlideCodec": ("codec", "GlideCodec"),
    "GlideArtifact": ("codec", "GlideArtifact"),
    "igfs_select": ("codec", "igfs_select"),
    "RealNVP": ("flow", "RealNVP"),
    "KineticCodec": ("kinetic_codec", "KineticCodec"),
    "CompressedTrajectory": ("kinetic_codec", "CompressedTrajectory"),
    "run_glide": ("runner", "run_glide"),
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
