"""
KATE, the Kinetic-Aware Trajectory Encoder: kinetics-preserving compression of
molecular dynamics trajectories with a bound on path-distribution fidelity.

Preserving the stationary ensemble does not preserve kinetics, since two ensembles
with identical stationary distributions can still carry different transition rates.
KATE instead bounds the divergence between the path distributions of the original
and reconstructed trajectories, which splits into an ensemble term and a transition
term,

    KL(path) = KL(ensemble) + KL(transition),

so transition rates and other kinetic observables are covered, not only static
ensemble averages.

Importing kate, including from kate import pathbound, pulls in neither torch nor
deeptime, so the pure-numpy code that evaluates the kinetic bound runs on a host
with neither installed. The torch-backed components (flow, codec, runner,
spline_flow, temporal_prior) and the deeptime-backed components (kinetics_deeptime,
vampnet_cv) are imported lazily, on first attribute access or when the matching CLI
subcommand runs. tests/test_no_eager_torch.py enforces this.
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

# Resolved lazily so that importing kate never imports torch or deeptime; each
# entry maps name -> (submodule, attribute).
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
