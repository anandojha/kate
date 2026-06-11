"""Coverage for the package __init__: the eager pure-numpy API and the lazy
torch/deeptime-backed attribute loader (PEP 562 __getattr__ / __dir__)."""
import pytest

import glide


def test_eager_pure_numpy_api():
    # these are imported eagerly and must not need torch/deeptime
    assert callable(glide.report_kinetic_fidelity)
    assert callable(glide.pinsker)
    assert glide.pathbound is not None
    assert glide.__version__


def test_lazy_attribute_loader():
    pytest.importorskip("torch")
    # accessing these triggers __getattr__ -> lazy submodule import
    assert glide.RealNVP is not None
    assert glide.GlideCodec is not None
    assert glide.KineticCodec is not None
    assert callable(glide.run_glide)
    assert callable(glide.save_artifact)
    assert callable(glide.load_artifact)
    assert "RealNVP" in dir(glide) and "run_glide" in dir(glide)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = glide.no_such_symbol
