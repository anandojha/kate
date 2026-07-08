"""Coverage for the package __init__: the eager pure-numpy API and the lazy
torch/deeptime-backed attribute loader (PEP 562 __getattr__ / __dir__)."""
import pytest

import kate


def test_eager_pure_numpy_api():
    # these are imported eagerly and must not need torch/deeptime
    assert callable(kate.report_kinetic_fidelity)
    assert callable(kate.pinsker)
    assert kate.pathbound is not None
    assert kate.__version__


def test_lazy_attribute_loader():
    pytest.importorskip("torch")
    # accessing these triggers __getattr__ -> lazy submodule import
    assert kate.RealNVP is not None
    assert kate.KateCodec is not None
    assert kate.KineticCodec is not None
    assert callable(kate.run_kate)
    assert callable(kate.save_artifact)
    assert callable(kate.load_artifact)
    assert "RealNVP" in dir(kate) and "run_kate" in dir(kate)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = kate.no_such_symbol
