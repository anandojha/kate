"""Coverage for the package __init__: the eager pure-numpy API and the lazy
torch/deeptime-backed attribute loader (PEP 562 __getattr__ / __dir__)."""
import pytest

import epc


def test_eager_pure_numpy_api():
    # these are imported eagerly and must not need torch/deeptime
    assert callable(epc.report_kinetic_fidelity)
    assert callable(epc.pinsker)
    assert epc.pathbound is not None
    assert epc.__version__


def test_lazy_attribute_loader():
    pytest.importorskip("torch")
    # accessing these triggers __getattr__ -> lazy submodule import
    assert epc.RealNVP is not None
    assert epc.EPCCodec is not None
    assert epc.KineticCodec is not None
    assert callable(epc.run_epc)
    assert callable(epc.save_artifact)
    assert callable(epc.load_artifact)
    assert "RealNVP" in dir(epc) and "run_epc" in dir(epc)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = epc.no_such_symbol
