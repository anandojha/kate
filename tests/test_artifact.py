"""Artifact save/load round-trip. The no-flow path must be pure numpy (loadable
without torch); the with-flow path reconstructs the live decoder."""
import numpy as np
import pytest

from epc.artifact import save_artifact, load_artifact
from _synth import toy_artifact


def test_roundtrip_without_flow_preserves_kinetics(tmp_path):
    art = toy_artifact(a=0.03, seed=1)
    p = str(tmp_path / "q.epc")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=False)
    assert loaded.flow_state is None
    assert np.array_equal(loaded.counts, art.counts)
    assert np.allclose(loaded.T_msm, art.T_msm)
    assert loaded.run_lengths == art.run_lengths
    assert len(loaded.dtraj) == 1 and np.array_equal(loaded.dtraj[0], art.dtraj[0])
    assert (loaded.n_states, loaded.lag) == (2, 1)
    assert (loaded.cv, loaded.flow_kind, loaded.entropy) == ("tica", "realnvp", "gaussian")


def test_with_flow_roundtrip_reconstructs_decoder(tmp_path):
    torch = pytest.importorskip("torch")
    from epc.flow import RealNVP
    flow = RealNVP(3, hidden=16, n_layers=4)
    art = toy_artifact()
    art.cv_dim = 3
    art.flow_arch = {"dim": 3, "hidden": 16, "n_layers": 4}
    art.flow_state = {k: v.detach().cpu() for k, v in flow.state_dict().items()}
    p = str(tmp_path / "b.epc")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=True)
    flow2 = loaded.build_flow()
    x = torch.randn(8, 3)
    z1, _ = flow.forward(x)
    z2, _ = flow2.forward(x)
    assert torch.allclose(z1, z2, atol=1e-6)
