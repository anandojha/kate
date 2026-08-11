"""Artifact save/load round-trip. The no-flow path must be pure numpy (loadable
without torch); the with-flow path reconstructs the live decoder."""
import numpy as np
import pytest

from kate.artifact import save_artifact, load_artifact
from _synth import toy_artifact


def test_roundtrip_without_flow_preserves_kinetics(tmp_path):
    art = toy_artifact(a=0.03, seed=1)
    p = str(tmp_path / "q.kate")
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
    from kate.flow import RealNVP
    flow = RealNVP(3, hidden=16, n_layers=4)
    art = toy_artifact()
    art.cv_dim = 3
    art.flow_arch = {"dim": 3, "hidden": 16, "n_layers": 4}
    art.flow_state = {k: v.detach().cpu() for k, v in flow.state_dict().items()}
    p = str(tmp_path / "b.kate")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=True)
    flow2 = loaded.build_flow()
    x = torch.randn(8, 3)
    z1, _ = flow.forward(x)
    z2, _ = flow2.forward(x)
    assert torch.allclose(z1, z2, atol=1e-6)


def test_kept_weights_survive_the_round_trip(tmp_path):
    """The selection weights must reach the file. Farthest-point selection
    over-represents the low-density tails, so an artifact without them yields a biased
    ensemble average and the ensemble term of the certificate does not describe the
    stored object. The weights were once computed on the demo path only and never
    written, which this pins down."""
    art = toy_artifact(a=0.03, seed=3)
    art.kept_weights = np.array([0.5, 0.25, 0.25])
    p = str(tmp_path / "w.kate")
    save_artifact(art, p)
    loaded = load_artifact(p, with_flow=False)
    assert loaded.kept_weights is not None
    assert np.allclose(loaded.kept_weights, art.kept_weights)
    assert abs(float(loaded.kept_weights.sum()) - 1.0) < 1e-12


def test_artifact_without_weights_still_loads(tmp_path):
    """Files written before the weights were stored must remain readable."""
    art = toy_artifact(a=0.03, seed=4)
    art.kept_weights = None
    p = str(tmp_path / "old.kate")
    save_artifact(art, p)
    assert load_artifact(p, with_flow=False).kept_weights is None
