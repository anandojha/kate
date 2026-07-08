"""T5: out-of-core scaling. Streaming TICA (chunked partial_fit, cross-chunk lagged
pairs preserved) must reproduce batch TICA EXACTLY, and the streaming compress path
must match the in-RAM path. A deeptime cross-check confirms we reproduce its streaming
TICA (the idea we reimplemented)."""
import numpy as np
import pytest

from kate.kinetic_codec import TICA, kabsch_align
from _synth import metastable_coords


def _aligned_flat(coords):
    a, _ = kabsch_align(coords, None)
    return a.reshape(len(coords), -1)


def test_streaming_tica_equals_batch_exactly():
    X = _aligned_flat(metastable_coords(n_steps=5000, n_atoms=8, seed=0))
    batch = TICA(lag=10, n_components=3).fit([X])
    stream = TICA(lag=10, n_components=3)
    sz = 700
    for i in range(0, len(X), sz):
        stream.partial_fit(X[i:i + sz], run_start=(i == 0))
    stream.finalize()
    # same slow timescales to float precision
    assert np.allclose(batch.timescales_, stream.timescales_, atol=1e-6)
    # CVs match up to a per-eigenvector sign
    cb, cs = batch.transform(X), stream.transform(X)
    for k in range(3):
        assert (np.allclose(cb[:, k], cs[:, k], atol=1e-6)
                or np.allclose(cb[:, k], -cs[:, k], atol=1e-6))


def test_streaming_tica_matches_deeptime():
    pytest.importorskip("deeptime")
    from kate import kinetics_deeptime as kd
    X = _aligned_flat(metastable_coords(n_steps=4000, n_atoms=8, seed=1))
    ours = TICA(lag=10, n_components=1).fit([X])
    cv_ours = ours.transform(X)[:, 0]
    _, cvs = kd.tica_cvs([X], lag=10, dim=1)         # deeptime streaming TICA
    corr = abs(np.corrcoef(cv_ours, cvs[0][:, 0])[0, 1])
    assert corr > 0.9                                 # same slow mode (up to sign/scale)


def test_compress_streaming_matches_batch():
    """Streaming reproduces the in-RAM path on every DETERMINISTIC quantity: the
    TICA, the (seeded k-means) discretization, the MSM, and the kinetics are identical.
    The flow/IGFS EXEMPLAR selection (kept_idx) is NOT asserted equal -- it varies even
    between two batch runs due to torch CPU multi-threading non-determinism in the flow
    training; the kinetics, ensemble, and bound are unaffected."""
    pytest.importorskip("torch")
    from kate.runner import compress_trajectory, compress_streaming
    coords = metastable_coords(n_steps=3000, n_atoms=8, seed=0)
    kw = dict(cv_dim=2, keep_frac=0.1, epochs=40, nstates=30, lag=10, seed=0,
              verbose=False)
    bart, brep = compress_trajectory([coords], **kw)

    def factory():
        return (coords[i:i + 700] for i in range(0, len(coords), 700))
    sart, srep = compress_streaming(factory, **kw)

    # streaming TICA == batch TICA (exactly, up to the canonical sign)
    assert np.allclose(bart.tica_timescales, sart.tica_timescales, atol=1e-5)
    # CVs are bit-identical -> the seeded discretization is identical
    assert np.array_equal(bart.dtraj[0], sart.dtraj[0])
    assert np.allclose(bart.centers, sart.centers)
    # the retained MSM and its kinetics are identical
    assert np.allclose(bart.T_msm, sart.T_msm)
    assert np.allclose(brep["implied_timescales_ns"], srep["implied_timescales_ns"], atol=1e-6)
    # full-atom reconstruction quality is comparable (kept frames may differ)
    assert sart.n_keep == bart.n_keep
    assert srep["fullatom_rmsd"] < 5 * brep["fullatom_rmsd"] + 1e-3
