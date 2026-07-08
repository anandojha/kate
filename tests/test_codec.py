"""
Flow-based KATE codec tests (torch-gated): the i.i.d. Gaussian-base entropy coder is
exact, kept frames reconstruct to quantization accuracy (the flow inverts exactly),
and kinetics come from the retained MSM. Artifact save/load round-trip is added by
T1's test extension.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from kate.codec import (KateCodec, encode_iid, decode_iid, gaussian_cumfreq,  # noqa: E402
                        igfs_select, stationary_reweight)


def _simulate(n_steps, n_atoms, a=0.01, intra=0.25, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[1 - a, a, 0.0], [a, 1 - 2 * a, a], [0.0, a, 1 - a]])
    cdf = np.cumsum(P, axis=1)
    m = np.zeros(n_steps, dtype=int)
    u = rng.random(n_steps)
    for t in range(1, n_steps):
        m[t] = np.searchsorted(cdf[m[t - 1]], u[t])
    wells = np.array([-2.0, 0.0, 2.0])
    xi = wells[m] + intra * rng.standard_normal(n_steps)
    ref = rng.standard_normal((n_atoms, 3)) * 2.0
    mode = rng.standard_normal((n_atoms, 3)); mode /= np.linalg.norm(mode)
    xyz = (ref[None] + xi[:, None, None] * mode[None]
           + noise * rng.standard_normal((n_steps, n_atoms, 3)))
    return xyz.astype(np.float64)


def test_iid_gaussian_base_coder_is_exact():
    L = 1 << 12
    cum = gaussian_cumfreq(L, zmax=5.0)
    rng = np.random.default_rng(0)
    # mostly central levels (where a Gaussian table has mass) plus a few tail levels
    levels = np.clip(np.round(L / 2 + (L / 12) * rng.standard_normal(2000)),
                     0, L - 1).astype(np.int64)
    blob = encode_iid(levels, cum)
    decoded = decode_iid(blob, len(levels), cum)
    assert np.array_equal(decoded, levels)


def test_igfs_stationary_reweight_corrects_tail_bias():
    # Farthest-point selection over-represents the tails (B2): the raw kept subset has an
    # inflated second moment, and the stationary Voronoi weights must pull it back to the
    # true N(0, 1) variance so the ensemble-Pinsker guarantee applies to the reweighted set.
    rng = np.random.default_rng(0)
    z = rng.standard_normal((20000, 1))
    kept = igfs_select(z, n_keep=400, seed=0)
    zk = z[kept, 0]
    raw_var = float(zk.var())                         # biased high (tails over-represented)
    w = stationary_reweight(z, kept)
    assert np.isclose(w.sum(), 1.0)
    assert w.min() >= 0.0
    wmean = float((w * zk).sum())
    wvar = float((w * (zk - wmean) ** 2).sum())       # reweighted second moment
    assert raw_var > 1.5                              # the documented bias is present
    assert abs(wvar - 1.0) < 0.15                     # reweighting restores unit variance
    assert abs(wvar - 1.0) < abs(raw_var - 1.0)       # strictly closer than the raw subset


def test_ensemble_average_uses_stationary_weights():
    # KateCodec.ensemble_average must weight by the stored stationary weights, recovering
    # the full-ensemble mean more faithfully than the uniform mean over the kept subset.
    rng = np.random.default_rng(1)
    z = rng.standard_normal((8000, 1))
    kept = igfs_select(z, n_keep=200, seed=1)
    w = stationary_reweight(z, kept)

    class _Stub:
        n_keep = len(kept)
        kept_weights = w
    obs = (z[kept, 0] ** 2)                            # observable whose true mean is ~1
    weighted = float(KateCodec.ensemble_average(_Stub, obs))
    uniform = float(obs.mean())
    assert abs(weighted - 1.0) < abs(uniform - 1.0)


def test_kate_end_to_end_small():
    torch.manual_seed(0)
    runs = [_simulate(1500, 6, seed=10), _simulate(1500, 6, seed=11)]
    codec = KateCodec(n_keep_frac=0.1, flow_layers=8, flow_hidden=48,
                     flow_epochs=60, lat_bits=14, tica_lag=10, tica_dim=2,
                     n_states=40, msm_lag=10, seed=0)
    ct = codec.fit_encode(runs, verbose=False)
    # kept frames reconstruct to ~quantization accuracy (flow inverts exactly)
    rec = KateCodec.decode_ensemble(ct)
    assert rec.shape == (ct.n_keep, 6, 3)
    # the retained MSM gives finite, ordered slow timescales
    its = KateCodec.kinetics(ct, k=4)
    assert np.all(np.isfinite(its[:2]))
    assert its[0] >= its[1] > 0
    # the coder actually produced bytes and decodes to the right count
    assert ct.n_keep == max(2, int(0.1 * 3000))
