"""Coverage for the baseline wrappers: local pseudo-baselines, availability detection,
the BaselineUnavailable paths, the reconstruct dispatch, and the external SZ3/ZFP
subprocess bodies (with a mocked tool, since the real binaries are cluster-side)."""
import numpy as np
import pytest

import glide.baselines as bl


def _coords(seed=0):
    return np.random.default_rng(seed).standard_normal((30, 4, 3))


def test_pseudo_baselines():
    c = _coords()
    assert bl.pseudo_shuffle(c, seed=0).shape == c.shape
    q = bl.pseudo_quantize(c, decimals=1)
    assert np.allclose(q, np.round(c, 1))


def test_available_local_and_external(monkeypatch):
    assert bl.available("glide") and bl.available("shuffle") and bl.available("quantize")
    monkeypatch.delenv("GLIDE_SZ3_BIN", raising=False)
    monkeypatch.setattr(bl.shutil, "which", lambda x: None)
    assert not bl.available("sz3")
    monkeypatch.setenv("GLIDE_SZ3_BIN", "/some/sz3")
    assert bl.available("sz3")


def test_require_external_raises_when_absent(monkeypatch):
    monkeypatch.delenv("GLIDE_ZFP_BIN", raising=False)
    monkeypatch.setattr(bl.shutil, "which", lambda x: None)
    with pytest.raises(bl.BaselineUnavailable):
        bl._require_external("zfp")


def test_reconstruct_dispatch_and_unknown():
    c = _coords()
    assert bl.reconstruct("shuffle", c, seed=0).shape == c.shape
    assert bl.reconstruct("quantize", c, decimals=2).shape == c.shape
    with pytest.raises(ValueError):
        bl.reconstruct("nope", c)


def test_external_methods_unavailable_dispatch(monkeypatch):
    monkeypatch.setattr(bl.shutil, "which", lambda x: None)
    for env in ("GLIDE_SZ3_BIN", "GLIDE_ZFP_BIN", "GLIDE_MDZIP_DIR"):
        monkeypatch.delenv(env, raising=False)
    for m in ("sz3", "zfp", "mdzip"):
        with pytest.raises(bl.BaselineUnavailable):
            bl.reconstruct(m, _coords())


def _fake_run():
    def run(args, check=False):
        args = list(map(str, args))
        n = int(args[args.index("-1") + 1]) if "-1" in args else 0
        for flag in ("-x", "-o"):                      # decompress -> write float32 out
            if flag in args:
                np.zeros(n, dtype=np.float32).tofile(args[args.index(flag) + 1])
                return
        if "-z" in args:                               # compress -> write (dummy) blob
            open(args[args.index("-z") + 1], "wb").close()
    return run


def test_run_sz3_body_with_mocked_tool(monkeypatch):
    monkeypatch.setattr(bl, "_require_external", lambda m: "/usr/bin/sz3")
    monkeypatch.setattr(bl.subprocess, "run", _fake_run())
    out = bl.run_sz3(_coords(1), abs_err=0.1)
    assert out.shape == (30, 4, 3)


def test_run_zfp_body_with_mocked_tool(monkeypatch):
    monkeypatch.setattr(bl, "_require_external", lambda m: "/usr/bin/zfp")
    monkeypatch.setattr(bl.subprocess, "run", _fake_run())
    out = bl.run_zfp(_coords(2), abs_err=0.1)
    assert out.shape == (30, 4, 3)


def test_run_mdzip_raises(monkeypatch):
    monkeypatch.setattr(bl, "_require_external", lambda m: "/some/mdzip")
    with pytest.raises(bl.BaselineUnavailable):
        bl.run_mdzip(_coords())
