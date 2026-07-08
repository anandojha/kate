"""Steric-validity check (REVIEW item 8): a reconstruction can satisfy the kinetic bound
yet be sterically broken. The compress report exposes, per-frame, the 1st-percentile
minimum inter-atomic distance for the original vs the reconstruction + an `ok` flag, so
decode-introduced atom overlaps are caught by a force-field-free geometry check."""
import numpy as np
import pytest

pytest.importorskip("torch")

from kate.runner import compress_trajectory
from _synth import metastable_coords


def test_steric_report_present_and_faithful_recon_is_ok():
    coords = metastable_coords(1500, 8, seed=0)
    _, rep = compress_trajectory([coords], cv_dim=2, keep_frac=0.1, epochs=30,
                                 nstates=20, lag=10, n_bits=8, seed=0, verbose=False)
    s = rep["steric"]
    assert s["orig_min_nm"] > 0 and s["rec_min_nm"] > 0
    # a faithful 8-bit reconstruction must not introduce overlaps below the original floor
    assert s["ok"] is True
    assert s["rec_min_nm"] >= 0.9 * s["orig_min_nm"]
