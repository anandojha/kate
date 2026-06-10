"""
vampnet_cv.py
=============
T6 -- learned NONLINEAR slow collective variables via VAMPnets (deeptime's PyTorch
VAMPNet). A drop-in alternative to the linear TICA CVs: train a small network lobe to
maximize the VAMP score (the variational principle for Markov dynamics), then run the
flow + MSM on the learned CVs instead of TICA. This is "deep learning of molecular
kinetics" and improves the DYNAMICS term directly.

It keeps the thesis intact: the flow on the CVs is still invertible, the full-atom
residual stage still recovers the fast modes (via a fitted linear CV->coordinate
decoder, CV-agnostic), and the path bound is unchanged. VAMPnets are PRIOR ART (Mardt
et al. 2018) -- cited, not claimed; only the integration + the kinetic bound are ours.

Featurize on LIGAND-POCKET CONTACTS (not raw Cartesian) for binding kinetics (the
caller chooses the features). API verified against deeptime 0.4.5:
  TrajectoryDataset(lagtime, traj) -> DataLoader
  VAMPNet(lobe, device, learning_rate).fit(loader, n_epochs).fetch_model()
  model.transform(traj);  VAMP(lagtime).fit_fetch(cvs).score(r=2)

Import-guarded: deeptime is the optional [kinetics] extra (torch is core).
"""
from __future__ import annotations

import numpy as np

try:
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.decomposition import VAMP
    from deeptime.util.data import TrajectoryDataset
    _HAVE_DEEPTIME = True
    _IMPORT_ERR = None
except Exception as _e:                              # pragma: no cover
    _HAVE_DEEPTIME = False
    _IMPORT_ERR = _e


def _require():
    if not _HAVE_DEEPTIME:
        raise ImportError(
            "vampnet_cv (T6) needs the kinetics engine: pip install epc[kinetics] "
            "(deeptime). Original import error: %r" % (_IMPORT_ERR,))


def _build_lobe(in_dim, out_dim, hidden):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(),
        nn.Linear(hidden, hidden), nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


def vampnet_cvs(runs_feat, lag, dim, *, hidden=32, n_epochs=30, batch=256, lr=5e-3,
                seed=0, device="cpu", verbose=False):
    """Train a VAMPNet on a list of per-run feature arrays and return
    (model, list_of_CV_trajectories, vamp2_score). Run-aware: lagged pairs are formed
    within each run (ConcatDataset of per-run TrajectoryDatasets), never across seams."""
    _require()
    import torch
    from torch.utils.data import DataLoader, ConcatDataset
    torch.manual_seed(int(seed))
    feat = [np.asarray(r, dtype=np.float32) for r in runs_feat]
    in_dim = feat[0].shape[1]
    dsets = [TrajectoryDataset(int(lag), r) for r in feat]
    ds = ConcatDataset(dsets) if len(dsets) > 1 else dsets[0]
    loader = DataLoader(ds, batch_size=int(batch), shuffle=True)
    lobe = _build_lobe(in_dim, int(dim), int(hidden))
    vnet = VAMPNet(lobe=lobe, device=device, learning_rate=float(lr))
    vnet.fit(loader, n_epochs=int(n_epochs), progress=None)
    model = vnet.fetch_model()
    cvs = [np.asarray(model.transform(r)).astype(np.float64) for r in feat]
    score = vamp_score(cvs, lag)
    if verbose:
        print("  VAMPNet CVs           : %d-D   VAMP2 score: %.3f" % (dim, score))
    return model, cvs, score


def vamp_score(cvs, lag, r=2):
    """VAMP-r score of a set of CV trajectories (run-aware list). Higher = the CVs
    capture more of the slow dynamics. Use it to compare VAMPnet vs the TICA baseline."""
    _require()
    cvs = [np.asarray(c, dtype=np.float64) for c in cvs]
    model = VAMP(lagtime=int(lag)).fit_fetch(cvs if len(cvs) > 1 else cvs[0])
    return float(model.score(r=r))
