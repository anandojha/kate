"""
Learned Nonlinear Slow Collective Variables (VAMPnets)
======================================================

Background
----------
This module learns nonlinear slow collective variables (CVs) via VAMPnets, using
deeptime's PyTorch VAMPNet implementation. It provides a drop-in alternative to the
linear TICA CVs: a small network lobe is trained to maximize the VAMP score, the
variational principle for Markov dynamics (VAMPnets: Mardt et al., Nat. Commun. 9, 5
(2018)), after which the flow and MSM are run on the learned CVs rather than on TICA.
This constitutes deep learning of molecular kinetics and improves the dynamics term
directly.

The overall design is preserved: the flow on the CVs remains invertible, the full-atom
residual stage still recovers the fast modes through a fitted, CV-agnostic linear
CV-to-coordinate decoder, and the path bound is unchanged. VAMPnets are prior art and
are cited rather than claimed; only the integration and the kinetic bound are
contributed here.

For binding kinetics, the features are ligand-pocket contacts rather than raw Cartesian
coordinates; the choice of features rests with the caller. The API has been verified
against deeptime 0.4.5:

    TrajectoryDataset(lagtime, traj) -> DataLoader
    VAMPNet(lobe, device, learning_rate).fit(loader, n_epochs).fetch_model()
    model.transform(traj);  VAMP(lagtime).fit_fetch(cvs).score(r=2)

The deeptime import is guarded: deeptime is the optional [kinetics] extra, whereas torch
is a core dependency.
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
            "vampnet_cv (T6) needs the kinetics engine: pip install glide[kinetics] "
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
    """Train a VAMPNet on per-run feature arrays and return the learned CVs.

    Returns the tuple (model, list_of_CV_trajectories, vamp2_score). The procedure is
    run-aware: lagged pairs are formed within each run, via a ConcatDataset of per-run
    TrajectoryDatasets, and never across run boundaries.
    """
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
    """Compute the VAMP-r score of a run-aware list of CV trajectories.

    A higher score indicates that the CVs capture more of the slow dynamics. The score
    is used to compare the VAMPnet CVs against the TICA baseline.
    """
    _require()
    cvs = [np.asarray(c, dtype=np.float64) for c in cvs]
    model = VAMP(lagtime=int(lag)).fit_fetch(cvs if len(cvs) > 1 else cvs[0])
    return float(model.score(r=r))
