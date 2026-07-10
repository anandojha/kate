"""
Nonlinear slow collective variables from a VAMPnet.

A VAMPnet learns a low-dimensional embedding chi(x) in R^d of the molecular features
by maximizing the variational score of the dynamics it induces at lag time tau. From
the instantaneous and time-lagged embeddings one forms the covariance matrices
C_00 = <chi(x_t) chi(x_t)^T>, C_11 = <chi(x_{t+tau}) chi(x_{t+tau})^T> and
C_01 = <chi(x_t) chi(x_{t+tau})^T>, and the Koopman matrix
K = C_00^{-1/2} C_01 C_11^{-1/2}. The VAMP-r score is the sum of the r-th powers of
the singular values sigma_i of K, VAMP-r = sum_i sigma_i^r. By the variational
principle for Markov processes this score is bounded above by the value attained by
the exact singular functions of the transfer operator, so maximizing it drives chi
toward the true slow modes. The r = 2 score is the kinetic variance retained at lag
tau and is the objective used here (Mardt, Pasquali, Wu, Noe, Nat. Commun. 9, 5
(2018)).

The lobe is a small feed-forward network shared between the instantaneous and lagged
views. The learned chi(x) feed the flow and the MSM, which otherwise run on the linear
TICA coordinates, and the flow stays invertible on them. Lagged pairs are formed
within each run and never across run boundaries. For binding kinetics the features are
ligand-pocket contacts rather than raw Cartesian coordinates; the choice of features
rests with the caller.

deeptime supplies the VAMPNet, VAMP, and TrajectoryDataset primitives as the optional
[kinetics] extra, so its import is guarded while torch stays a core dependency.
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
            "vampnet_cv (T6) needs the kinetics engine: pip install kate[kinetics] "
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
    """VAMP-r score of a run-aware list of CV trajectories.

    The score is sum_i sigma_i^r over the singular values sigma_i of the Koopman matrix
    at lag tau; a larger value means the CVs retain more of the slow dynamics. Lagged
    pairs stay within each run.
    """
    _require()
    cvs = [np.asarray(c, dtype=np.float64) for c in cvs]
    model = VAMP(lagtime=int(lag)).fit_fetch(cvs if len(cvs) > 1 else cvs[0])
    return float(model.score(r=r))
