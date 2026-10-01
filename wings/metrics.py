"""
Shared shape-comparison metrics for landmark predictions, used by the
evaluation notebooks (25, 26; originally prototyped ad hoc in 20). Two
downstream metrics are computed from a GPA-ordered pair of shapes:

- Raw per-point pixel distance -- production-realistic accuracy, in the same
  units as val_mean_error_px, but not comparable across images of different
  scale/resolution and with no absolute reference point.
- `wing_positional_precision` -- DeepWings' own metric (Rodrigues et al.
  2022, BDCC 6(3):70), reported in their paper as an average of 0.943.
  Translation/scale/rotation-invariant (Procrustes-superimposed first), so
  it asks "is the shape right" rather than "are the points in the right
  place," and is bounded [0, 1] with a published external reference point --
  this is the headline number to report for cross-config and cross-paper
  comparison, not the pixel distances.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist, pdist, squareform

from wings.gpa import FULL_ROTATION_MULTISTART_ANGLES, handle_coordinates


def gpa_order(coords, mean_coords, allow_reflection=True,
              multistart_angles=FULL_ROTATION_MULTISTART_ANGLES, pca_prealign=False):
    """GPA-orders an unordered point set against mean_coords. Returns None if
    coords is empty (nothing to order)."""
    if len(coords) == 0:
        return None
    t = torch.as_tensor(coords, dtype=torch.float32)
    return handle_coordinates(
        t, mean_coords, allow_reflection=allow_reflection,
        multistart_angles=multistart_angles, pca_prealign=pca_prealign,
    )


def wing_positional_precision(gt, pred):
    """DeepWings' own "metric 2" -- reported in their paper as an average
    positional precision of 0.943. gt/pred are (19, 2) arrays of
    *corresponding* landmarks (same landmark index = same landmark on both
    sides, exactly what GPA ordering above guarantees). Superimposes both
    shapes via Procrustes (centered, unit-scale, pred optimally rotated onto
    gt), takes every pairwise inter-landmark distance within each shape, and
    scores each of the 342 off-diagonal pairs as min(d_gt, d_pred) /
    max(d_gt, d_pred) -- 1.0 means the two shapes' pairwise geometry is
    identical. Returns (wing_precision, per_landmark)."""
    mtx_gt, mtx_pred, _ = procrustes(np.asarray(gt, float), np.asarray(pred, float))
    d_gt = squareform(pdist(mtx_gt))
    d_pred = squareform(pdist(mtx_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(d_gt, d_pred) / np.maximum(d_gt, d_pred)
    np.fill_diagonal(ratio, np.nan)
    return np.nanmean(ratio), np.nanmean(ratio, axis=1)


def nn_matched_distances(pred_coords, true_coords):
    """Diagnostic-only direct Hungarian match in shared pixel space, ignoring
    landmark identity/order -- tells apart "detection itself is wrong" (both
    this and the GPA-ordered metric are bad) from "the GPA ordering step got
    confused" (this stays low while the GPA-ordered metric spikes). Matches
    min(len(pred), len(true)) pairs. NOT a valid stand-in for a real accuracy
    metric: it uses ground truth as a matching shortcut unavailable at real
    inference time."""
    if len(pred_coords) == 0 or len(true_coords) == 0:
        return None
    pred = np.asarray(pred_coords)
    true = np.asarray(true_coords)
    cost = cdist(true, pred)
    row, col = linear_sum_assignment(cost)
    return torch.from_numpy(cost[row, col]).float()


def gpa_ordered_metrics_single(pred_coords, true_ordered, mean_coords, **gpa_kwargs):
    """For evaluation setups where ground truth is already in canonical
    landmark order (e.g. transforming the dataset's own labeled keypoints
    through a known augmentation -- notebook 25's in-domain test set) and
    only the prediction needs GPA ordering against mean_coords. Returns
    (distances, precision); (None, nan) if pred_coords is empty."""
    pred_o = gpa_order(pred_coords, mean_coords, **gpa_kwargs)
    if pred_o is None:
        return None, float("nan")
    dist = torch.norm(pred_o - true_ordered, dim=1)
    try:
        precision, _ = wing_positional_precision(true_ordered.numpy(), pred_o.numpy())
    except ValueError:
        precision = float("nan")
    return dist, precision


def gpa_ordered_metrics_both(pred_coords, gt_coords, mean_coords, **gpa_kwargs):
    """For evaluation setups where NEITHER side is in canonical order (e.g.
    an external dataset with its own annotation convention -- notebook 26's
    DeepWings evaluation): GPA-orders both sides against mean_coords once,
    then derives distances/precision from that single pass. Returns
    (distances, precision); (None, nan) if ordering failed on either side
    (empty prediction/ground truth, or a degenerate all-identical shape
    scipy.procrustes rejects)."""
    pred_o = gpa_order(pred_coords, mean_coords, **gpa_kwargs)
    gt_o = gpa_order(gt_coords, mean_coords, **gpa_kwargs)
    if pred_o is None or gt_o is None:
        return None, float("nan")
    dist = torch.norm(pred_o - gt_o, dim=1)
    try:
        precision, _ = wing_positional_precision(gt_o.numpy(), pred_o.numpy())
    except ValueError:
        precision = float("nan")
    return dist, precision
