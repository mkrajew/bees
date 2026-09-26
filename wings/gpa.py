import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# A reasonable general-purpose multistart_angles default (see recover_order /
# handle_coordinates) for input that could plausibly be rotated by any amount
# -- e.g. a real photo uploaded to the app, not just the +/-90 degree range
# TrainAugmentConfig trains with. 8 candidates 45 degrees apart, tried both
# normal and mirrored when allow_reflection=True, at close to single-start
# cost (see handle_coordinates's docstring for the measurements this is based
# on).
FULL_ROTATION_MULTISTART_ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


def center_shape(x):
    """Removes translation by shifting the shape so that its centroid is at (0, 0)."""
    return x - x.mean(dim=0, keepdim=True)


def normalize_shape(x):
    """Scales the shape so that it has unit length (normalizes its size)."""
    return x / torch.norm(x)


def procrustes_align(x, y, only_matrix=False, allow_reflection=False):
    """
    Aligns shape X to shape Y by rotation (Procrustes alignment).
    Assumes both shapes are already centered and normalized.
    Uses SVD.

    allow_reflection: if False (default, backward compatible), a det(r)<0
        solution (a reflection) is corrected back to the nearest pure
        rotation -- this is what you want when computing/aligning to the
        mean shape itself, where every input is known to share the same
        chirality. If True, the uncorrected SVD-optimal orthogonal alignment
        is used instead, which may include a reflection when that fits
        better -- needed wherever the shape being aligned could legitimately
        be mirrored (e.g. matching a prediction from a horizontally-flipped
        image against mean_coords).
    """
    u, _, vt = torch.linalg.svd(x.T @ y)
    r = u @ vt
    if not allow_reflection and torch.det(r) < 0:
        # det(r)<0 means that it is a flip, not a rotation
        u[:, -1] *= -1
        r = u @ vt
    if only_matrix:
        return r
    return x @ r


def generalized_procrustes_analysis(
    shapes, tol=1e-6, max_iter=100, device=torch.device("cpu")
):
    """
    Performs Generalized Procrustes Analysis (GPA) on a set of 2D shapes.

    Args:
        shapes (torch.Tensor): Tensor of shape (N, n_points, 2) containing N shapes.
        tol (float): Convergence tolerance for mean shape updates.
        max_iter (int): Maximum number of iterations allowed.
        device (torch.device): Device on which to perform computations.

    Returns:
        mean_shape (torch.Tensor): The resulting mean shape of shape (n_points, 2).
    """
    shapes = shapes.to(device, dtype=torch.float32)
    shapes = torch.stack([normalize_shape(center_shape(s)) for s in shapes])
    mean_shape = normalize_shape(shapes.mean(dim=0))

    for i in range(max_iter):
        aligned = []
        for s in shapes:
            aligned.append(procrustes_align(s, mean_shape))
        aligned = torch.stack(aligned)

        new_mean = normalize_shape(aligned.mean(dim=0))
        diff = torch.norm(mean_shape - new_mean)
        mean_shape = new_mean

        print(f"Iteration {i} diff: {diff}")

        if diff < tol:
            print(f"Convergence reached after {i + 1} iterations.")
            break

    return mean_shape


def solve_assignment(cost_matrix):
    """
    Solves the linear assignment problem for a given cost matrix.
    Returns an array of length n_mean, where idx[j] is the index
            of the shape assigned to mean j.
    """
    r, c = linear_sum_assignment(cost_matrix)
    idx = np.empty(cost_matrix.shape[0], dtype=int)
    idx[r] = c
    return idx


def _rotate_2d(points, degrees):
    """Rotates (n,2) points about the origin by `degrees` (counter-clockwise).
    Used to seed recover_order's initial correspondence guess from several
    candidate starting orientations -- see multistart_angles there."""
    theta = torch.deg2rad(torch.as_tensor(float(degrees), dtype=points.dtype, device=points.device))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([
        torch.stack([cos_t, sin_t]),
        torch.stack([-sin_t, cos_t]),
    ])
    return points @ rot


def _pca_angle(points):
    """Angle (degrees) of the dominant principal axis of a centered 2D point
    cloud, via eigendecomposition of its 2x2 covariance matrix. Ambiguous up
    to 180 degrees -- a principal axis has no inherent "forward" direction,
    only an orientation -- so callers needing a specific heading must try
    both this angle and its +180 counterpart (see _pca_prealign_angles)."""
    cov = points.T @ points
    _, eigvecs = torch.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # eigh returns ascending eigenvalues; last = largest
    return torch.rad2deg(torch.atan2(principal[1], principal[0])).item()


def _pca_prealign_angles(mean_torch, shapes_torch):
    """Two candidate rotation angles (180 degrees apart, see _pca_angle) that
    would align shapes_torch's own dominant axis to mean_torch's -- a cheap
    (one 2x2 eigendecomposition each), data-driven alternative to a fixed
    angle grid like FULL_ROTATION_MULTISTART_ANGLES: instead of blindly
    trying a fixed spread of angles, this estimates the *actual* orientation
    of this specific shape directly from its own point cloud, independent of
    landmark identity/order (PCA doesn't care which point is which, only
    their overall spread) -- so it isn't thrown off by the same
    extra/missing-point noise that makes a fixed grid or the raw unrotated
    guess unreliable at hard angles. Both inputs must already be centered."""
    delta = _pca_angle(mean_torch) - _pca_angle(shapes_torch)
    return (delta, delta + 180.0)


def _recover_order_index(mean_torch, shapes_torch, max_iter, device, allow_reflection):
    """Runs the iterative nearest-assignment + Procrustes-realign loop from a
    given starting point-cloud orientation (mean_torch/shapes_torch already
    centered+normalized). Returns (index, residual), where residual is the
    final alignment error -- used by recover_order to compare candidate
    starting orientations (rotated and/or mirrored)."""
    mean_temp = mean_torch.cpu().numpy()
    shapes_temp = shapes_torch.cpu().numpy()

    cost = cdist(mean_temp, shapes_temp)  # (n,n)
    index = solve_assignment(cost)  # idx[row]=col

    for it in range(max_iter):
        perm = torch.tensor(index, dtype=torch.long, device=device)
        s_perm = shapes_torch[perm]

        r = procrustes_align(s_perm, mean_torch, only_matrix=True, allow_reflection=allow_reflection)
        s_rot = shapes_torch @ r  # (n,2)
        cost = cdist(mean_temp, s_rot.cpu().numpy())
        new_idx = solve_assignment(cost)

        if np.array_equal(new_idx, index):
            break
        index = new_idx

    perm = torch.tensor(index, dtype=torch.long, device=device)
    final_aligned = procrustes_align(shapes_torch[perm], mean_torch, allow_reflection=allow_reflection)
    residual = torch.norm(final_aligned - mean_torch).item()
    return index, residual


def recover_order(
    mean_shape,
    unordered_shape,
    max_iter=5,
    device=torch.device("cpu"),
    allow_reflection=False,
    multistart_angles=(0.0,),
    pca_prealign=False,
):
    """
    mean_shape: torch tensor (n_points, 2)
    unordered_shape: torch tensor (n_points, 2) - same points but random order/transform
    allow_reflection: see procrustes_align -- pass True when unordered_shape
        might be a mirrored version of mean_shape (e.g. a prediction from a
        horizontally-flipped image).

        Note this needs more than just passing allow_reflection through to
        procrustes_align: the very first correspondence guess (before any
        rotation/reflection is applied) is a raw nearest-neighbor match on
        the unaligned points, which for a genuinely mirrored shape is close
        to arbitrary -- the small fixed number of refinement iterations
        can't reliably recover from that bad a start. So when
        allow_reflection is set, every candidate below is additionally tried
        both normal and pre-mirrored.
    multistart_angles: candidate starting rotation angles (degrees, applied to
        unordered_shape) to seed that same initial correspondence guess with.
        Rotation magnitude alone doesn't stop procrustes_align from recovering
        the true alignment once a decent number of points are matched
        correctly -- its SVD fit is exact for any angle -- but for a large
        true rotation (e.g. near 90 degrees) combined with the small
        imperfections real predictions have (extra/missing points,
        positional jitter), the *unrotated* initial guess can be bad enough
        that the fixed small number of refinement iterations gets stuck in
        the wrong permutation, the same failure mode as the mirrored case
        above. Pre-rotating the input before that first guess (only) gives
        it a fair starting point for each candidate angle; whichever
        candidate (angle x reflection) converges to the lowest residual is
        used. Rotating/mirroring only changes coordinate values, not row
        order, so the resulting index is always valid against the original
        unordered_shape. The default (0.0,) preserves the previous
        single-unrotated-start behavior exactly.
    pca_prealign: if True, also try the (up to two) rotation angles that PCA
        estimates would align unordered_shape's own dominant axis to
        mean_shape's -- see _pca_prealign_angles. Cheap (two 2x2
        eigendecompositions, negligible next to the O(max_iter) Hungarian
        matching this function already does) and, unlike multistart_angles's
        fixed grid, adapts to this specific shape's actual orientation
        instead of blindly sampling a spread -- can replace a much wider
        fixed grid rather than just supplementing it.
    Returns:
        reordered_shape: torch tensor (n_points, 2) = unordered_shape[perm_idx]
    """
    mean = mean_shape.to(device).float()
    s = unordered_shape.to(device).float()
    assert s.shape[0] == mean.shape[0] and mean.shape[1] == 2

    mean_torch = normalize_shape(center_shape(mean))
    shapes_torch = normalize_shape(center_shape(s))

    angles = tuple(multistart_angles)
    if pca_prealign:
        angles = angles + _pca_prealign_angles(mean_torch, shapes_torch)

    reflect_options = (False, True) if allow_reflection else (False,)

    best_index, best_residual = None, None
    for angle in angles:
        base = shapes_torch if angle == 0.0 else _rotate_2d(shapes_torch, angle)
        for reflect in reflect_options:
            candidate = base
            if reflect:
                candidate = base.clone()
                candidate[:, 0] *= -1
            index, residual = _recover_order_index(mean_torch, candidate, max_iter, device, allow_reflection)
            if best_residual is None or residual < best_residual:
                best_index, best_residual = index, residual

    reordered_shape = unordered_shape[best_index]

    return reordered_shape


def _estimate_orientation(mean_coords, coords, allow_reflection, multistart_angles, max_iter=5):
    """Cheap, non-combinatorial upfront pass: picks whichever candidate
    (angle, reflection) from multistart_angles converges to the lowest
    residual for `coords` against mean_coords as a whole, using a naive
    same-size truncation/pad when their counts don't match exactly (fine
    here -- this is only used to estimate the shape's *overall* orientation,
    not an exact per-point correspondence). Used by handle_coordinates to
    give its expensive per-combination search (extra/missing point
    selection) a single good starting angle instead of paying full
    multistart cost on every combination -- see handle_coordinates."""
    n = len(mean_coords)
    c = coords[:n] if len(coords) > n else coords
    if len(c) < n:
        c = torch.cat([c, mean_coords[len(c):n].clone()], dim=0)

    mean_torch = normalize_shape(center_shape(mean_coords.float()))
    shapes_torch = normalize_shape(center_shape(c.float()))
    reflect_options = (False, True) if allow_reflection else (False,)

    best_angle, best_residual = multistart_angles[0], None
    for angle in multistart_angles:
        base = shapes_torch if angle == 0.0 else _rotate_2d(shapes_torch, angle)
        for reflect in reflect_options:
            candidate = base.clone() if reflect else base
            if reflect:
                candidate[:, 0] *= -1
            _, residual = _recover_order_index(
                mean_torch, candidate, max_iter, torch.device("cpu"), allow_reflection
            )
            if best_residual is None or residual < best_residual:
                best_angle, best_residual = angle, residual
    return best_angle


def handle_coordinates(
    coords, mean_coords, allow_reflection=False, multistart_angles=(0.0,), pca_prealign=False
):
    """
    allow_reflection: see procrustes_align -- pass True when coords might come
    from a horizontally-flipped image (e.g. TrainAugmentConfig's
    horizontal_flip_p > 0). Without this, procrustes_align's rotation-only
    constraint means a mirrored prediction can be matched to entirely wrong
    landmark identities even though the underlying detection is accurate.
    multistart_angles: see recover_order -- pass a spread of candidate angles
    when coords might come from a strongly rotated image, to avoid
    recover_order's correspondence search getting stuck from a bad initial
    guess at large rotations. Applied directly to the final recover_order
    call below. For the itertools.combinations search above it (extra/missing
    point selection), applying full multistart to every candidate combination
    would multiply an already combinatorial cost by len(multistart_angles) x
    (2 if allow_reflection else 1) -- measured up to ~7x slower on real
    mismatched-point-count predictions, which are the common case here. So
    when more than one angle is given, a single cheap non-combinatorial pass
    (_estimate_orientation) first picks one extra candidate starting angle
    from the full set, and only that angle plus the untouched 0.0 (the old
    default, already sufficient for small/moderate rotations -- the failure
    mode this whole feature targets only shows up near hard angles like 90
    degrees) seed every combination in the search, instead of the full set --
    measured to recover nearly all of the accuracy of full per-combination
    multistart (2.69px vs 2.60px mean error at 90 degrees rotation on the
    real trained checkpoint, vs. 12.7px unfixed) at close to single-start
    cost (measured near-identical wall time on real mismatched-point-count
    predictions -- the fixed per-combination overhead dominates over the
    1-vs-2-candidates difference -- vs. up to ~7x slower for full
    per-combination multistart),
    *and*, unlike using the estimated angle alone, without regressing angles
    the plain 0.0 start already handled fine (an earlier version of this that
    dropped 0.0 in favor of the estimate alone regressed 15/30 degrees --
    0.95->1.58px / 1.08->2.13px -- because a single coarse-grid estimate can
    occasionally be a worse seed than plain 0.0 for a combination the
    estimate wasn't computed from). When multistart_angles is left at its
    default (a single angle), this pre-pass is skipped and behavior/cost are
    unchanged from before this parameter existed.
    pca_prealign: see recover_order -- unlike multistart_angles's fixed grid,
    PCA-derived candidates cost one 2x2 eigendecomposition each, cheap enough
    to apply directly to *every* recover_order call here, including inside
    the itertools.combinations search, without that search's combinatorial
    blowup. When True, this replaces the _estimate_orientation workaround
    above for the inner search (each combination gets its own PCA estimate,
    tailored to exactly the points in that combination, rather than sharing
    one estimate computed from a naive truncation of the whole coords) and
    is passed straight through to the final call too, on top of whatever
    multistart_angles already provides there.
    """
    mask_coords = coords.detach().clone()

    if pca_prealign:
        inner_multistart_angles = (0.0,)
    else:
        inner_multistart_angles = multistart_angles
        if len(multistart_angles) > 1:
            est_angle = _estimate_orientation(mean_coords, mask_coords, allow_reflection, multistart_angles)
            inner_multistart_angles = (0.0, est_angle) if est_angle != 0.0 else (0.0,)

    if len(mask_coords) > 19:
        extra_points = len(mask_coords) - len(mean_coords)
        best_loss = float("inf")
        best_coords = None

        if extra_points <= 3:
            for remove_idx in itertools.combinations(
                range(len(mask_coords)), extra_points
            ):
                reduced = torch.stack(
                    [p for i, p in enumerate(mask_coords) if i not in remove_idx]
                )
                reordered = recover_order(
                    mean_coords, reduced, allow_reflection=allow_reflection,
                    multistart_angles=inner_multistart_angles, pca_prealign=pca_prealign,
                )
                gpa = procrustes_align(
                    normalize_shape(center_shape(reordered)), mean_coords, allow_reflection=allow_reflection
                )
                loss = torch.norm(gpa - mean_coords).item()
                if loss < best_loss:
                    best_loss = loss
                    best_coords = reduced
            mask_coords = best_coords
        else:
            mask_coords = mask_coords[:19]

    elif len(mask_coords) <= 18:
        missing_points = len(mean_coords) - len(mask_coords)
        best_loss = float("inf")
        best_missing_idxs = None
        best_reordered = None
        best_temp_mean = None

        if missing_points <= 3:
            for remove_idx in itertools.combinations(
                range(len(mean_coords)), missing_points
            ):
                temp_mean = torch.stack(
                    [p for i, p in enumerate(mean_coords) if i not in remove_idx]
                )
                temp_mean_cn = normalize_shape(center_shape(temp_mean))
                reordered = recover_order(
                    temp_mean_cn, mask_coords, allow_reflection=allow_reflection,
                    multistart_angles=inner_multistart_angles, pca_prealign=pca_prealign,
                )
                gpa = procrustes_align(
                    normalize_shape(center_shape(reordered)), temp_mean_cn, allow_reflection=allow_reflection
                )
                loss = torch.norm(gpa - temp_mean_cn).item()

                if loss < best_loss:
                    best_loss = loss
                    best_missing_idxs = remove_idx
                    best_reordered = reordered
                    best_temp_mean = temp_mean

            t_coords = best_reordered.mean(dim=0, keepdim=True)
            s_coords = torch.norm(center_shape(best_reordered))

            t_mean = best_temp_mean.mean(dim=0, keepdim=True)
            s_mean = torch.norm(center_shape(best_temp_mean))

            r = procrustes_align(
                normalize_shape(center_shape(best_reordered)),
                normalize_shape(center_shape(best_temp_mean)),
                only_matrix=True,
                allow_reflection=allow_reflection,
            )

            mean_coords_temp = (
                (mean_coords - t_mean) / s_mean
            ) @ r.T * s_coords + t_coords
            missing_points_tensor = mean_coords_temp[list(best_missing_idxs)]

            mask_coords = torch.cat([best_reordered, missing_points_tensor], dim=0)
        else:
            if len(mask_coords) < 2:
                mask_coords = torch.rand(len(mean_coords), 2) * 100
            else:
                xmin, ymin = mask_coords.min(dim=0).values
                xmax, ymax = mask_coords.max(dim=0).values
                eps = 1e-6
                random_x = torch.empty(missing_points).uniform_(
                    xmin.item(), max(xmax.item(), xmin.item() + eps)
                )
                random_y = torch.empty(missing_points).uniform_(
                    ymin.item(), max(ymax.item(), ymin.item() + eps)
                )
                random_points = torch.stack([random_x, random_y], dim=1)
                mask_coords = torch.cat([mask_coords, random_points], dim=0)

    reordered = recover_order(
        mean_coords, mask_coords, allow_reflection=allow_reflection,
        multistart_angles=multistart_angles, pca_prealign=pca_prealign,
    )
    return reordered
