import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def center_shape(x):
    """Removes translation by shifting the shape so that its centroid is at (0, 0)."""
    return x - x.mean(dim=0, keepdim=True)


def normalize_shape(x):
    """Scales the shape so that it has unit length (normalizes its size)."""
    return x / torch.norm(x)


def procrustes_align(x, y, only_matrix=False):
    """
    Aligns shape X to shape Y by rotation (Procrustes alignment).
    Assumes both shapes are already centered and normalized.
    Uses SVD.
    """
    u, _, vt = torch.linalg.svd(x.T @ y)
    r = u @ vt
    if torch.det(r) < 0:
        # det(r)<0 means that it is a flip, not a rotation
        u[:, -1] *= -1
        r = u @ vt
    if only_matrix:
        return r
    return x @ r


def generalized_procrustes_analysis(shapes, tol=1e-6, max_iter=100, device=torch.device('cpu')):
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


def recover_order(mean_shape, unordered_shape, max_iter=5, device=torch.device('cpu')):
    """
    mean_shape: torch tensor (n_points, 2)
    unordered_shape: torch tensor (n_points, 2) - same points but random order/transform
    Returns:
        reordered_shape: torch tensor (n_points, 2) = unordered_shape[perm_idx]
    """
    mean = mean_shape.to(device).float()
    s = unordered_shape.to(device).float()
    assert s.shape[0] == mean.shape[0] and mean.shape[1] == 2

    mean_torch = normalize_shape(center_shape(mean))
    shapes_torch = normalize_shape(center_shape(s))

    mean_temp = mean_torch.cpu().numpy()
    shapes_temp = shapes_torch.cpu().numpy()

    cost = cdist(mean_temp, shapes_temp)  # (n,n)
    index = solve_assignment(cost)  # idx[row]=col

    for it in range(max_iter):
        perm = torch.tensor(index, dtype=torch.long, device=device)
        s_perm = shapes_torch[perm]

        r = procrustes_align(s_perm, mean_torch, only_matrix=True)
        s_rot = shapes_torch @ r  # (n,2)
        cost = cdist(mean_temp, s_rot.cpu().numpy())
        new_idx = solve_assignment(cost)

        if np.array_equal(new_idx, index):
            break
        index = new_idx

    reordered_shape = unordered_shape[index]

    return reordered_shape


def handle_coordinates(coords, mean_coords):
    mask_coords = coords.detach().clone()
    if len(mask_coords) > 19:
        extra_points = len(mask_coords) - len(mean_coords)
        best_loss = float('inf')
        best_coords = None

        if extra_points <= 3:
            for remove_idx in itertools.combinations(range(len(mask_coords)), extra_points):
                reduced = torch.stack(
                    [p for i, p in enumerate(mask_coords) if i not in remove_idx]
                )
                reordered = recover_order(mean_coords, reduced)
                gpa = procrustes_align(normalize_shape(center_shape(reordered)), mean_coords)
                loss = torch.norm(gpa - mean_coords).item()
                if loss < best_loss:
                    best_loss = loss
                    best_coords = reduced
            mask_coords = best_coords
        else:
            mask_coords = mask_coords[:19]

    elif len(mask_coords) <= 18:
        missing_points = len(mean_coords) - len(mask_coords)
        best_loss = float('inf')
        best_missing_idxs = None
        best_reordered = None
        best_temp_mean = None

        if missing_points <= 3:
            for remove_idx in itertools.combinations(range(len(mean_coords)), missing_points):
                temp_mean = torch.stack(
                    [p for i, p in enumerate(mean_coords) if i not in remove_idx]
                )
                temp_mean_cn = normalize_shape(center_shape(temp_mean))
                reordered = recover_order(temp_mean_cn, mask_coords)
                gpa = procrustes_align(normalize_shape(center_shape(reordered)), temp_mean_cn)
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
                only_matrix=True
            )

            mean_coords_temp = ((mean_coords - t_mean) / s_mean) @ r.T * s_coords + t_coords
            missing_points_tensor = mean_coords_temp[list(best_missing_idxs)]

            mask_coords = torch.cat([best_reordered, missing_points_tensor], dim=0)
        else:
            xmin, ymin = mask_coords.min(dim=0).values
            xmax, ymax = mask_coords.max(dim=0).values
            random_x = torch.empty(missing_points).uniform_(xmin.item(), xmax.item())
            random_y = torch.empty(missing_points).uniform_(ymin.item(), ymax.item())
            random_points = torch.stack([random_x, random_y], dim=1)
            mask_coords = torch.cat([mask_coords, random_points], dim=0)

    reordered = recover_order(mean_coords, mask_coords)
    return reordered
