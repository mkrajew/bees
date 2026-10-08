"""Oriented bounding boxes (OBB) of wings, computed from the 19 landmarks with the PCA recipe.

Coordinates are image pixels with the origin in the top-left corner (x to the right, y down).
The box is described by its centre, its side along the wing axis (`length`), its side across
(`width`) and the axis angle. The margins match the axis-aligned labels of the old detector
(`wings.detection.dataset.process_bbox`: x 1.2, y 1.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LENGTH_FACTOR = 1.2  # margin along the wing axis
WIDTH_FACTOR = 1.4  # margin across the wing axis
MIN_EIG_RATIO = 1.5  # below this the principal axis is ill-defined (reported, never enforced)
CORNER_COLUMNS = [f"{axis}{i}" for i in range(1, 5) for axis in ("x", "y")]  # label-table columns x1, y1, ..., x4, y4


@dataclass(frozen=True)
class Obb:
    """Oriented box. `theta_deg` is the axis angle in (-90, 90]; `eig_ratio` is the ratio of the
    two covariance eigenvalues of the landmarks (large = well-defined axis)."""

    cx: float
    cy: float
    length: float
    width: float
    theta_deg: float
    eig_ratio: float

    @property
    def axis(self) -> np.ndarray:
        t = np.deg2rad(self.theta_deg)
        return np.array([np.cos(t), np.sin(t)])

    @property
    def normal(self) -> np.ndarray:
        a = self.axis
        return np.array([-a[1], a[0]])

    def corners(self) -> np.ndarray:
        """(4, 2) corners, in the (axis, normal) frame: (-L/2,-W/2), (+L/2,-W/2), (+L/2,+W/2), (-L/2,+W/2)."""
        a, n = self.axis, self.normal
        centre = np.array([self.cx, self.cy])
        hl, hw = self.length / 2.0, self.width / 2.0
        return np.stack([centre - hl * a - hw * n, centre + hl * a - hw * n, centre + hl * a + hw * n, centre - hl * a + hw * n])


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit vector of the largest covariance eigenvalue (sign: x >= 0, and y > 0 if x == 0) and
    the ratio largest / smallest eigenvalue."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError(f"expected at least 3 points of shape (n, 2), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("non-finite landmark coordinates")
    eigvals, eigvecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))  # ascending order
    if eigvals[-1] <= 1e-12:
        raise ValueError("degenerate landmark set: no spread")
    axis = eigvecs[:, -1]
    if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
        axis = -axis
    return axis, float(eigvals[-1] / max(eigvals[0], 1e-12))


def obb_from_landmarks(points: np.ndarray, length_factor: float = LENGTH_FACTOR, width_factor: float = WIDTH_FACTOR) -> Obb:
    """PCA recipe: axis = direction of the largest spread; the box spans the extreme projections
    of the points on the axis and on its normal, enlarged by the margin factors about the midpoint."""
    pts = np.asarray(points, dtype=np.float64)
    axis, ratio = principal_axis(pts)
    normal = np.array([-axis[1], axis[0]])
    centroid = pts.mean(axis=0)
    u = (pts - centroid) @ axis
    v = (pts - centroid) @ normal
    centre = centroid + (u.min() + u.max()) / 2.0 * axis + (v.min() + v.max()) / 2.0 * normal
    length = (u.max() - u.min()) * length_factor
    width = (v.max() - v.min()) * width_factor
    if width <= 1e-6 * length:
        raise ValueError("landmarks are collinear: the box has no width")
    theta = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return Obb(float(centre[0]), float(centre[1]), float(length), float(width), theta, ratio)


def points_inside_box(corners: np.ndarray, points: np.ndarray, tol: float = 1e-6) -> bool:
    """True if every point lies inside the rectangle given by `corners` (order as in `Obb.corners()`), within `tol` px."""
    a, n = corners[1] - corners[0], corners[3] - corners[0]
    rel = np.asarray(points, dtype=np.float64) - corners[0]
    s, t = rel @ a / (a @ a), rel @ n / (n @ n)
    slack_s, slack_t = tol / np.linalg.norm(a), tol / np.linalg.norm(n)
    return bool(((s >= -slack_s) & (s <= 1 + slack_s) & (t >= -slack_t) & (t <= 1 + slack_t)).all())


def obb_to_xywhr(obb: Obb) -> tuple[float, float, float, float, float]:
    """(cx, cy, w, h, r) in the Ultralytics convention: w >= h, r in radians within [-pi/4, 3*pi/4)."""
    w, h, r = obb.length, obb.width, float(np.deg2rad(obb.theta_deg))
    if w < h:
        w, h = h, w
        r += np.pi / 2
    while r >= 3 * np.pi / 4:
        r -= np.pi
    while r < -np.pi / 4:
        r += np.pi
    return obb.cx, obb.cy, w, h, r


def reference_landmarks(mean_shape: np.ndarray) -> tuple[int, int]:
    """Indices (i_lo, i_hi) of the mean-shape landmarks with the smallest and largest projection on
    its principal axis. They define a consistent wing direction for every image."""
    shape = np.asarray(mean_shape, dtype=np.float64)
    axis, _ = principal_axis(shape)
    projection = (shape - shape.mean(axis=0)) @ axis
    return int(projection.argmin()), int(projection.argmax())


def direction_sign(points: np.ndarray, axis: np.ndarray, i_lo: int, i_hi: int) -> int:
    """+1 if the vector from landmark i_lo to landmark i_hi points along `axis`, else -1."""
    pts = np.asarray(points, dtype=np.float64)
    return 1 if float((pts[i_hi] - pts[i_lo]) @ axis) >= 0.0 else -1


def background_color(image: np.ndarray, d: int = 5, offset: int = 2, width: int = 5, q: int = 8) -> tuple[int, int, int]:
    """BGR background colour of a wing image, by the same rule as `wings.detection.dataset.pad_image`:
    the most frequent colour of the pixel rows and columns at distance `d` from the border; if that is
    pure black or pure white, the dominant colour of the inner border strip."""
    h, w = image.shape[:2]
    d = max(0, min(d, h // 2 - 1, w // 2 - 1))
    frame = np.concatenate([image[d], image[-d - 1], image[:, d], image[:, -d - 1]])
    colors, counts = np.unique(frame, axis=0, return_counts=True)
    color = tuple(int(c) for c in colors[counts.argmax()])
    if color in [(255, 255, 255), (0, 0, 0)]:
        from wings.detection.dataset import dominant_inner_border_color

        color = dominant_inner_border_color(image, offset=offset, width=width, q=q)
    return color
