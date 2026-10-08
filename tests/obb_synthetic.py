"""Synthetic wings for the OBB tests (no GPU, no real data). Imported by name (`from obb_synthetic import ...`):
pytest puts this directory on sys.path, and a package called `tests` already exists in site-packages."""

import cv2
import numpy as np

N_LANDMARKS = 19


def make_landmarks(rng: np.random.Generator, centre=(400.0, 160.0), half_length=300.0, half_width=100.0, angle_deg=0.0) -> np.ndarray:
    """19 points scattered over an ellipse-like wing, rotated by angle_deg (top-left pixel coordinates).
    Points 0 and 1 sit at the two ends of the axis, so the extent along it is exactly 2 * half_length."""
    u = rng.uniform(-1, 1, N_LANDMARKS) * half_length
    v = rng.uniform(-1, 1, N_LANDMARKS) * half_width * np.sqrt(np.clip(1 - (u / half_length) ** 2, 0.05, 1))
    u[0], u[1] = -half_length, half_length
    v[0], v[1] = 0.0, 0.0
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.stack([u, v], axis=1) @ rot.T + np.asarray(centre)


def make_wing_image(width: int = 800, height: int = 326, background=(170, 180, 160)) -> np.ndarray:
    """BGR image with a flat background and a darker ellipse as the 'wing'."""
    img = np.full((height, width, 3), background, np.uint8)
    cv2.ellipse(img, (width // 2, height // 2), (int(width * 0.375), int(height * 0.3)), 0, 0, 360, (60, 70, 50), -1)
    return img
