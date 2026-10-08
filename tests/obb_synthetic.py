"""Synthetic wings for the OBB tests (no GPU, no real data). Imported by name (`from obb_synthetic import ...`):
pytest puts this directory on sys.path, and a package called `tests` already exists in site-packages."""

from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

N_LANDMARKS = 19
SPLIT_PLAN = {"AA": ["train", "train", "train", "train", "val", "test"], "BB": ["train", "train", "val", "test"]}


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


def build_synthetic_raw(root, plan=None) -> SimpleNamespace:
    """Two 'countries' of raw wings with CSVs (y from the bottom, as in the real data), the old detector's
    split folders (empty files named like the images) and a mean shape in the same landmark order."""
    plan = SPLIT_PLAN if plan is None else plan
    rng = np.random.default_rng(42)
    raw, split_dir = root / "raw", root / "detection"
    for split in ("train", "val", "test"):
        (split_dir / "images" / split).mkdir(parents=True)
    names, landmarks, backgrounds = [], {}, {}
    for country, splits in plan.items():
        (raw / f"{country}-wing-images").mkdir(parents=True)
        records = []
        for i, split in enumerate(splits):
            name = f"{country}-{i:04d}.png"
            width, height = int(rng.integers(600, 900)), int(rng.integers(250, 350))
            background = tuple(int(c) for c in rng.integers(100, 200, 3))
            image = make_wing_image(width, height, background)
            points = make_landmarks(rng, centre=(width / 2, height / 2), half_length=0.35 * width, half_width=0.28 * height, angle_deg=float(rng.uniform(-8, 8)))
            cv2.imwrite(str(raw / f"{country}-wing-images" / name), image)
            (split_dir / "images" / split / name).touch()
            csv_xy = np.column_stack([points[:, 0], height - 1 - points[:, 1]]).reshape(-1)
            records.append([name, *csv_xy.tolist()])
            names.append(name)
            landmarks[name], backgrounds[name] = points, background
        columns = ["file"] + [f"{a}{k}" for k in range(1, 20) for a in "xy"]
        pd.DataFrame(records, columns=columns).to_csv(raw / f"{country}-raw-coordinates.csv", index=False)
    mean_shape = make_landmarks(np.random.default_rng(0), centre=(0.0, 0.0), half_length=1.0, half_width=0.35, angle_deg=0.0)
    return SimpleNamespace(raw=raw, split_dir=split_dir, countries=list(plan), names=names, landmarks=landmarks, backgrounds=backgrounds, mean_shape=mean_shape)
