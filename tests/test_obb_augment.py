import cv2
import numpy as np
import pytest
from obb_synthetic import make_landmarks, make_wing_image
from scipy import stats

from wings.detection.obb_augment import (
    AugConfig,
    Item,
    apply_affine,
    apply_photometric,
    make_single_sample,
    plan_placement,
    warp_image,
    warp_mask,
)
from wings.detection.obb_labels import Obb, obb_from_landmarks

PLAIN = AugConfig(photometric=False)
BACKGROUND = (200, 210, 190)


def painted_item(corners, fill, size=(326, 800), background=BACKGROUND) -> Item:
    """Source image whose only dark structure is the filled box itself, so label and pixels can be compared."""
    img = np.full((size[0], size[1], 3), background, np.uint8)
    cv2.fillPoly(img, [np.round(corners).astype(np.int32)], fill)
    return Item(img, np.asarray(corners, np.float64), background)


def painted_box_corners(canvas: np.ndarray, fill) -> np.ndarray:
    """Corners of the minimum-area rectangle of the pixels that have (about) the colour `fill`."""
    distance = np.abs(canvas.astype(np.int32) - np.array(fill)).sum(axis=2)
    ys, xs = np.nonzero(distance < 60)
    return cv2.boxPoints(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32)))


def corner_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from a corner of one box to the nearest corner of the other (both directions)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


BOX = Obb(cx=400.0, cy=163.0, length=500.0, width=180.0, theta_deg=3.0, eig_ratio=9.0).corners()


@pytest.mark.parametrize("seed", range(20))
def test_the_label_follows_the_pixels(seed):
    item = painted_item(BOX, fill=(30, 30, 30))
    sample = make_single_sample(item, np.random.default_rng(seed), PLAIN)
    assert sample.image.shape == (640, 640, 3) and sample.corners.shape == (1, 4, 2)
    assert corner_distance(sample.corners[0], painted_box_corners(sample.image, (30, 30, 30))) < 2.5


@pytest.mark.parametrize("seed", range(10))
def test_pixels_outside_the_source_image_have_exactly_the_background_colour(seed):
    item = Item(np.full((326, 800, 3), (200, 100, 50), np.uint8), BOX, (10, 20, 30))
    rng = np.random.default_rng(seed)
    placement = plan_placement(item.corners, 800, (0, 0, 640, 640), rng, PLAIN)
    canvas = warp_image(item.image, placement.matrix, 640, item.background)
    outside = cv2.erode((warp_mask(item.image.shape[:2], placement.matrix, 640) == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert outside.any()
    assert (canvas[outside] == (10, 20, 30)).all()


def random_boxes(rng, n):
    for _ in range(n):
        yield obb_from_landmarks(make_landmarks(rng, centre=(400, 160), half_length=rng.uniform(120, 380), half_width=rng.uniform(40, 150), angle_deg=rng.uniform(-30, 30))).corners()


def test_the_box_always_stays_inside_the_canvas_with_the_margin():
    rng = np.random.default_rng(0)
    cfg = AugConfig()
    for corners in random_boxes(rng, 100):
        for _ in range(5):
            placement = plan_placement(corners, 800, (0, 0, 640, 640), rng, cfg)
            assert placement.corners.min() >= cfg.edge_margin - 1e-6
            assert placement.corners.max() <= 640 - cfg.edge_margin + 1e-6


def test_a_box_that_exactly_fills_the_region_is_still_placed():
    """With fraction 1.0 the fit constraint binds at most angles and leaves no room to move; rounding must not break that."""
    cfg = AugConfig(fraction_range=(1.0, 1.0))
    rng = np.random.default_rng(4)
    for _ in range(2000):
        placement = plan_placement(BOX, 800, (0, 0, 640, 640), rng, cfg)
        assert placement.corners.min() >= cfg.edge_margin - 1e-6
        assert placement.corners.max() <= 640 - cfg.edge_margin + 1e-6


def test_the_rotation_angle_is_uniform():
    rng = np.random.default_rng(1)
    angles = np.array([plan_placement(BOX, 800, (0, 0, 640, 640), rng, AugConfig()).angle_deg for _ in range(4000)])
    assert angles.min() >= -180 and angles.max() <= 180
    assert stats.kstest((angles + 180) / 360, "uniform").pvalue > 0.01


def test_the_box_angle_modulo_180_is_uniform():
    rng = np.random.default_rng(2)
    thetas = []
    for _ in range(4000):
        corners = plan_placement(BOX, 800, (0, 0, 640, 640), rng, AugConfig()).corners
        edge = corners[1] - corners[0]
        thetas.append(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
    assert stats.kstest(np.array(thetas) / 180.0, "uniform").pvalue > 0.01


def test_flip_probability_and_scale_range():
    rng = np.random.default_rng(3)
    cfg = AugConfig()
    placements = [plan_placement(BOX, 800, (0, 0, 640, 640), rng, cfg) for _ in range(2000)]
    assert 0.45 < np.mean([p.flip for p in placements]) < 0.55
    lengths = np.array([np.linalg.norm(p.corners[1] - p.corners[0]) for p in placements])
    assert lengths.max() <= cfg.fraction_range[1] * 640 + 1e-6


def test_same_seed_same_sample_different_seed_different_sample():
    item = painted_item(BOX, fill=(30, 30, 30))
    a = make_single_sample(item, np.random.default_rng(5))
    b = make_single_sample(item, np.random.default_rng(5))
    c = make_single_sample(item, np.random.default_rng(6))
    assert (a.image == b.image).all() and (a.corners == b.corners).all()
    assert not np.allclose(a.corners, c.corners)


@pytest.mark.parametrize("size", [(40, 100), (2400, 6000)], ids=["tiny", "huge"])
def test_extreme_image_sizes_still_give_an_inside_box(size):
    h, w = size
    corners = Obb(cx=w / 2, cy=h / 2, length=0.9 * w, width=0.5 * h, theta_deg=0.0, eig_ratio=5.0).corners()
    item = Item(np.full((h, w, 3), BACKGROUND, np.uint8), corners, BACKGROUND)
    sample = make_single_sample(item, np.random.default_rng(0), PLAIN)
    assert sample.image.shape == (640, 640, 3)
    assert sample.corners.min() >= 0 and sample.corners.max() <= 640


def test_a_box_that_extends_beyond_the_source_image_is_fine():
    corners = Obb(cx=400.0, cy=163.0, length=860.0, width=360.0, theta_deg=0.0, eig_ratio=5.0).corners()  # larger than the 800x326 image
    item = Item(make_wing_image(), corners, BACKGROUND)
    sample = make_single_sample(item, np.random.default_rng(1), PLAIN)
    assert sample.corners.min() >= 0 and sample.corners.max() <= 640


def test_photometric_steps_keep_shape_and_dtype_and_change_pixels():
    img = make_wing_image(640, 640)
    out = apply_photometric(img, np.random.default_rng(0), AugConfig(blur_p=1.0, jpeg_p=1.0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 1.0


def test_apply_affine_accepts_two_by_three_and_three_by_three():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = np.array([[0.0, -1.0, 5.0], [1.0, 0.0, 6.0]])
    assert np.allclose(apply_affine(m, pts), apply_affine(np.vstack([m, [0, 0, 1]]), pts))
    assert np.allclose(apply_affine(m, pts)[0], [3.0, 7.0])
