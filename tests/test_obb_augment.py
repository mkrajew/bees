from collections import Counter

import cv2
import numpy as np
import pytest
from obb_synthetic import make_landmarks, make_wing_image
from scipy import stats

from wings.detection.obb_augment import (
    AugConfig,
    Item,
    TonePool,
    apply_affine,
    apply_photometric,
    choose_k,
    composition_picks,
    grid_cells,
    heal_edges,
    load_item,
    make_multi_sample,
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


def test_choose_k_follows_the_weights():
    rng = np.random.default_rng(0)
    ks = np.array([choose_k(rng) for _ in range(4000)])
    assert set(ks) == {2, 3, 4}
    assert abs((ks == 2).mean() - 0.5) < 0.05 and abs((ks == 3).mean() - 0.25) < 0.05


@pytest.mark.parametrize("k", [2, 3, 4])
def test_grid_cells_are_disjoint_and_inside_the_canvas(k):
    for seed in range(10):
        cells = grid_cells(k, 640, np.random.default_rng(seed))
        assert len(cells) == k
        for i, (x0, y0, x1, y1) in enumerate(cells):
            assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 640
            for x2, y2, x3, y3 in cells[i + 1 :]:
                assert x1 <= x2 or x3 <= x0 or y1 <= y2 or y3 <= y0


@pytest.mark.parametrize("k", [2, 3, 4])
def test_multi_wing_boxes_do_not_overlap_and_stay_inside(k):
    rng = np.random.default_rng(k)
    for _ in range(40):
        items = [painted_item(BOX, fill=(30, 30, 30)) for _ in range(k)]
        sample = make_multi_sample(items, rng, PLAIN)
        assert sample.corners.shape == (k, 4, 2)
        assert sample.corners.min() >= 0 and sample.corners.max() <= 640
        boxes = np.concatenate([sample.corners.min(axis=1), sample.corners.max(axis=1)], axis=1)  # x0, y0, x1, y1
        for i in range(k):
            for j in range(i + 1, k):
                a, b = boxes[i], boxes[j]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_every_wing_of_a_composition_keeps_its_own_pixels():
    fills = [(30, 30, 30), (30, 30, 230), (30, 230, 30)]
    items = [painted_item(BOX, fill=f, background=BACKGROUND) for f in fills]
    sample = make_multi_sample(items, np.random.default_rng(11), PLAIN)
    for fill in fills:
        painted = painted_box_corners(sample.image, fill)
        assert min(corner_distance(label, painted) for label in sample.corners) < 2.5


def test_five_thousand_compositions_never_overlap_or_leave_the_canvas():
    """The geometry of the full-size case on a 160 px canvas, so that 5,000 samples take seconds."""
    cfg = AugConfig(imgsz=160, photometric=False)
    rng = np.random.default_rng(7)
    box = Obb(cx=50.0, cy=20.0, length=70.0, width=26.0, theta_deg=3.0, eig_ratio=9.0).corners()
    item = Item(np.full((40, 100, 3), BACKGROUND, np.uint8), box, BACKGROUND)
    for _ in range(5000):
        k = choose_k(rng, cfg)
        sample = make_multi_sample([item] * k, rng, cfg)
        assert sample.corners.shape == (k, 4, 2)
        assert sample.corners.min() >= 0 and sample.corners.max() <= cfg.imgsz
        boxes = np.concatenate([sample.corners.min(axis=1), sample.corners.max(axis=1)], axis=1)  # x0, y0, x1, y1
        for i in range(k):
            for j in range(i + 1, k):
                a, b = boxes[i], boxes[j]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def scan_with_white_wedges(label_files):
    """The first synthetic image with pure-white wedges in two corners (as in the real scans), rewritten on disk.
    Returns its path, its label-table row, the image without the wedges and the wedge mask."""
    row = label_files.table.iloc[0].copy()
    path = label_files.raw / row["file"]
    clean = cv2.imread(str(path))
    h, w = clean.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array([[0, 0], [90, 0], [0, 60]]), np.array([[w - 1, h - 1], [w - 80, h - 1], [w - 1, h - 50]])], 255)
    painted = clean.copy()
    painted[mask > 0] = 255
    cv2.imwrite(str(path), painted)
    return path, row, clean, mask > 0


def test_white_spots_of_a_scan_are_filled_with_the_background_colour(label_files):
    """Raw scans carry pure-white wedges at their corners (left over from the scan rotation). Like `pad_image`, `load_item`
    fills them, otherwise every rotated sample shows a sharp white triangle fixed to the wing axis."""
    path, row, clean, wedges = scan_with_white_wedges(label_files)
    item = load_item(path, row)
    assert wedges.sum() > 500 and (item.image[wedges] == item.background).all()
    assert (item.image[~wedges] == clean[~wedges]).all()  # nothing else is touched


def test_a_near_white_background_keeps_its_pixels(label_files):
    """`pad_image` does not fill when the whole background is near white (bright scans): neither does `load_item`."""
    path, row, clean, wedges = scan_with_white_wedges(label_files)
    row[["bg_b", "bg_g", "bg_r"]] = 250
    item = load_item(path, row)
    assert (item.image[wedges] == 255).all()


def test_partners_have_a_similar_tone_and_are_distinct():
    tones = np.array([100, 101, 103, 104, 105, 200, 201, 203, 250, 251], float)  # three clusters
    pool = TonePool(tones, tol=6.0)
    rng = np.random.default_rng(0)
    for first in range(len(tones)):
        near = set(np.flatnonzero(np.abs(tones - tones[first]) <= 6.0).tolist()) - {first}
        for count in (1, 2, 3):
            if len(near) >= count:
                for _ in range(20):
                    partners = pool.partners(first, count, rng).tolist()
                    assert len(partners) == count and len(set(partners)) == count and first not in partners and set(partners) <= near


def test_a_tiny_pool_falls_back_to_repeats():
    pool = TonePool(np.array([10.0, 100.0, 100.5, 200.0]), tol=6.0)
    rng = np.random.default_rng(1)
    assert pool.partners(0, 3, rng).tolist() == [0, 0, 0]  # nobody else is near: the wing itself, repeated
    two = pool.partners(1, 3, rng).tolist()  # the pool holds two images, three partners need repeats
    assert len(two) == 3 and set(two) <= {1, 2}


def test_every_candidate_is_equally_likely():
    pool = TonePool(np.array([100.0, 101.0, 102.0, 103.0, 104.0]), tol=6.0)  # one pool of five
    rng = np.random.default_rng(2)
    pairs = Counter(frozenset(pool.partners(0, 2, rng).tolist()) for _ in range(6000))
    assert len(pairs) == 6  # all C(4, 2) pairs of the other four images occur
    assert max(pairs.values()) - min(pairs.values()) < 0.04 * 6000


def test_without_a_tolerance_every_image_is_a_candidate():
    pool = TonePool(np.array([0.0, 100.0, 200.0, 250.0]), tol=None)
    assert sorted(pool.pool(0).tolist()) == [0, 1, 2, 3]
    assert len(set(pool.partners(0, 3, np.random.default_rng(4)).tolist())) == 3


def test_composition_picks_start_with_the_first_wing_and_follow_the_k_weights():
    pool = TonePool(np.linspace(100.0, 101.0, 50), tol=6.0)
    sizes = np.array([len(composition_picks(pool, np.random.default_rng(s), AugConfig(), first=7)) for s in range(2000)])
    assert set(sizes) == {2, 3, 4} and abs((sizes == 2).mean() - 0.5) < 0.05
    assert all(composition_picks(pool, np.random.default_rng(s), AugConfig(), first=7)[0] == 7 for s in range(20))
    assert len({composition_picks(pool, np.random.default_rng(s), AugConfig())[0] for s in range(200)}) > 30  # without `first` it is drawn uniformly
    assert composition_picks(pool, np.random.default_rng(5), AugConfig()) == composition_picks(pool, np.random.default_rng(5), AugConfig())


def healing_setup(photo_tone=200, canvas_tone=190, line=False, size=240):
    """A flat photo of 60 x 100 px pasted with its top-left corner at (x 60, y 40) on a flat canvas; `line`: a dark vein crosses the top border."""
    image = np.full((60, 100, 3), photo_tone, np.uint8)
    if line:
        image[:, 50:53] = 30
    matrix = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 40.0]])
    coverage = warp_mask(image.shape[:2], matrix, size) > 0
    return np.full((size, size, 3), canvas_tone, np.uint8), image, matrix, coverage, (0, 0, size, size), np.full(3, float(canvas_tone), np.float32)


def test_healing_fades_the_canvas_from_the_border_colour_and_leaves_the_photo_alone():
    canvas, image, matrix, coverage, cell, tone = healing_setup(photo_tone=200, canvas_tone=190)
    heal_edges(canvas, image, matrix, coverage, cell, tone, AugConfig())
    assert (canvas[coverage] == 190).all()  # the function never writes inside the photo (the caller pastes it)
    column = canvas[:40, 100, 0]  # from the top of the canvas down to row 39, the row next to the photo's top border
    assert column[39] >= 199 and column[27] == 190 and column[0] == 190  # starts at the border colour, back at the canvas tone 12 px further out
    assert (np.diff(column[27:40]) >= -1e-4).all()  # a smooth, monotone fade


def test_healing_never_paints_a_glow_or_a_shadow():
    canvas, image, matrix, coverage, cell, tone = healing_setup(photo_tone=250, canvas_tone=150)
    heal_edges(canvas, image, matrix, coverage, cell, tone, AugConfig())
    assert canvas[39, 100, 0] == pytest.approx(162.0, abs=0.5) and canvas.max() <= 162.0 + 1e-4  # limited to the canvas tone + 12
    canvas, image, matrix, coverage, cell, tone = healing_setup(photo_tone=60, canvas_tone=150)
    heal_edges(canvas, image, matrix, coverage, cell, tone, AugConfig())
    assert canvas[39, 100, 0] == pytest.approx(138.0, abs=0.5) and canvas.min() >= 138.0 - 1e-4  # and the canvas tone - 12


def test_healing_ignores_a_vein_that_crosses_the_border():
    canvas, image, matrix, coverage, cell, tone = healing_setup(photo_tone=200, canvas_tone=190, line=True)
    heal_edges(canvas, image, matrix, coverage, cell, tone, AugConfig())
    assert canvas[39, 111, 0] == pytest.approx(canvas[39, 100, 0], abs=0.5)  # the vein is not dragged out of the photo as a streak


def test_healing_stays_inside_the_cell_and_can_be_switched_off():
    canvas, image, matrix, coverage, _, tone = healing_setup()
    heal_edges(canvas, image, matrix, coverage, (0, 0, 120, 240), tone, AugConfig())  # the cell ends at x = 120
    assert (canvas[:, 120:] == 190).all() and (canvas[:, :120][~coverage[:, :120]] != 190).any()
    canvas, image, matrix, coverage, cell, tone = healing_setup()
    heal_edges(canvas, image, matrix, coverage, cell, tone, AugConfig(edge_heal_px=0))
    assert (canvas == 190).all()


def test_edge_healing_only_changes_background_pixels_of_the_canvas():
    items = [painted_item(BOX, fill=(30, 30, 30), background=(150, 150, 150)), painted_item(BOX, fill=(30, 30, 30), background=(158, 158, 158))]
    plain = make_multi_sample(items, np.random.default_rng(3), AugConfig(photometric=False, edge_heal_px=0))
    healed = make_multi_sample(items, np.random.default_rng(3), AugConfig(photometric=False))
    assert np.array_equal(plain.corners, healed.corners)  # the geometry is the same
    changed = (plain.image != healed.image).any(axis=2)
    assert changed.any()
    colours, counts = np.unique(plain.image.reshape(-1, 3), axis=0, return_counts=True)
    assert (plain.image[changed] == colours[counts.argmax()]).all()  # only flat canvas pixels were touched, never a pixel of a photo


def test_edge_healing_is_a_no_op_when_all_photos_have_the_canvas_tone():
    items = [painted_item(BOX, fill=(30, 30, 30)) for _ in range(3)]  # all share BACKGROUND
    a = make_multi_sample(items, np.random.default_rng(4), AugConfig(photometric=False, edge_heal_px=0))
    b = make_multi_sample(items, np.random.default_rng(4), AugConfig(photometric=False))
    assert np.array_equal(a.image, b.image)
