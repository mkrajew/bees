import numpy as np
import pytest
from obb_synthetic import make_landmarks
from ultralytics.utils import ops

from wings.detection.obb_labels import (
    Obb,
    background_color,
    direction_sign,
    obb_from_landmarks,
    obb_to_xywhr,
    points_inside_box,
    principal_axis,
    reference_landmarks,
)


def test_box_contains_every_landmark(landmarks):
    obb = obb_from_landmarks(landmarks)
    assert points_inside_box(obb.corners(), landmarks)


def test_points_inside_box_notices_a_point_outside(landmarks):
    corners = obb_from_landmarks(landmarks).corners()
    outside = np.vstack([landmarks, corners[0] - 5.0 * (corners[1] - corners[0]) / np.linalg.norm(corners[1] - corners[0])])
    assert not points_inside_box(corners, outside)
    assert points_inside_box(corners, outside, tol=6.0)


def test_margins_scale_the_extents(landmarks):
    tight = obb_from_landmarks(landmarks, 1.0, 1.0)
    obb = obb_from_landmarks(landmarks, 1.2, 1.4)
    assert obb.length == pytest.approx(tight.length * 1.2)
    assert obb.width == pytest.approx(tight.width * 1.4)
    assert (obb.cx, obb.cy) == pytest.approx((tight.cx, tight.cy))


@pytest.mark.parametrize("alpha", [-120.0, -35.0, 12.0, 80.0, 171.0])
def test_rotating_the_landmarks_rotates_the_axis(rng, alpha):
    base = make_landmarks(rng, angle_deg=5.0)
    centre = base.mean(axis=0)
    a = np.deg2rad(alpha)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = (base - centre) @ rot.T + centre
    t0, t1 = obb_from_landmarks(base).theta_deg, obb_from_landmarks(rotated).theta_deg
    diff = (t1 - t0 - alpha + 90.0) % 180.0 - 90.0  # equal modulo 180 degrees
    assert abs(diff) < 1e-6


def test_point_order_does_not_matter(landmarks, rng):
    a = obb_from_landmarks(landmarks)
    b = obb_from_landmarks(landmarks[rng.permutation(len(landmarks))])
    assert (a.cx, a.cy, a.length, a.width, a.theta_deg) == pytest.approx((b.cx, b.cy, b.length, b.width, b.theta_deg))


def test_axis_sign_convention(rng):
    for angle in (-80.0, -10.0, 10.0, 80.0, 100.0, 170.0):
        axis, _ = principal_axis(make_landmarks(rng, angle_deg=angle))
        assert axis[0] > 0 or (abs(axis[0]) < 1e-9 and axis[1] > 0)


@pytest.mark.parametrize("angle", [0.0, 7.0, 40.0, 89.0, 100.0, 133.0, 175.0])
def test_xywhr_agrees_with_ultralytics(rng, angle):
    obb = obb_from_landmarks(make_landmarks(rng, angle_deg=angle))
    ours = np.array(obb_to_xywhr(obb))
    theirs = ops.xyxyxyxy2xywhr(obb.corners().astype(np.float32).reshape(1, 8))[0]
    assert ours[:4] == pytest.approx(theirs[:4], abs=1e-2)
    assert abs((ours[4] - theirs[4] + np.pi / 2) % np.pi - np.pi / 2) < 1e-3
    assert -np.pi / 4 <= ours[4] < 3 * np.pi / 4


def test_xywhr_swaps_sides_when_the_width_is_the_longer_one():
    obb = Obb(10.0, 20.0, 30.0, 80.0, 10.0, 5.0)
    cx, cy, w, h, r = obb_to_xywhr(obb)
    assert (w, h) == (80.0, 30.0)
    assert r == pytest.approx(np.deg2rad(100.0))


def test_direction_sign_flips_when_the_wing_is_turned_by_180(rng):
    mean_shape = make_landmarks(rng, centre=(0.0, 0.0), angle_deg=0.0)
    i_lo, i_hi = reference_landmarks(mean_shape)
    assert {i_lo, i_hi} == {0, 1}  # the two points placed at the extreme ends in make_landmarks
    wing = make_landmarks(np.random.default_rng(5), angle_deg=15.0)
    turned = 2 * wing.mean(axis=0) - wing  # rotation by 180 degrees about the centroid
    sign = direction_sign(wing, principal_axis(wing)[0], i_lo, i_hi)
    assert sign == -direction_sign(turned, principal_axis(turned)[0], i_lo, i_hi)
    assert sign in (-1, 1)


@pytest.mark.parametrize(
    "points",
    [np.zeros((19, 2)), np.column_stack([np.arange(19.0), 2 * np.arange(19.0)]), np.full((19, 2), np.nan), np.zeros((2, 2))],
    ids=["identical", "collinear", "nan", "too_few"],
)
def test_degenerate_landmarks_are_rejected(points):
    with pytest.raises(ValueError):
        obb_from_landmarks(points)


def test_background_colour_of_a_flat_image(wing_image):
    assert background_color(wing_image) == (170, 180, 160)


def test_background_colour_ignores_a_pure_white_frame_line():
    img = np.full((120, 300, 3), (120, 130, 140), np.uint8)
    img[5, :] = 255  # white row exactly where the frame is sampled
    img[-6, :] = 255
    img[:, 5] = 255
    img[:, -6] = 255
    assert background_color(img) != (255, 255, 255)


def test_background_colour_of_a_tiny_image_does_not_crash():
    assert background_color(np.full((8, 12, 3), 77, np.uint8)) == (77, 77, 77)
