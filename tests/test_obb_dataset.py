import cv2
import numpy as np
import pandas as pd
import pytest

from wings.detection.obb_dataset import build_labels, read_split_map, write_image_lists
from wings.detection.obb_labels import CORNER_COLUMNS, points_inside_box


def build(raw):
    return build_labels(raw.raw, raw.countries, raw.split_dir, raw.mean_shape)


def test_one_row_per_image_with_the_expected_columns_and_split(synthetic_raw):
    table = build(synthetic_raw)
    assert len(table) == 10
    assert table["split"].value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}
    expected = {"file", "country", "split", "img_w", "img_h", *CORNER_COLUMNS, "cx", "cy", "length", "width", "theta_deg", "eig_ratio", "dir_sign", "bg_b", "bg_g", "bg_r"}
    assert expected <= set(table.columns)


def test_every_landmark_lies_inside_its_box(synthetic_raw):
    table = build(synthetic_raw)
    for _, row in table.iterrows():
        corners = row[CORNER_COLUMNS].to_numpy(np.float64).reshape(4, 2)
        assert points_inside_box(corners, synthetic_raw.landmarks[row["file"].split("/")[-1]], tol=1e-6)


def test_size_and_background_columns_match_the_images(synthetic_raw):
    table = build(synthetic_raw)
    for _, row in table.iterrows():
        image = cv2.imread(str(synthetic_raw.raw / row["file"]))
        assert (row["img_h"], row["img_w"]) == image.shape[:2]
        assert (row["bg_b"], row["bg_g"], row["bg_r"]) == synthetic_raw.backgrounds[row["file"].split("/")[-1]]


def test_direction_sign_is_consistent_for_upright_wings(synthetic_raw):
    assert set(build(synthetic_raw)["dir_sign"]) == {1}


def test_image_lists_hold_absolute_paths_of_each_split(synthetic_raw, tmp_path):
    table = build(synthetic_raw)
    write_image_lists(table, synthetic_raw.raw, tmp_path)
    train = (tmp_path / "train.txt").read_text().split()
    assert len(train) == 6 and all(p.endswith(".png") for p in train)
    assert all(pd.io.common.file_exists(p) for p in train)


def test_an_unreadable_image_fails_with_its_name(synthetic_raw):
    (synthetic_raw.raw / "AA-wing-images" / "AA-0002.png").unlink()
    with pytest.raises(FileNotFoundError, match="AA-0002"):
        build(synthetic_raw)


def test_an_image_missing_from_every_split_fails(synthetic_raw):
    (synthetic_raw.split_dir / "images" / "train" / "AA-0001.png").unlink()
    with pytest.raises(ValueError, match="AA-0001"):
        build(synthetic_raw)


def test_an_image_in_two_splits_fails(synthetic_raw):
    (synthetic_raw.split_dir / "images" / "val" / "AA-0001.png").touch()
    with pytest.raises(ValueError, match="AA-0001"):
        read_split_map(synthetic_raw.split_dir)


def test_degenerate_landmarks_fail_with_the_file_name(synthetic_raw):
    path = synthetic_raw.raw / "BB-raw-coordinates.csv"
    table = pd.read_csv(path)
    table.iloc[1, 1:] = 50.0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="BB-0001"):
        build(synthetic_raw)


def test_non_numeric_landmarks_fail_with_the_file_name(synthetic_raw):
    path = synthetic_raw.raw / "BB-raw-coordinates.csv"
    table = pd.read_csv(path).astype(object)
    table.iloc[0, 3] = "n/a"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="BB-0000"):
        build(synthetic_raw)
