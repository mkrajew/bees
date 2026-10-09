import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG, ops

from wings.detection import obb_dataset
from wings.detection.obb_augment import AugConfig
from wings.detection.obb_dataset import app, build_labels, freeze_split, read_split_map, write_dataset_yaml, write_image_lists
from wings.detection.obb_labels import CORNER_COLUMNS, obb_from_landmarks, points_inside_box


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


def test_the_table_boxes_have_the_extra_room_on_the_lower_side(synthetic_raw):
    table = build(synthetic_raw)
    for _, row in table.iterrows():
        plain = obb_from_landmarks(synthetic_raw.landmarks[row["file"].split("/")[-1]], bottom_extra=0.0)
        assert row["length"] == pytest.approx(plain.length) and row["width"] == pytest.approx(plain.width * 1.1)


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


def test_unchanged_image_lists_are_not_rewritten(synthetic_raw, tmp_path):
    """Jobs on the cluster refresh the lists when they start, and two of them may run at once: a list that is already right must stay untouched."""
    table = build(synthetic_raw)
    write_image_lists(table, synthetic_raw.raw, tmp_path)
    old = 1_000_000_000_000_000_000  # an old modification time in ns, so that a rewrite would be visible
    for name in ("train.txt", "val.txt", "test.txt"):
        os.utime(tmp_path / name, ns=(old, old))
    write_image_lists(table, synthetic_raw.raw, tmp_path)
    assert {(tmp_path / n).stat().st_mtime_ns for n in ("train.txt", "val.txt", "test.txt")} == {old}
    (tmp_path / "train.txt").write_text("stale\n")  # a list from another machine is replaced, and no temporary file is left behind
    write_image_lists(table, synthetic_raw.raw, tmp_path)
    assert len((tmp_path / "train.txt").read_text().split()) == 6
    assert not list(tmp_path.glob("*.tmp"))


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


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_frozen_split_is_complete_valid_and_reproducible(label_files, tmp_path):
    table = label_files.table
    first, second, other = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    assert freeze_split(table, "val", label_files.raw, first, seed=7, n_multi=3) == 2 + 3
    freeze_split(table, "val", label_files.raw, second, seed=7, n_multi=3)
    freeze_split(table, "val", label_files.raw, other, seed=8, n_multi=3)
    assert digest(first) == digest(second)
    assert digest(first) != digest(other)

    images = sorted((first / "val" / "images").glob("*.jpg"))
    labels = sorted((first / "val" / "labels").glob("*.txt"))
    assert [p.stem for p in images] == [p.stem for p in labels] and len(images) == 5
    for image_path, label_path in zip(images, labels):
        assert cv2.imread(str(image_path)).shape == (640, 640, 3)
        for line in label_path.read_text().strip().splitlines():
            values = line.split()
            assert values[0] == "0" and len(values) == 9
            assert all(0.0 <= float(v) <= 1.0 for v in values[1:])
    assert sum(len(p.read_text().strip().splitlines()) for p in labels if p.stem.startswith("multi_")) >= 3 * 2


def test_the_stock_ultralytics_dataset_reads_the_frozen_labels_back(label_files, tmp_path):
    """Stage 2 validates on these folders with the stock dataset, so it must reproduce our corners."""
    freeze_split(label_files.table, "val", label_files.raw, tmp_path, seed=7, n_multi=3)
    dataset = YOLODataset(
        img_path=str(tmp_path / "val" / "images"), imgsz=640, augment=False, hyp=DEFAULT_CFG, rect=False, cache=None,
        data={"names": {0: "wing"}, "channels": 3}, task="obb", batch_size=4, stride=32, pad=0.5, prefix="val: ",
    )
    assert len(dataset) == 5
    for index, image_file in enumerate(dataset.im_files):
        rboxes = dataset[index]["bboxes"].clone()
        rboxes[:, [0, 2]] *= 640
        rboxes[:, [1, 3]] *= 640
        theirs = ops.xywhr2xyxyxyxy(rboxes).numpy().reshape(-1, 2)
        label_file = tmp_path / "val" / "labels" / (Path(image_file).stem + ".txt")
        for line in label_file.read_text().strip().splitlines():
            polygon = np.array(line.split()[1:], dtype=float).reshape(4, 2) * 640
            assert np.linalg.norm(polygon[:, None, :] - theirs[None], axis=-1).min(axis=1).max() < 0.5


def test_dataset_yaml(tmp_path):
    content = yaml.safe_load(write_dataset_yaml(tmp_path).read_text())
    assert content["train"] == "train.txt" and content["val"] == "val/images" and content["test"] == "test/images"
    assert content["nc"] == 1 and content["names"] == {0: "wing"}
    assert "path" not in content  # without it Ultralytics resolves the folders next to the file, on any machine


def test_the_lists_command_rebuilds_the_lists_and_the_yaml_from_labels_csv(label_files):
    """On another machine only labels.csv and the raw images are needed: no old split folders, no image is read."""
    for name in ("train.txt", "val.txt", "test.txt"):
        (label_files.out / name).unlink()
    result = CliRunner().invoke(app, ["lists", "--out", str(label_files.out), "--raw-dir", str(label_files.raw)])
    assert result.exit_code == 0, result.output
    train = (label_files.out / "train.txt").read_text().split()
    assert len(train) == 6 and all(Path(p).is_absolute() and Path(p).exists() for p in train)
    assert (label_files.out / "dataset.yaml").exists()


def test_the_lists_command_fails_with_a_file_name_when_the_raw_folder_is_wrong(label_files, tmp_path):
    result = CliRunner().invoke(app, ["lists", "--out", str(label_files.out), "--raw-dir", str(tmp_path / "elsewhere")])
    assert result.exit_code != 0 and "AA-0000" in str(result.exception)


def test_frozen_compositions_combine_wings_of_similar_tone(label_files, tmp_path, monkeypatch):
    table = label_files.table.copy()
    train = (table["split"] == "train").to_numpy()
    table.loc[train, ["bg_b", "bg_g", "bg_r"]] = np.repeat(np.array([100] * 3 + [200] * 3)[:, None], 3, axis=1)  # two tone clusters far apart
    composed = []
    original = obb_dataset.make_multi_sample

    def spy(items, rng, cfg=AugConfig()):
        composed.append([float(np.mean(item.background)) for item in items])
        return original(items, rng, cfg)

    monkeypatch.setattr(obb_dataset, "make_multi_sample", spy)
    freeze_split(table, "train", label_files.raw, tmp_path, seed=7, n_multi=40)
    assert len(composed) == 40 and all(max(tones) - min(tones) <= 6.0 for tones in composed)
    assert {round(t) for tones in composed for t in tones} == {100, 200}  # both clusters were composed, so the check is not vacuous
