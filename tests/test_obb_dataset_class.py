import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from ultralytics.utils import DEFAULT_CFG, ops

from wings.detection.obb_augment import AugConfig, WingOBBDataset


def make_dataset(files, **cfg_kwargs):
    return WingOBBDataset(files.out / "labels.csv", "train", files.out / "train.txt", raw_dir=files.raw, cfg=AugConfig(**cfg_kwargs))


def vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def assert_loss_boxes_match(dataset) -> int:
    """For every image: the boxes `Format` hands to the loss, turned back into corners, equal our own corners.
    Returns the largest number of boxes seen in one sample."""
    most = 0
    for index in range(len(dataset)):
        label = dataset.label_for(index, seed=index)
        ours = label["instances"].segments.copy()
        out = dataset.transforms(label)
        rboxes = out["bboxes"].clone()
        rboxes[:, [0, 2]] *= 640
        rboxes[:, [1, 3]] *= 640
        theirs = ops.xywhr2xyxyxyxy(rboxes).numpy()
        assert len(theirs) == len(ours)
        for polygon in ours:
            assert min(vertex_distance(polygon, other) for other in theirs) < 0.5
        most = max(most, len(ours))
    return most


def test_dataset_covers_the_requested_split(label_files):
    assert len(make_dataset(label_files)) == 6


def test_item_has_the_keys_and_shapes_ultralytics_expects(label_files):
    item = make_dataset(label_files)[0]
    assert item["img"].shape == (3, 640, 640) and item["img"].dtype == torch.uint8
    assert item["bboxes"].shape == (1, 5) and item["cls"].shape == (1, 1) and item["batch_idx"].shape == (1,)
    assert 0.0 < float(item["bboxes"][:, :4].max()) <= 1.0
    assert {"im_file", "ori_shape", "resized_shape", "ratio_pad"} <= set(item)


def test_collated_batch_is_a_stock_obb_batch(label_files):
    dataset = make_dataset(label_files)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=WingOBBDataset.collate_fn)))
    assert batch["img"].shape == (4, 3, 640, 640)
    assert batch["bboxes"].shape == (4, 5) and batch["cls"].shape == (4, 1)
    assert batch["batch_idx"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_boxes_given_to_the_loss_match_our_corners(label_files):
    assert assert_loss_boxes_match(make_dataset(label_files)) == 1


def test_label_for_is_deterministic(label_files):
    dataset = make_dataset(label_files)
    a, b, c = dataset.label_for(0, 5), dataset.label_for(0, 5), dataset.label_for(0, 6)
    assert (a["img"] == b["img"]).all() and np.array_equal(a["instances"].segments, b["instances"].segments)
    assert not np.array_equal(a["instances"].segments, c["instances"].segments)


def test_items_follow_the_torch_seed_and_differ_between_calls(label_files):
    dataset = make_dataset(label_files)
    torch.manual_seed(0)
    first = dataset[0]["bboxes"]
    second = dataset[0]["bboxes"]
    torch.manual_seed(0)
    again = dataset[0]["bboxes"]
    assert torch.equal(first, again) and not torch.equal(first, second)


def test_a_size_mismatch_between_table_and_image_fails_loudly(label_files):
    table = pd.read_csv(label_files.out / "labels.csv")
    table.loc[table["split"] == "train", "img_w"] += 10
    table.to_csv(label_files.out / "labels.csv", index=False)
    with pytest.raises(ValueError, match="labels table says"):
        make_dataset(label_files).label_for(0, 1)


def test_an_image_missing_from_the_table_is_rejected(label_files):
    table = pd.read_csv(label_files.out / "labels.csv")
    table[table["file"] != table.loc[table["split"] == "train", "file"].iloc[0]].to_csv(label_files.out / "labels.csv", index=False)
    with pytest.raises(ValueError, match="missing from the labels table"):
        make_dataset(label_files)


def test_close_mosaic_keeps_working(label_files):
    dataset = make_dataset(label_files)
    dataset.close_mosaic(copy.copy(DEFAULT_CFG))
    assert dataset[0]["img"].shape == (3, 640, 640)
