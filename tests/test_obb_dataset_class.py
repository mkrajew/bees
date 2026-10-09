import copy
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from ultralytics.utils import DEFAULT_CFG, ops

from wings.detection.obb_augment import AugConfig, WingOBBDataset


def make_dataset(files, **cfg_kwargs):
    cfg_kwargs.setdefault("p_multi", 0.0)
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


def test_boxes_given_to_the_loss_match_our_corners_for_compositions(label_files):
    assert 2 <= assert_loss_boxes_match(make_dataset(label_files, p_multi=1.0)) <= 4


def test_the_default_mix_contains_single_wings_and_compositions(label_files):
    dataset = make_dataset(label_files, p_multi=0.4)
    counts = [len(dataset.label_for(i % len(dataset), seed=i)["instances"].segments) for i in range(200)]
    assert min(counts) == 1 and max(counts) >= 2
    assert 0.2 < np.mean(np.array(counts) > 1) < 0.6


def test_a_one_image_split_can_still_make_compositions(label_files, tmp_path):
    table = pd.read_csv(label_files.out / "labels.csv")
    only = table[table["split"] == "train"].iloc[:1]
    small = tmp_path / "small"
    small.mkdir()
    only.to_csv(small / "labels.csv", index=False)
    (small / "train.txt").write_text(str((label_files.raw / only.iloc[0]["file"]).resolve()) + "\n")
    dataset = WingOBBDataset(small / "labels.csv", "train", small / "train.txt", raw_dir=label_files.raw, cfg=AugConfig(p_multi=1.0))
    assert len(dataset) == 1 and len(dataset.label_for(0, 3)["instances"].segments) >= 2


def link_directory(link, target):
    """`link` -> `target` as a symbolic link, or as a junction on Windows without the right to create symbolic links."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt" or subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True).returncode != 0:
            pytest.skip("cannot create a directory link here")


def test_a_linked_raw_folder_still_matches_the_image_list(label_files, tmp_path):
    """On shared clusters `data/` is often a link. The image lists hold resolved paths (`write_image_lists`), so the
    dataset must compare resolved paths too, instead of failing with 'missing from the labels table'."""
    link = tmp_path / "linked-raw"
    link_directory(link, label_files.raw)
    dataset = WingOBBDataset(label_files.out / "labels.csv", "train", label_files.out / "train.txt", raw_dir=link, cfg=AugConfig(p_multi=0.0))
    assert len(dataset) == 6 and dataset.label_for(0, 1)["img"].shape == (640, 640, 3)


def test_compositions_combine_wings_of_similar_tone(label_files):
    table = pd.read_csv(label_files.out / "labels.csv")
    train = (table["split"] == "train").to_numpy()
    table.loc[train, ["bg_b", "bg_g", "bg_r"]] = np.repeat(np.array([100] * 3 + [200] * 3)[:, None], 3, axis=1)  # two tone clusters far apart
    table.to_csv(label_files.out / "labels.csv", index=False)
    dataset = make_dataset(label_files, p_multi=1.0)
    loaded = []
    original = dataset.load_item
    dataset.load_item = lambda index: (loaded.append(index), original(index))[1]
    tone_of = lambda index: float(dataset.table.iloc[dataset._rows[index]][["bg_b", "bg_g", "bg_r"]].mean())
    seen = set()
    for seed in range(60):
        loaded.clear()
        dataset.label_for(seed % len(dataset), seed)
        tones = [tone_of(i) for i in loaded]
        assert len(loaded) >= 2 and max(tones) - min(tones) <= 6.0
        seen.update(round(t) for t in tones)
    assert seen == {100, 200}  # both clusters were composed, so the check is not vacuous


def test_close_mosaic_turns_the_compositions_off(label_files):
    """The trainer calls close_mosaic for the last epochs: from then on only single wings, the case that matters in production."""
    dataset = make_dataset(label_files, p_multi=1.0)
    assert max(len(dataset.label_for(i % len(dataset), seed=i)["instances"].segments) for i in range(20)) >= 2
    dataset.close_mosaic(copy.copy(DEFAULT_CFG))
    assert dataset.cfg.p_multi == 0.0
    assert {len(dataset.label_for(i % len(dataset), seed=i)["instances"].segments) for i in range(30)} == {1}


def test_fraction_keeps_only_the_first_images(label_files):
    dataset = WingOBBDataset(label_files.out / "labels.csv", "train", label_files.out / "train.txt", raw_dir=label_files.raw, cfg=AugConfig(p_multi=0.0), fraction=0.5)
    assert len(dataset) == 3 and len(dataset.labels) == 3 and len(dataset.tone_pool) == 3
    assert dataset.label_for(2, seed=1)["img"].shape == (640, 640, 3)
