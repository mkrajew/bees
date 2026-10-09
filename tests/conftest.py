from types import SimpleNamespace

import numpy as np
import pytest
from obb_synthetic import build_synthetic_raw, make_landmarks, make_wing_image


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def landmarks(rng):
    return make_landmarks(rng, angle_deg=7.0)


@pytest.fixture
def wing_image():
    return make_wing_image()


@pytest.fixture
def synthetic_raw(tmp_path):
    return build_synthetic_raw(tmp_path)


@pytest.fixture
def label_files(synthetic_raw, tmp_path):
    """labels.csv and the {split}.txt image lists built from the synthetic raw data."""
    from wings.detection.obb_dataset import build_labels, write_image_lists

    out = tmp_path / "detection-obb"
    out.mkdir()
    table = build_labels(synthetic_raw.raw, synthetic_raw.countries, synthetic_raw.split_dir, synthetic_raw.mean_shape)
    table.to_csv(out / "labels.csv", index=False)
    write_image_lists(table, synthetic_raw.raw, out)
    return SimpleNamespace(out=out, table=table, raw=synthetic_raw.raw)
