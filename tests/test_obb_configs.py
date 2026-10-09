"""The run configurations and their SLURM jobs must stay consistent (a typo would otherwise show up only on the cluster)."""

import re

import pytest

from wings.config import PROJ_ROOT
from wings.detection.train_obb import aug_config_from, load_config

CONFIGS = sorted((PROJ_ROOT / "configs" / "obb").glob("*.yaml"))


def test_there_are_run_configurations():
    assert {p.stem for p in CONFIGS} >= {"pilot-coco", "pilot-wings", "pilot-dota"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_a_configuration_is_valid_and_named_like_its_file(path):
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert cfg.optimizer != "auto"  # 'auto' silently ignores lr0 and momentum
    aug_config_from(cfg.augment, cfg.imgsz)  # raises on a misspelled augmentation key
    assert cfg.init is None or cfg.init.startswith("models/")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_configuration_has_a_job_that_starts_it_safely(path):
    job = PROJ_ROOT / "jobs" / "obb" / f"obb_{path.stem.replace('-', '_')}.sh"
    text = job.read_text(encoding="utf-8")
    assert f"configs/obb/{path.name}" in text
    assert "--gres=gpu:geforce_rtx_4090:1" in text  # a generic GPU can land on the Blackwell node, where torch has no kernels
    assert re.search(r"^uv sync\s*$", text, re.MULTILINE) and "--reinstall" not in text.replace("never `--reinstall`", "")
    assert "--cpus-per-task=8" in text and "\r" not in text  # two jobs share the 16 CPUs of the node; Windows line endings would break bash
