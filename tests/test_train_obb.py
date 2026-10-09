import os
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from obb_synthetic import build_synthetic_raw
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import SettingsManager

from wings.config import PROJ_ROOT
from wings.detection import train_obb
from wings.detection.obb_augment import AugConfig, WingOBBDataset
from wings.detection.obb_dataset import SPLITS, build_labels, freeze_split, write_dataset_yaml, write_image_lists
from wings.detection.train_obb import (
    ObbPaths,
    ObbRunConfig,
    apply_overrides,
    aug_config_from,
    config_from_dict,
    evaluate,
    load_config,
    make_trainer,
    resolve_workers,
    train,
    ultralytics_args,
    wandb_integration,
)


# ---------------------------------------------------------------- configuration


def write_config(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_a_config_file_is_loaded_with_explicit_defaults(tmp_path):
    cfg = load_config(write_config(tmp_path / "c.yaml", "name: x\ninit: models/a.pt\nepochs: 3\naugment:\n  p_multi: 0.2\n  fraction_range: [0.3, 0.8]\n"))
    assert (cfg.name, cfg.init, cfg.epochs) == ("x", "models/a.pt", 3)
    assert cfg.augment == {"p_multi": 0.2, "fraction_range": [0.3, 0.8]}
    assert cfg.optimizer == "MuSGD"  # never 'auto': it silently ignores lr0


def test_an_unknown_key_is_rejected_with_its_name(tmp_path):
    with pytest.raises(ValueError, match="epochz"):
        load_config(write_config(tmp_path / "c.yaml", "name: x\nepochz: 3\n"))


def test_a_run_needs_a_name():
    with pytest.raises(ValueError, match="name"):
        config_from_dict({"epochs": 3})


def test_augmentation_overrides_become_an_augconfig():
    aug = aug_config_from({"p_multi": 0.2, "fraction_range": [0.3, 0.8]}, imgsz=640)
    assert aug == AugConfig(p_multi=0.2, fraction_range=(0.3, 0.8), imgsz=640)  # YAML lists become tuples
    assert aug_config_from({}, imgsz=320).imgsz == 320  # the canvas follows the training size
    with pytest.raises(ValueError, match="p_mutli"):
        aug_config_from({"p_mutli": 0.2}, imgsz=640)
    with pytest.raises(ValueError, match="imgsz"):
        aug_config_from({"imgsz": 320}, imgsz=640)


def test_command_line_overrides_set_top_level_and_augmentation_values():
    data = apply_overrides({"name": "x", "epochs": 100, "augment": {"p_multi": 0.4}}, ["epochs=2", "augment.p_multi=0.1", "augment.brightness_range=[0.6, 1.2]", "wandb=false", "init=null"])
    assert data == {"name": "x", "epochs": 2, "augment": {"p_multi": 0.1, "brightness_range": [0.6, 1.2]}, "wandb": False, "init": None}
    with pytest.raises(ValueError, match="epochs"):
        apply_overrides({"name": "x"}, ["epochs"])


def test_workers_default_to_the_cpus_of_the_slurm_task(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert resolve_workers(None) == 16 and resolve_workers(4) == 4
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert resolve_workers(None) == (os.cpu_count() or 1)


@pytest.mark.parametrize("stored, wanted", [(True, False), (False, True)])
def test_the_config_decides_whether_ultralytics_logs_to_wandb(tmp_path, monkeypatch, stored, wanted):
    """Ultralytics' own W&B callback follows a per-user setting. Where that setting was on, `wandb: false` still started the callback,
    which crashed on the absolute `project` path (found by the first run on real data); the file of the user must not be rewritten either."""
    file = tmp_path / "settings.json"
    settings = SettingsManager(file)
    settings["wandb"] = stored
    monkeypatch.setattr(train_obb, "SETTINGS", settings)
    stored_text = file.read_text(encoding="utf-8")
    with wandb_integration(wanted):
        assert settings["wandb"] is wanted
    assert settings["wandb"] is stored
    assert file.read_text(encoding="utf-8") == stored_text


def test_paths_follow_the_config_and_relative_ones_start_at_the_project_root():
    paths = ObbPaths.from_config(ObbRunConfig(name="r1", init="models/obb/yolo26n-obb.pt"))
    assert paths.init == PROJ_ROOT / "models" / "obb" / "yolo26n-obb.pt"
    assert paths.save_dir.name == "r1" and paths.save_dir.parent.name == "obb"
    assert paths.dataset_yaml.name == "dataset.yaml" and paths.labels_csv.name == "labels.csv"
    assert ObbPaths.from_config(ObbRunConfig(name="r2")).init is None
    assert ObbPaths.from_config(ObbRunConfig(name="r3", data_dir="/somewhere/obb")).labels_csv.as_posix().endswith("/somewhere/obb/labels.csv")


def test_the_arguments_for_ultralytics_switch_its_own_augmentation_off_and_name_the_optimizer():
    cfg = ObbRunConfig(name="r1", init="models/obb/yolo26n-obb.pt", epochs=7, close_mosaic=3, workers=5, device=0)
    paths = ObbPaths.from_config(cfg)
    args = ultralytics_args(cfg, paths)
    assert args["data"] == str(paths.dataset_yaml) and args["name"] == "r1" and args["project"] == str(paths.runs_dir) and args["exist_ok"] is True
    assert args["pretrained"] == str(paths.init) and ultralytics_args(ObbRunConfig(name="r"), ObbPaths.from_config(ObbRunConfig(name="r")))["pretrained"] is False
    assert (args["epochs"], args["close_mosaic"], args["workers"], args["optimizer"], args["lr0"]) == (7, 3, 5, "MuSGD", 0.01)
    assert all(args[k] == 0 for k in ("mosaic", "mixup", "cutmix", "copy_paste", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "erasing", "multi_scale"))


def test_the_trainer_class_carries_the_augmentation_and_the_label_table(tmp_path):
    aug = AugConfig(p_multi=0.2, imgsz=64)
    trainer_cls = make_trainer(aug, tmp_path / "labels.csv", tmp_path / "raw")
    assert issubclass(trainer_cls, OBBTrainer)
    assert (trainer_cls.aug_config, trainer_cls.labels_path, trainer_cls.raw_dir) == (aug, tmp_path / "labels.csv", tmp_path / "raw")


# ---------------------------------------------------------------- a real (tiny, CPU) training run on synthetic wings


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """Synthetic raw wings, their label table, image lists and frozen val/test sets, in the layout the training code expects."""
    root = tmp_path_factory.mktemp("obb-smoke")
    raw = build_synthetic_raw(root)
    data_dir = root / "detection-obb"
    data_dir.mkdir()
    table = build_labels(raw.raw, raw.countries, raw.split_dir, raw.mean_shape)
    table.to_csv(data_dir / "labels.csv", index=False)
    write_image_lists(table, raw.raw, data_dir)
    for split in ("val", "test"):
        freeze_split(table, split, raw.raw, data_dir, seed=7, n_multi=2, cfg=AugConfig(imgsz=640))
    write_dataset_yaml(data_dir)
    return SimpleNamespace(root=root, raw=raw.raw, data_dir=data_dir, runs=root / "runs")


def smoke_config(smoke, name, **changes):
    values = dict(
        name=name, init=None, epochs=2, patience=5, batch=2, workers=0, imgsz=64, device="cpu", amp=False, plots=False, wandb=False,
        close_mosaic=1, seed=1, data_dir=str(smoke.data_dir), raw_dir=str(smoke.raw), project=str(smoke.runs),
    )
    values.update(changes)
    return ObbRunConfig(**values)


@pytest.fixture(scope="module")
def epoch_log():
    return []


@pytest.fixture(scope="module")
def trained(smoke, epoch_log):
    def note(trainer):
        dataset = trainer.train_loader.dataset
        epoch_log.append((trainer.epoch, type(dataset).__name__, dataset.cfg.p_multi))

    return train(smoke_config(smoke, "first"), callbacks={"on_train_epoch_end": note})


def test_a_run_trains_validates_and_leaves_its_checkpoints_and_config(trained):
    assert (trained / "weights" / "last.pt").exists() and (trained / "weights" / "best.pt").exists()
    results = pd.read_csv(trained / "results.csv")
    assert len(results) == 2 and "metrics/mAP50-95(B)" in {c.strip() for c in results.columns}
    saved = yaml.safe_load((trained / "obb_config.yaml").read_text())
    assert saved["name"] == "first" and saved["close_mosaic"] == 1  # the exact configuration is kept next to the weights


def test_the_real_training_loop_uses_our_dataset_and_ends_without_compositions(trained, epoch_log):
    """close_mosaic=1 of a 2-epoch run: the first epoch has compositions (p_multi 0.4), the last one single wings only.
    (Observed at the end of each epoch: the trainer closes the mosaic right after the start-of-epoch callbacks.)"""
    assert epoch_log == [(0, "WingOBBDataset", 0.4), (1, "WingOBBDataset", 0.0)]


def test_the_training_dataset_is_ours_and_the_validation_one_is_the_frozen_set(smoke):
    trainer_cls = make_trainer(AugConfig(imgsz=64), smoke.data_dir / "labels.csv", smoke.raw)
    trainer = trainer_cls(overrides={**{k: v for k, v in ultralytics_args(smoke_config(smoke, "probe"), ObbPaths.from_config(smoke_config(smoke, "probe"))).items() if k != "pretrained"}, "model": "yolo26n-obb.yaml"})
    trainer.setup_model()  # the stock validation dataset takes its stride from the model
    assert isinstance(trainer.build_dataset(str(smoke.data_dir / "train.txt"), mode="train", batch=2), WingOBBDataset)
    validation = trainer.build_dataset(str(smoke.data_dir / "val" / "images"), mode="val", batch=2)
    assert not isinstance(validation, WingOBBDataset) and len(validation) == 2 + 2  # the frozen val: one image per wing + two compositions


def test_a_trained_checkpoint_is_evaluated_on_the_frozen_val_and_test_sets(smoke, trained):
    for split in ("val", "test"):
        metrics = evaluate(trained / "weights" / "best.pt", split=split, data_dir=smoke.data_dir, imgsz=64, batch=2, device="cpu", project=smoke.root / "eval", workers=0)
        assert set(metrics) == {"map50-95", "map50", "precision", "recall"} and all(isinstance(v, float) for v in metrics.values())
    assert sorted(p.name for p in (smoke.root / "eval").iterdir()) == ["first-test", "first-val"]  # kept where asked, not in the real runs folder


def test_the_command_line_evaluates_a_checkpoint_and_prints_the_metrics_as_json(smoke, trained):
    import json

    from typer.testing import CliRunner

    from wings.detection.train_obb import app

    arguments = ["eval", str(trained / "weights" / "best.pt"), "--data-dir", str(smoke.data_dir), "--imgsz", "64", "--batch", "2", "--device", "cpu", "--workers", "0"]
    result = CliRunner().invoke(app, arguments + ["--project", str(smoke.root / "cli-eval")])
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.output[result.output.rindex("{\n") :])) == {"map50-95", "map50", "precision", "recall"}
    assert (smoke.root / "cli-eval" / "first-val").is_dir()


def test_an_interrupted_run_is_resumed_from_its_last_checkpoint(smoke):
    def interrupt_after_the_first_epoch(trainer):
        raise RuntimeError("interrupted")

    cfg = smoke_config(smoke, "interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        train(cfg, callbacks={"on_model_save": interrupt_after_the_first_epoch})
    save_dir = ObbPaths.from_config(cfg).save_dir
    assert (save_dir / "weights" / "last.pt").exists() and len(pd.read_csv(save_dir / "results.csv")) == 1
    train(cfg, resume=True)
    assert len(pd.read_csv(save_dir / "results.csv")) == 2


def test_a_finished_run_is_not_overwritten_by_a_new_one_with_the_same_name(smoke, trained):
    with pytest.raises(FileExistsError, match="first"):
        train(smoke_config(smoke, "first"))


def test_missing_inputs_are_reported_by_name(smoke, tmp_path):
    with pytest.raises(FileNotFoundError, match="no-such-weights"):
        train(smoke_config(smoke, "x", init=str(tmp_path / "no-such-weights.pt")))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="labels.csv"):
        train(smoke_config(smoke, "y", data_dir=str(empty)))


def test_the_command_line_trains_a_run_from_a_yaml_file_with_overrides(smoke, tmp_path):
    from typer.testing import CliRunner

    from wings.detection.train_obb import app

    config = tmp_path / "cli.yaml"
    config.write_text(yaml.safe_dump({"name": "from-yaml", "model": "yolo26n-obb.yaml", "batch": 2, "imgsz": 64, "device": "cpu", "amp": False, "plots": False}), encoding="utf-8")
    arguments = ["train", str(config), "--set", "epochs=1", "--set", "close_mosaic=0", "--set", "workers=0", "--set", "wandb=false", "--set", f"data_dir={smoke.data_dir}"]
    result = CliRunner().invoke(app, arguments + ["--set", f"raw_dir={smoke.raw}", "--set", f"project={smoke.runs}", "--set", "augment.p_multi=0.2"])
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load((smoke.runs / "from-yaml" / "obb_config.yaml").read_text())
    assert saved["epochs"] == 1 and saved["augment"] == {"p_multi": 0.2} and len(pd.read_csv(smoke.runs / "from-yaml" / "results.csv")) == 1
