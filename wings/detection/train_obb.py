"""Train and evaluate the oriented-box (OBB) wing detector (stage 2).

    uv run python -m wings.detection.train_obb train configs/obb/pilot-dota.yaml
    uv run python -m wings.detection.train_obb train configs/obb/pilot-dota.yaml --resume      # continue an interrupted run
    uv run python -m wings.detection.train_obb train configs/obb/pilot-dota.yaml --set epochs=2 --set fraction=0.02 --set wandb=false --set name=smoke
    uv run python -m wings.detection.train_obb eval wings/detection/runs/obb/pilot-dota/weights/best.pt --split test

A run is described by one YAML file (`configs/obb/*.yaml`, see `ObbRunConfig`); nothing else is needed to repeat it. The training set is
`WingOBBDataset` (online augmentation, `wings.detection.obb_augment`), the validation set the frozen val set of `obb_dataset freeze`.
`jobs/obb/README.md` explains how the runs are organised and how they are started on the cluster.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer
import yaml
from loguru import logger
from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import SETTINGS

from wings.config import PROJ_ROOT, RAW_DATA_DIR
from wings.detection.obb_augment import AugConfig, WingOBBDataset
from wings.detection.obb_dataset import DEFAULT_OUT_DIR, write_image_lists

RUNS_DIR = PROJ_ROOT / "wings" / "detection" / "runs" / "obb"  # inside the already git-ignored `wings/detection/runs/`

app = typer.Typer(help="Train and evaluate the oriented-box (OBB) wing detector.")


@dataclass(frozen=True)
class ObbRunConfig:
    """Everything that defines one training run. Relative paths start at the project root."""

    name: str  # folder name under runs/obb and name of the W&B run
    init: str | None = None  # .pt checkpoint to start from (a detect or an OBB model; matching layers are transferred), None: random weights
    model: str = "yolo26n-obb.yaml"  # architecture (n: the size that runs in the browser)
    epochs: int = 100
    patience: int = 30  # early stopping: epochs without improvement of the validation fitness
    batch: int = 32
    workers: int | None = None  # None: the CPUs of the SLURM task (the data pipeline is what limits the speed), else all CPUs
    imgsz: int = 640  # the training canvas of WingOBBDataset and the size the model is trained and validated at
    optimizer: str = "MuSGD"  # explicit on purpose: 'auto' ignores lr0 and momentum
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    cos_lr: bool = False
    close_mosaic: int = 10  # the last N epochs are trained without compositions (single wings only)
    seed: int = 42
    amp: bool = True
    deterministic: bool = True
    fraction: float = 1.0  # share of the training images, for smoke tests
    device: int | str = 0
    time: float | None = None  # hard limit in hours (Ultralytics shortens the schedule to fit)
    save_period: int = -1  # also keep a checkpoint every N epochs (-1: only last.pt and best.pt)
    plots: bool = True
    data_dir: str | None = None  # folder with dataset.yaml, labels.csv and the frozen sets (default: data/processed/detection-obb)
    raw_dir: str | None = None  # folder with the raw wing images (default: data/raw)
    project: str | None = None  # folder that holds the runs (default: wings/detection/runs/obb)
    wandb: bool = True
    wandb_entity: str = "furkot-team"
    wandb_project: str = "wings-detection-obb"
    wandb_group: str | None = None
    augment: dict[str, Any] = field(default_factory=dict)  # overrides of AugConfig (p_multi, fraction_range, brightness_range, ...)


def config_from_dict(data: dict[str, Any]) -> ObbRunConfig:
    known = {f.name for f in fields(ObbRunConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in the run configuration: {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    if "name" not in data:
        raise ValueError("the run configuration needs a 'name'")
    return ObbRunConfig(**data)


def load_config(path: Path | str) -> ObbRunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping of settings")
    return config_from_dict(data)


def apply_overrides(data: dict[str, Any], pairs: Sequence[str]) -> dict[str, Any]:
    """Apply `key=value` overrides (values parsed as YAML, `augment.p_multi=0.2` reaches into the augmentation settings)."""
    out = copy.deepcopy(data)
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"override '{pair}' must look like key=value")
        *parents, leaf = key.strip().split(".")
        target = out
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = yaml.safe_load(raw)
    return out


def aug_config_from(overrides: dict[str, Any], imgsz: int) -> AugConfig:
    if "imgsz" in overrides:
        raise ValueError("'imgsz' is set at the top level of the run configuration, not under 'augment'")
    unknown = sorted(set(overrides) - {f.name for f in fields(AugConfig)})
    if unknown:
        raise ValueError(f"unknown augmentation key(s): {', '.join(unknown)}")
    return AugConfig(imgsz=imgsz, **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})


def resolve_workers(workers: int | None) -> int:
    if workers is not None:
        return int(workers)
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJ_ROOT / path


@dataclass(frozen=True)
class ObbPaths:
    data_dir: Path
    raw_dir: Path
    runs_dir: Path
    save_dir: Path
    init: Path | None

    @classmethod
    def from_config(cls, cfg: ObbRunConfig) -> ObbPaths:
        runs_dir = _resolve(cfg.project) if cfg.project else RUNS_DIR
        return cls(
            data_dir=_resolve(cfg.data_dir) if cfg.data_dir else DEFAULT_OUT_DIR,
            raw_dir=_resolve(cfg.raw_dir) if cfg.raw_dir else RAW_DATA_DIR,
            runs_dir=runs_dir,
            save_dir=runs_dir / cfg.name,
            init=_resolve(cfg.init) if cfg.init else None,
        )

    @property
    def dataset_yaml(self) -> Path:
        return self.data_dir / "dataset.yaml"

    @property
    def labels_csv(self) -> Path:
        return self.data_dir / "labels.csv"


def ultralytics_args(cfg: ObbRunConfig, paths: ObbPaths) -> dict[str, Any]:
    """The arguments of `YOLO.train`. Ultralytics' own augmentations are off: the training samples come from `WingOBBDataset`
    and the validation set is frozen. `exist_ok` makes the run folder exactly `project/name` (a finished run is guarded in `train`)."""
    return dict(
        data=str(paths.dataset_yaml), epochs=cfg.epochs, patience=cfg.patience, batch=cfg.batch, imgsz=cfg.imgsz, workers=resolve_workers(cfg.workers),
        device=cfg.device, optimizer=cfg.optimizer, lr0=cfg.lr0, lrf=cfg.lrf, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, cos_lr=cfg.cos_lr, close_mosaic=cfg.close_mosaic, seed=cfg.seed, amp=cfg.amp, deterministic=cfg.deterministic,
        fraction=cfg.fraction, time=cfg.time, save_period=cfg.save_period, plots=cfg.plots, cache=False, val=True,
        pretrained=str(paths.init) if paths.init else False, project=str(paths.runs_dir), name=cfg.name, exist_ok=True,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, multi_scale=0.0,
    )  # fmt: skip


class WingOBBTrainer(OBBTrainer):
    """OBBTrainer whose training set is `WingOBBDataset`; validation uses the stock dataset on the frozen val set."""

    aug_config: AugConfig = AugConfig()
    labels_path: Path = DEFAULT_OUT_DIR / "labels.csv"
    raw_dir: Path | None = None

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        return WingOBBDataset(
            self.labels_path, "train", img_path, raw_dir=self.raw_dir, cfg=self.aug_config, hyp=self.args, batch_size=batch or self.args.batch,
            fraction=self.args.fraction,
        )  # fmt: skip

    def plot_training_labels(self) -> None:
        """Skipped: labels.jpg would draw the raw, unaugmented boxes, while the training samples are generated online."""


def make_trainer(aug: AugConfig, labels_path: Path | str, raw_dir: Path | str | None) -> type[WingOBBTrainer]:
    """`YOLO.train(trainer=...)` instantiates the class itself, so the settings are bound to a subclass."""
    attributes = {"aug_config": aug, "labels_path": Path(labels_path), "raw_dir": Path(raw_dir) if raw_dir else None}
    return type("BoundWingOBBTrainer", (WingOBBTrainer,), attributes)


def prepare_data(paths: ObbPaths) -> None:
    """Check the inputs and refresh the image lists for this machine (they hold absolute paths). Safe while other jobs run."""
    for required in (paths.labels_csv, paths.dataset_yaml, paths.data_dir / "val" / "images", paths.data_dir / "val" / "labels"):
        if not required.exists():
            raise FileNotFoundError(f"{required} is missing; jobs/obb/README.md describes how the data gets onto this machine")
    if paths.init is not None and not paths.init.exists():
        raise FileNotFoundError(f"initial weights {paths.init} not found")
    write_image_lists(pd.read_csv(paths.labels_csv), paths.raw_dir, paths.data_dir)


@contextmanager
def wandb_integration(enabled: bool) -> Iterator[None]:
    """Make the configuration, not the per-user Ultralytics setting `wandb`, decide whether its W&B callback is active. Where the setting was
    on, `wandb: false` still started the callback, which opens a run of its own named after the absolute `project` path and crashes on it.
    The callback module reads the setting when the first trainer of the process is created, so this wraps the whole training. Only this
    process is changed (`dict.update`): assigning to SETTINGS would rewrite the settings file of the user and of the other projects."""
    previous = SETTINGS["wandb"]
    dict.update(SETTINGS, {"wandb": enabled})
    try:
        yield
    finally:
        dict.update(SETTINGS, {"wandb": previous})


def start_wandb(cfg: ObbRunConfig, paths: ObbPaths, aug: AugConfig):
    """One W&B run per training run, continued after a resume. Ultralytics' W&B callback (see `wandb_integration`) finds this run and logs into it."""
    import wandb

    id_file = paths.save_dir / "wandb_run_id.txt"
    run_id = id_file.read_text(encoding="utf-8").strip() if id_file.exists() else wandb.util.generate_id()
    id_file.write_text(run_id, encoding="utf-8")
    return wandb.init(
        entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.name, group=cfg.wandb_group, id=run_id, resume="allow", dir=str(paths.save_dir),
        config={**asdict(cfg), "augment_effective": asdict(aug)},
    )  # fmt: skip


def train(cfg: ObbRunConfig, resume: bool = False, callbacks: dict[str, Callable] | None = None) -> Path:
    """Run one training run and return its folder. `callbacks` are extra Ultralytics callbacks (the tests interrupt a run with one)."""
    paths = ObbPaths.from_config(cfg)
    last = paths.save_dir / "weights" / "last.pt"
    if resume and not last.exists():
        raise FileNotFoundError(f"nothing to resume: {last} does not exist")
    if not resume and last.exists():
        raise FileExistsError(f"{paths.save_dir} already holds a run named '{cfg.name}'; continue it with --resume or choose another name")
    prepare_data(paths)
    aug = aug_config_from(cfg.augment, cfg.imgsz)
    paths.save_dir.mkdir(parents=True, exist_ok=True)
    (paths.save_dir / "obb_config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    logger.info(f"run '{cfg.name}': init={paths.init}, data={paths.data_dir}, runs={paths.save_dir}, augmentation={aug}")
    trainer_cls = make_trainer(aug, paths.labels_csv, paths.raw_dir)
    with wandb_integration(cfg.wandb):
        run = start_wandb(cfg, paths, aug) if cfg.wandb else None
        try:
            model = YOLO(str(last) if resume else cfg.model)
            for event, function in (callbacks or {}).items():
                model.add_callback(event, function)
            if resume:
                model.train(resume=True, trainer=trainer_cls)
            else:
                model.train(trainer=trainer_cls, **ultralytics_args(cfg, paths))
        finally:
            if run is not None:
                run.finish()
    return paths.save_dir


def evaluate(
    weights: Path | str, split: str = "val", data_dir: Path | str | None = None, imgsz: int = 640, batch: int = 32, device: int | str = 0,
    project: Path | str | None = None, workers: int | None = None,
) -> dict[str, float]:  # fmt: skip
    """Rotated-box precision, recall and mAP of a checkpoint on the frozen `val` or `test` set (the images are the frozen samples, no raw data
    needed). Ultralytics' output of the evaluation goes to `project`/<run>-<split> (default: runs/obb/eval); `workers` as in the run configuration."""
    data_dir = _resolve(data_dir) if data_dir else DEFAULT_OUT_DIR
    project = _resolve(project) if project else RUNS_DIR / "eval"
    if split == "test":
        logger.warning("evaluating on the test set: use it once, for the final model")
    metrics = YOLO(str(weights)).val(
        data=str(data_dir / "dataset.yaml"), split=split, imgsz=imgsz, batch=batch, device=device, workers=resolve_workers(workers), plots=False, verbose=False,
        project=str(project), name=f"{Path(weights).parent.parent.name}-{split}", exist_ok=True,
    )  # fmt: skip
    results = metrics.results_dict
    return {
        "map50-95": float(results["metrics/mAP50-95(B)"]), "map50": float(results["metrics/mAP50(B)"]),
        "precision": float(results["metrics/precision(B)"]), "recall": float(results["metrics/recall(B)"]),
    }  # fmt: skip


@app.command("train")
def train_command(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="YAML file of the run (configs/obb/*.yaml)."),
    resume: bool = typer.Option(False, "--resume", help="Continue the run from its last checkpoint."),
    overrides: list[str] = typer.Option([], "--set", help="key=value override, e.g. --set epochs=2 --set augment.p_multi=0.2 (repeatable)."),
) -> None:
    """Train one run."""
    data = apply_overrides(yaml.safe_load(config.read_text(encoding="utf-8")) or {}, overrides)
    save_dir = train(config_from_dict(data), resume=resume)
    logger.info(f"run finished: {save_dir}")


@app.command("eval")
def eval_command(
    weights: Path = typer.Argument(..., exists=True, dir_okay=False, help="Checkpoint, e.g. runs/obb/<name>/weights/best.pt."),
    split: str = typer.Option("val", "--split", help="'val' or 'test' (the test set is for the final model only)."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Folder with dataset.yaml (default: data/processed/detection-obb)."),  # noqa: UP007
    imgsz: int = typer.Option(640, "--imgsz"),
    batch: int = typer.Option(32, "--batch"),
    device: str = typer.Option("0", "--device"),
    project: Optional[Path] = typer.Option(None, "--project", help="Folder for the output of the evaluation (default: wings/detection/runs/obb/eval)."),  # noqa: UP007
    workers: Optional[int] = typer.Option(None, "--workers", help="DataLoader workers (default: the CPUs of the SLURM task, else all CPUs)."),  # noqa: UP007
) -> None:
    """Evaluate a checkpoint on the frozen val or test set and print the metrics as JSON."""
    print(json.dumps(evaluate(weights, split=split, data_dir=data_dir, imgsz=imgsz, batch=batch, device=device, project=project, workers=workers), indent=2))


if __name__ == "__main__":
    app()
