import json
from pathlib import Path

import lightning as L
import torch
import wandb
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from loguru import logger

from wings.modeling.litnet import LitNet, compute_statistics
from wings.config import PROCESSED_DATA_DIR
from wings.transforms import seed_worker
from wings.deepwings_eval import evaluate_checkpoint_on_deepwings


def train(
    model: torch.nn.Module,
    datasets: tuple[data.Dataset, data.Dataset, data.Dataset],
    params: dict,
    path=None,
    strict: bool = True,
) -> None:
    """
    Trains and evaluates a PyTorch model using the Lightning framework.

    This function sets up the Lightning training pipeline, including logging with Weights & Biases,
    model checkpointing, early stopping, and progress monitoring. It takes the given model and datasets,
    wraps the model in a `LitNet` LightningModule, and trains it using the provided parameters.

    Args:
        model: The PyTorch model to be trained.
        datasets: A tuple containing (train_dataset, val_dataset, test_dataset), each a Dataset object.
        params: A dictionary of training configuration parameters. Expected keys include:
            - "num_epochs" (int): Number of training epochs.
            - "project_name" (str): Project name for W&B logging.
            - "logger_save_dir" (str): Directory to save W&B logs.
            - "run_name" (str): Name of the current training run.
            - "early_stop_min_delta" (float): Minimum change in validation loss to qualify as improvement.
            - "early_stop_patience" (int): Number of epochs with no improvement after which training will stop.
            - "checkpoint_save_dir" (str): Directory to save model checkpoints.
            - "checkpoint_filename" (str): Filename pattern for checkpoint files.
            - "batch_size" (int): Batch size for all dataloaders.
            - "num_workers" (int): Number of subprocesses to use for data loading.
            - "criterion" (torch.nn.Module): Loss function to optimize.
        path: Optional checkpoint to warm-start from.
        strict: Passed to `LitNet.load_from_checkpoint` when `path` is given. Set to
            False when the checkpoint was saved with a different criterion than
            `params["criterion"]` (e.g. its own stateful buffers, like
            BCEDiceLoss's `pos_weight`, won't have a matching key to load into).
    """

    mean_coords = torch.load(
        PROCESSED_DATA_DIR / "mask_datasets" / "rectangle" / "mean_shape.pth",
        weights_only=False,
    )

    if path is None:
        lit_net = LitNet(
            model,
            criterion=params["criterion"],
            num_epochs=params["num_epochs"],
            mean_coords=mean_coords,
        )
    else:
        lit_net = LitNet.load_from_checkpoint(
            path,
            model=model,
            criterion=params["criterion"],
            num_epochs=params["num_epochs"],
            mean_coords=mean_coords,
            strict=strict,
        )

    wandb_logger = WandbLogger(
        project=params["project_name"],
        save_dir=params["logger_save_dir"],
        name=params["run_name"],
    )

    csv_logger = CSVLogger(
        save_dir="logs",
        name="csv_logs",
        version=params["run_name"],
    )

    # Both callbacks monitor the *smoothed* metric (LitNet.on_validation_epoch_end,
    # a trailing moving average over `smooth_window` epochs), not the raw
    # per-epoch val_mean_error_px: across the online-augmentation series, the
    # raw "best" epoch landed on epoch 5 for nearly every config, with more
    # same-run epoch-to-epoch noise than the actual gap between configs --
    # i.e. the raw signal was picking up noise, not real differences.
    early_stop_callback = EarlyStopping(
        monitor="val_mean_error_px_smooth",
        min_delta=params["early_stop_min_delta"],
        patience=params["early_stop_patience"],
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        # Raised from 2: the smoothed metric picks a more trustworthy single
        # winner than the raw one did, but val_wrong_spot_count_pct_smooth
        # and val_loss_smooth can still rank a nearby epoch differently --
        # keeping the top 5 by the primary smoothed metric leaves real
        # candidates on disk to check against those other two by hand
        # instead of only ever having the one epoch this callback picked.
        save_top_k=5,
        save_last=True,
        monitor="val_mean_error_px_smooth",
        mode="min",
        dirpath=params["checkpoint_save_dir"],
        filename=params["checkpoint_filename"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=params["num_epochs"],
        logger=[wandb_logger, csv_logger],
        # callbacks=[early_stop_callback, RichProgressBar(), checkpoint_callback],
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        deterministic=True,
    )

    train_dataset, val_dataset, test_dataset = datasets
    use_persistent_workers = params["num_workers"] > 0

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        persistent_workers=use_persistent_workers,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker if params["num_workers"] > 0 else None,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        persistent_workers=use_persistent_workers,
        shuffle=False,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        persistent_workers=use_persistent_workers,
        shuffle=False,
    )

    trainer.fit(lit_net, train_dataloader, val_dataloader)

    trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    # Score every checkpoint this run actually saved (the top save_top_k by
    # val_mean_error_px_smooth, plus save_last) on both our own held-out
    # test set AND DeepWings' published one, side by side in one report --
    # DeepWings is the evaluation neither training nor validation ever
    # touches, so it isn't subject to the per-epoch noise that motivated
    # smoothing val_mean_error_px in the first place (see litnet.py).
    #
    # The our-test-set half is done with a fresh model/LitNet per checkpoint
    # rather than by looping trainer.test(ckpt_path=...) on the existing
    # `trainer`: that would re-log through wandb_logger each time and
    # repeatedly overwrite the run's test_mean_error_px *summary*, which the
    # single trainer.test(ckpt_path="best", ...) call just above is relied
    # on elsewhere (comparing configs 8/8a-9e) to hold specifically the
    # *best* checkpoint's result, not whichever one this loop tests last.
    #
    # Checkpoints are discovered by scanning checkpoint_save_dir directly,
    # not via checkpoint_callback.best_k_models/last_model_path: on config
    # 10a's actual run (61 epochs, 5 saved + last.ckpt on disk), that
    # in-memory bookkeeping only yielded 1 path at this point despite 6
    # files existing -- root cause not fully pinned down (network
    # filesystem quirk interacting with Lightning's own tracking is the
    # leading suspect, given this runs over NFS-mounted home dirs), but a
    # directory scan reflects what actually got saved regardless of why the
    # callback's own state diverged from disk.
    checkpoint_paths = sorted(
        str(p) for p in Path(params["checkpoint_save_dir"]).glob("*.ckpt")
    )

    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for ckpt_path in checkpoint_paths:
        entry = {"checkpoint": str(ckpt_path)}

        try:
            eval_model = type(model)(
                in_channels=1, out_channels=1, kernel_size=5, sigmoid=model.sigmoid
            )
            eval_lit_net = LitNet.load_from_checkpoint(
                ckpt_path,
                model=eval_model,
                criterion=params["criterion"],
                num_epochs=1,
                mean_coords=mean_coords,
                strict=False,
            )
            eval_lit_net.eval()
            eval_lit_net.to(eval_device)

            error_distances, wrong_spot_count = [], []
            with torch.no_grad():
                for bx, _, bcoords, (bx_size, by_size) in test_dataloader:
                    output = eval_lit_net.model(bx.to(eval_device))
                    dists, wrong = compute_statistics(
                        output=output,
                        coords=bcoords,
                        x_size=bx_size,
                        y_size=by_size,
                        mean_coords=mean_coords,
                        output_is_logits=not eval_model.sigmoid,
                    )
                    error_distances.extend(dists)
                    wrong_spot_count.extend(wrong)

            distances = torch.tensor(error_distances)
            entry["our_test_mean_error_px"] = (
                distances.mean().item() if len(distances) else float("nan")
            )
            entry["our_test_median_error_px"] = (
                distances.median().item() if len(distances) else float("nan")
            )
            entry["our_test_wrong_spot_count_pct"] = (
                torch.tensor(wrong_spot_count).mean().item() * 100.0
                if wrong_spot_count
                else float("nan")
            )

            del eval_lit_net, eval_model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Our-test-set evaluation failed for {ckpt_path}: {e!r}")

        try:
            dw_result = evaluate_checkpoint_on_deepwings(
                ckpt_path, mean_coords, sigmoid=model.sigmoid
            )
            entry["deepwings_precision_mean"] = dw_result["precision_mean"]
            entry["deepwings_precision_median"] = dw_result["precision_median"]
            entry["deepwings_reliable_pct"] = dw_result["reliable_pct"]
            entry["deepwings_n_samples"] = dw_result["n_samples"]
        except Exception as e:
            logger.warning(f"DeepWings evaluation failed for {ckpt_path}: {e!r}")

        results.append(entry)
        logger.info(f"Checkpoint eval [{ckpt_path}]: {entry}")

    if results:
        report_path = params["checkpoint_save_dir"] / "checkpoint_eval.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved combined checkpoint evaluation report to {report_path}")

        columns = sorted({k for r in results for k in r})
        wandb_logger.experiment.log(
            {
                "checkpoint_eval": wandb.Table(
                    columns=columns,
                    data=[[r.get(c) for c in columns] for r in results],
                )
            }
        )

    wandb_logger.experiment.finish()

    del model
    del lit_net
    del trainer

    torch.cuda.empty_cache()
