import lightning as L
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from litnet import LitNet


def train(model: torch.nn.Module, datasets: tuple[data.Dataset, data.Dataset, data.Dataset], params: dict) -> None:
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
    """

    lit_net = LitNet(model, num_epochs=params["num_epochs"])

    wandb_logger = WandbLogger(
        project=params["project_name"],
        save_dir=params["logger_save_dir"],
        name=params["run_name"]
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=params["early_stop_min_delta"],
        patience=params["early_stop_patience"],
        verbose=False,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=params["checkpoint_save_dir"],
        filename=params["checkpoint_filename"],
    )

    trainer = L.Trainer(
        max_epochs=params["num_epochs"],
        logger=wandb_logger,
        callbacks=[early_stop_callback, RichProgressBar(), checkpoint_callback],
        deterministic=True
    )

    train_dataset, val_dataset, test_dataset = datasets

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=True,
        drop_last=True
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=False
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=False
    )

    trainer.fit(lit_net, train_dataloader, val_dataloader)

    trainer.test(ckpt_path="best", dataloaders=test_dataloader)
    wandb_logger.experiment.finish()

    del model
    del lit_net
    del trainer

    torch.cuda.empty_cache()
