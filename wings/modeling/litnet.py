from collections import deque

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from wings.modeling.loss import DiceLoss
from wings.visualizing.image_preprocess import final_coords
from wings.gpa import handle_coordinates


class LitNet(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = DiceLoss(),
        num_epochs: int = 60,
        mean_coords=None,
        smooth_window: int = 5,
        monitor_metric: str = "val_mean_error_px_smooth",
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.mean_coords = mean_coords
        # Which validation metric configure_optimizers' ReduceLROnPlateau
        # watches -- kept in sync with train.py's EarlyStopping/ModelCheckpoint
        # via params["checkpoint_monitor"], so all three either watch the
        # smoothed metric together or the raw one together, never a mix.
        self.monitor_metric = monitor_metric

        self.mse_test = torchmetrics.regression.MeanSquaredError()

        # Trailing moving average over the last `smooth_window` validation
        # epochs, logged alongside the raw per-epoch values (see
        # on_validation_epoch_end). Checkpointing/early-stopping/LR-plateau
        # on the raw val_mean_error_px alone picks up single-epoch noise --
        # across the online-augmentation series, the "best" raw epoch landed
        # on epoch 5 for nearly every config, with a same-run epoch-to-epoch
        # std (~0.015-0.04px) larger than the actual gap between configs'
        # best values (~0.001-0.003px), i.e. the raw signal can't reliably
        # tell configs apart. Smoothing three metrics (not just mean error)
        # separately, rather than collapsing them into one weighted score,
        # keeps them independently inspectable -- a model can be smoothed-good
        # on error but not on wrong_spot_count_pct, which matters just as much.
        self.smooth_window = smooth_window
        self._val_loss_history = deque(maxlen=smooth_window)
        self._val_mean_error_history = deque(maxlen=smooth_window)
        self._val_wrong_pct_history = deque(maxlen=smooth_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, target, _, _ = batch
        target = target.float()
        output = self.model(x)
        loss = self.criterion(output, target)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_start(self):
        self.val_error_distances = []
        self.val_wrong_spot_count = []

    def validation_step(self, batch, batch_idx: int):
        x, target, coords, (x_size, y_size) = batch
        target = target.float()

        output = self.model(x)

        loss = self.criterion(output, target)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        binary_metrics = binary_stats(
            output=output,
            target=target,
            output_is_logits=not self.model.sigmoid,
        )

        self.log(
            "val_dice",
            binary_metrics["dice"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_iou",
            binary_metrics["iou"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_precision", binary_metrics["precision"], on_step=False, on_epoch=True
        )
        self.log("val_recall", binary_metrics["recall"], on_step=False, on_epoch=True)

        error_distances, wrong_spot_count = compute_statistics(
            output=output,
            coords=coords,
            x_size=x_size,
            y_size=y_size,
            mean_coords=self.mean_coords,
            output_is_logits=not self.model.sigmoid,
        )

        self.val_error_distances.extend(error_distances)
        self.val_wrong_spot_count.extend(wrong_spot_count)

    def on_validation_epoch_end(self):
        mean_error, wrong_pct = self.log_epoch_statistics(
            error_distances=self.val_error_distances,
            wrong_spot_count=self.val_wrong_spot_count,
            prefix="val",
        )

        val_loss = self.trainer.callback_metrics.get("val_loss")
        self._val_loss_history.append(
            val_loss.item() if val_loss is not None else float("nan")
        )
        self._val_mean_error_history.append(
            mean_error.item() if torch.is_tensor(mean_error) else float(mean_error)
        )
        self._val_wrong_pct_history.append(
            wrong_pct.item() if torch.is_tensor(wrong_pct) else float(wrong_pct)
        )

        def _nanmean(values):
            finite = [v for v in values if v == v]  # drop NaNs
            return sum(finite) / len(finite) if finite else float("nan")

        self.log("val_loss_smooth", _nanmean(self._val_loss_history), prog_bar=True)
        self.log(
            "val_mean_error_px_smooth",
            _nanmean(self._val_mean_error_history),
            prog_bar=True,
        )
        self.log(
            "val_wrong_spot_count_pct_smooth",
            _nanmean(self._val_wrong_pct_history),
            prog_bar=True,
        )

    def on_test_epoch_start(self):
        self.test_error_distances = []
        self.test_wrong_spot_count = []

    def test_step(self, batch, batch_idx: int):
        x, target, coords, (x_size, y_size) = batch
        target = target.float()

        output = self.model(x)

        loss = self.criterion(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        binary_metrics = binary_stats(
            output=output,
            target=target,
            output_is_logits=not self.model.sigmoid,
        )

        self.log(
            "test_dice",
            binary_metrics["dice"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_iou",
            binary_metrics["iou"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_precision", binary_metrics["precision"], on_step=False, on_epoch=True
        )
        self.log("test_recall", binary_metrics["recall"], on_step=False, on_epoch=True)

        error_distances, wrong_spot_count = compute_statistics(
            output=output,
            coords=coords,
            x_size=x_size,
            y_size=y_size,
            mean_coords=self.mean_coords,
            output_is_logits=not self.model.sigmoid,
        )

        self.test_error_distances.extend(error_distances)
        self.test_wrong_spot_count.extend(wrong_spot_count)

    def on_test_epoch_end(self):
        self.log_epoch_statistics(
            error_distances=self.test_error_distances,
            wrong_spot_count=self.test_wrong_spot_count,
            prefix="test",
        )

    def log_epoch_statistics(self, error_distances, wrong_spot_count, prefix: str):
        """Returns (mean_error_px, wrong_spot_count_pct) as plain tensors so
        callers (on_validation_epoch_end) can feed them into the rolling
        smoothed-metric history without re-deriving them from logged state."""
        if len(error_distances) > 0:
            distances = torch.tensor(error_distances)
            mean_error = distances.mean()

            self.log(f"{prefix}_mean_error_px", mean_error, prog_bar=True)
            self.log(f"{prefix}_median_error_px", distances.median(), prog_bar=True)
        else:
            mean_error = torch.tensor(float("nan"))
            self.log(f"{prefix}_mean_error_px", mean_error, prog_bar=True)
            self.log(
                f"{prefix}_median_error_px", torch.tensor(float("nan")), prog_bar=True
            )

        if len(wrong_spot_count) > 0:
            wrong_count_rate = torch.tensor(wrong_spot_count).mean() * 100.0
        else:
            wrong_count_rate = torch.tensor(float("nan"))

        self.log(f"{prefix}_wrong_spot_count_pct", wrong_count_rate, prog_bar=True)

        return mean_error, wrong_count_rate

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            threshold=0.01,
            threshold_mode="abs",
            cooldown=2,
            min_lr=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def compute_statistics(
    output, coords, x_size, y_size, mean_coords, output_is_logits=True
):
    threshold = 0.5
    if output_is_logits:
        pred_masks = (torch.sigmoid(output) > threshold).detach().cpu()
    else:
        pred_masks = (output > threshold).detach().cpu()

    coords = coords.detach().cpu()
    x_size = x_size.detach().cpu()
    y_size = y_size.detach().cpu()

    error_distances = []
    wrong_spot_count = []

    batch_size = output.shape[0]

    for i in range(batch_size):
        mask = pred_masks[i].squeeze().numpy()

        pred_coords = final_coords(mask, int(x_size[i]), int(y_size[i]))

        pred_coords = torch.tensor(pred_coords, dtype=torch.float32)

        true_coords = coords[i].view(-1, 2).float()

        n_pred_points = len(pred_coords)

        wrong_spot_count.append(float(n_pred_points != 19))

        # allow_reflection=True: predictions can come from a horizontally-flipped
        # sample (TrainAugmentConfig.horizontal_flip_p), which a rotation-only
        # match can't align correctly against mean_coords.
        reordered = handle_coordinates(pred_coords, mean_coords, allow_reflection=True)
        reordered = reordered.detach().cpu().float()

        distances = torch.norm(reordered - true_coords, dim=1)
        error_distances.extend(distances.tolist())

    return error_distances, wrong_spot_count


def binary_stats(
    output,
    target,
    threshold=0.5,
    eps=1e-7,
    output_is_logits=True,
):
    if output_is_logits:
        probs = torch.sigmoid(output)
    else:
        probs = output

    pred = (probs > threshold).float()
    target = target.float()

    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)

    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "dice": dice.mean(),
        "iou": iou.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
    }
