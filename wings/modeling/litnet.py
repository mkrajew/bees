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
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.mean_coords = mean_coords

        self.mse_test = torchmetrics.regression.MeanSquaredError()

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
        )

        self.val_error_distances.extend(error_distances)
        self.val_wrong_spot_count.extend(wrong_spot_count)

    def on_validation_epoch_end(self):
        self.log_epoch_statistics(
            error_distances=self.val_error_distances,
            wrong_spot_count=self.val_wrong_spot_count,
            prefix="val",
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
        if len(error_distances) > 0:
            distances = torch.tensor(error_distances)

            self.log(f"{prefix}_mean_error_px", distances.mean(), prog_bar=True)
            self.log(f"{prefix}_median_error_px", distances.median(), prog_bar=True)
        else:
            self.log(
                f"{prefix}_mean_error_px", torch.tensor(float("nan")), prog_bar=True
            )
            self.log(
                f"{prefix}_median_error_px", torch.tensor(float("nan")), prog_bar=True
            )

        if len(wrong_spot_count) > 0:
            wrong_count_rate = torch.tensor(wrong_spot_count).mean() * 100.0
        else:
            wrong_count_rate = torch.tensor(float("nan"))

        self.log(f"{prefix}_wrong_spot_count_pct", wrong_count_rate, prog_bar=True)

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
                "monitor": "val_mean_error_px",
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

        reordered = handle_coordinates(pred_coords, mean_coords)
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
