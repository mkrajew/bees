import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class LitNet(L.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, num_epochs: int) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_epochs = num_epochs

        self.mse_test = torchmetrics.regression.MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, target = batch
        target = target.float()
        output = self.model(x)
        loss = self.criterion(output, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, target = batch
        target = target.float()
        output = self.model(x)
        loss = self.criterion(output, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x, target = batch
        target = target.float()
        output = self.model(x)

        self.mse_test(output, target)
        self.log('test_loss', self.mse_test, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


