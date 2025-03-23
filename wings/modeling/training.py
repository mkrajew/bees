import torch
import lightning as L
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchvision.models import resnet50, ResNet50_Weights

from litnet import LitNet
from wings.config import COUNTRIES, RAW_DATA_DIR, DEVICE, MODELLING_DIR
from wings.dataset import Dataset
from wings.modeling.models import ResnetPreTrained

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="val_loss",
    mode="min",
    dirpath= MODELLING_DIR / "lightning-checkpoints" /" resnet",
    filename="sample-resnet50-{epoch:02d}-{val_loss:.2f}",
)


if __name__ == "__main__":
    weights = ResNet50_Weights.DEFAULT
    resnet_dataset = Dataset(COUNTRIES, RAW_DATA_DIR, weights.transforms())
    model = ResnetPreTrained(resnet50, weights)
    model.to(DEVICE)
    num_epochs = 15
    batch_size = 32
    num_workers = 4
    lit_net = LitNet(model, num_epochs=num_epochs)
    wandb_logger = WandbLogger(project="bees-wings-modeling", save_dir=MODELLING_DIR)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=100, patience=5, verbose=False, mode="min")
    trainer = L.Trainer(max_epochs=num_epochs, logger=wandb_logger,
                        callbacks=[early_stop_callback, RichProgressBar(), checkpoint_callback])

    train_dataset, val_dataset, test_dataset = resnet_dataset.split(0.2, 0.1)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                       drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    trainer.fit(lit_net, train_dataloader, val_dataloader)

    trainer.test(ckpt_path="best", dataloaders=test_dataloader)
    wandb_logger.experiment.finish()

    del model
    del lit_net
    del trainer

    torch.cuda.empty_cache()
