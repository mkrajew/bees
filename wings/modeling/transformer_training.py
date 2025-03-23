import lightning as L
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchvision.models import ViT_B_32_Weights, vit_b_32

from litnet import LitNet
from wings.config import COUNTRIES, RAW_DATA_DIR, DEVICE, MODELLING_DIR
from wings.dataset import Dataset
from wings.modeling.models import TransformerPreTrained

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="val_loss",
    mode="min",
    dirpath=MODELLING_DIR / "lightning-checkpoints" / " resnet",
    filename="transformer-{epoch:02d}-{val_loss:.2f}-v01",
)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def preprocess(img: torch.Tensor) -> torch.Tensor:
    img = F.resize(img, [224, 224], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    # img = F.center_crop(img, self.crop_size)
    if not isinstance(img, torch.Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)
    return img


if __name__ == "__main__":
    weights = ViT_B_32_Weights.DEFAULT
    # resnet_dataset = Dataset(COUNTRIES, RAW_DATA_DIR, weights.transforms())
    resnet_dataset = Dataset(COUNTRIES, RAW_DATA_DIR, preprocess)
    model = TransformerPreTrained(vit_b_32, weights)
    model.to(DEVICE)
    num_epochs = 60
    batch_size = 16
    num_workers = 4
    lit_net = LitNet(model, num_epochs=num_epochs)
    wandb_logger = WandbLogger(project="bees-wings-modeling", save_dir=MODELLING_DIR, name="transformer-v01")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=20, patience=15, verbose=False, mode="min")
    trainer = L.Trainer(max_epochs=num_epochs, logger=wandb_logger,
                        callbacks=[early_stop_callback, RichProgressBar(), checkpoint_callback], deterministic=True)

    train_dataset, val_dataset, test_dataset = resnet_dataset.split(0.2, 0.1)

    torch.save(test_dataset, "test_dataset-transformer-v01.pth")

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
