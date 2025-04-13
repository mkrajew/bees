from loguru import logger
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from wings.config import DEVICE, MODELLING_DIR, PROCESSED_DATA_DIR
from wings.dataset import load_datasets
from wings.modeling.models import ResnetPreTrained
from wings.modeling.training import train

run_num = 4
run_name = "no-crop"
model_name = 'resnet50'
PARAMETERS = {
    "project_name": "bees-wings-modeling",
    "logger_save_dir": MODELLING_DIR,
    "run_name": f"{run_name}_{run_num}",
    "checkpoint_save_dir": MODELLING_DIR / "lightning-checkpoints" / model_name,
    "checkpoint_filename": model_name + "-{epoch:02d}-{val_loss:.2f}-" + f"{run_name}_{run_num}",
    "num_epochs": 50,
    "batch_size": 16,
    "num_workers": 4,
    "early_stop_min_delta": 20,
    "early_stop_patience": 12,
    "criterion": nn.MSELoss,
}

if __name__ == "__main__":
    train_val_test_datasets = load_datasets(
        [PROCESSED_DATA_DIR / 'train_dataset.pth',
         PROCESSED_DATA_DIR / 'val_dataset.pth',
         PROCESSED_DATA_DIR / 'test_dataset.pth']
    )
    logger.info("Loaded datasets.")

    weights = ResNet50_Weights.DEFAULT
    model = ResnetPreTrained(resnet50, weights)
    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS)
