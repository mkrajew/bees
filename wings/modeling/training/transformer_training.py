from loguru import logger
from torch import nn
from torchvision.models import ViT_B_32_Weights, vit_b_32

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR
from wings.dataset import load_datasets
from wings.modeling.models import TransformerPreTrained
from wings.modeling.train import train

run_num = 1
run_name = "test-transformer"
model_name = 'transformer32'
PARAMETERS = {
    "project_name": "bees-wings-modeling2",
    "logger_save_dir": TRAINING_DIR,
    "run_name": f"{run_name}_{run_num}",
    "checkpoint_save_dir": TRAINING_DIR / "lightning-checkpoints" / model_name / "square_images",
    "checkpoint_filename": model_name + "-{epoch:02d}-{val_loss:.2f}-" + f"{run_name}_{run_num}",
    "num_epochs": 60,
    "batch_size": 16,
    "num_workers": 4,
    "early_stop_min_delta": 0.1,
    "early_stop_patience": 5,
    "criterion": nn.MSELoss,
}

if __name__ == "__main__":
    data_dir = PROCESSED_DATA_DIR / "resize_datasets"
    train_val_test_datasets = load_datasets(
        [data_dir / 'train_dataset.pth',
         data_dir / 'val_dataset.pth',
         data_dir / 'test_dataset.pth']
    )
    logger.info("Loaded datasets.")

    weights = ViT_B_32_Weights.DEFAULT
    model = TransformerPreTrained(vit_b_32, weights)
    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS)
