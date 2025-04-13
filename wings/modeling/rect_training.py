from loguru import logger
from torch import nn
from torchvision.models import ViT_B_32_Weights, vit_b_32, ResNet50_Weights, resnet50

from wings.config import PROCESSED_DATA_DIR, MODELLING_DIR, DEVICE
from wings.dataset import load_datasets
from wings.modeling.models import TransformerPreTrained, ResnetPreTrained
from wings.modeling.training import train

run_num = 2
run_name = "rectangle_images"

transf_model_name = 'transformer32'
transformer_params = {
    "project_name": "bees-wings-modeling2",
    "logger_save_dir": MODELLING_DIR,
    "run_name": f"{transf_model_name}_{run_name}_{run_num}",
    "checkpoint_save_dir": MODELLING_DIR / "lightning-checkpoints" / transf_model_name / "rectangle_images",
    "checkpoint_filename": transf_model_name + "-{epoch:02d}-{val_loss:.2f}-" + f"{run_name}_run{run_num}",
    "num_epochs": 60,
    "batch_size": 16,
    "num_workers": 10,
    "early_stop_min_delta": 0.1,
    "early_stop_patience": 5,
}

resnet_model_name = 'resnet50'
resnet_params = {
    "project_name": "bees-wings-modeling2",
    "logger_save_dir": MODELLING_DIR,
    "run_name": f"{resnet_model_name}_{run_name}_{run_num}",
    "checkpoint_save_dir": MODELLING_DIR / "lightning-checkpoints" / resnet_model_name / "rectangle_images",
    "checkpoint_filename": resnet_model_name + "-{epoch:02d}-{val_loss:.2f}-" + f"{run_name}_{run_num}",
    "num_epochs": 50,
    "batch_size": 16,
    "num_workers": 10,
    "early_stop_min_delta": 1,
    "early_stop_patience": 10,
    "criterion": nn.MSELoss,
}

if __name__ == "__main__":
    train_val_test_datasets = load_datasets(
        [PROCESSED_DATA_DIR / 'train_rec_dataset.pth',
         PROCESSED_DATA_DIR / 'val_rec_dataset.pth',
         PROCESSED_DATA_DIR / 'test_rec_dataset.pth']
    )
    logger.info("Loaded datasets.")

    weights = ViT_B_32_Weights.DEFAULT
    transf_model = TransformerPreTrained(vit_b_32, weights)
    transf_model.to(DEVICE)

    train(transf_model, train_val_test_datasets, transformer_params)

    weights = ResNet50_Weights.DEFAULT
    resnet_model = ResnetPreTrained(resnet50, weights)
    resnet_model.to(DEVICE)

    train(resnet_model, train_val_test_datasets, resnet_params)
