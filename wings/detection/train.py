import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

from wings.config import PROCESSED_DATA_DIR, MODELS_DIR, SEED


def main():
    data_yaml = PROCESSED_DATA_DIR / "detection" / "dataset.yaml"

    wandb.init(
        entity="furkot-team",
        project="wings-detection",
        name="26n-run-1",
        config={
            "model": "yolo26n.pt",
            "epochs_stage_1": 25,
            "epochs_stage_2": 35,
            "epochs_stage_3": 40,
            "batch": 8,
            "seed": SEED,
            "multi_scale": 0.15,
            "stage_1_degrees": 20,
            "stage_1_shear": 10,
            "stage_2_degrees": 10,
            "stage_2_shear": 5,
            "stage_3_degrees": 5,
            "stage_3_shear": 2,
        },
    )

    # Stage 1: train from pretrained model
    model = YOLO(MODELS_DIR / "yolo26n.pt")
    add_wandb_callback(model, enable_model_checkpointing=True)

    model.train(
        data=data_yaml,
        epochs=25,
        patience=5,
        batch=8,
        project="26n-train",
        name="run-1",
        device=0,
        seed=SEED,
        multi_scale=0.15,
        profile=True,
        degrees=20,
        shear=10,
    )

    model.train(
        resume=True,
        epochs=35,
        degrees=10,
        shear=5,
    )

    model.train(
        resume=True,
        epochs=40,
        degrees=5,
        shear=2,
    )

    metrics = model.val(data=data_yaml)

    wandb.log(
        {
            "final/mAP50-95": metrics.box.map,
            "final/mAP50": metrics.box.map50,
            "final/precision": metrics.box.mp,
            "final/recall": metrics.box.mr,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
