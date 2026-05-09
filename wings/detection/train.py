from wings.config import PROCESSED_DATA_DIR, MODELS_DIR, SEED, PROJ_ROOT

import wandb
from ultralytics import YOLO


def main():
    data_yaml = PROCESSED_DATA_DIR / "detection" / "dataset.yaml"
    project_name = "26n-train"
    run_name = "run-3"

    wandb.init(
        entity="furkot-team",
        project="wings-detection",
        name=f"{project_name}-{run_name}",
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
    # model = YOLO(MODELS_DIR / "yolo26n.pt")
    model_folder = (
        PROJ_ROOT / "wings" / "detection" / "runs" / "detect" / project_name / run_name
    )
    model = YOLO(model_folder / "last.pt")

    model.train(
        data=data_yaml,
        epochs=16,
        patience=5,
        batch=16,
        workers=4,
        project=project_name,
        name=run_name,
        device=0,
        seed=SEED,
        multi_scale=0.15,
        profile=False,
        degrees=20,
        shear=10,
        resume=True,
    )

    model = YOLO(model_folder / "last.pt")
    model.train(
        resume=True,
        epochs=25,
        degrees=10,
        shear=5,
    )

    model = YOLO(model_folder / "last.pt")
    model.train(
        resume=True,
        epochs=30,
        degrees=5,
        shear=2,
    )

    model = YOLO(model_folder / "best.pt")
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
