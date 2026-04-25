from ultralytics import YOLO

from wings.config import PROCESSED_DATA_DIR, MODELS_DIR, SEED


def main():
    # model = YOLO(MODELS_DIR / "yolo26m.pt")
    model = YOLO("wings\\detection\\runs\\detect\\26m-train\\run-1\\weights\\last.pt")

    results = model.train(
        data=PROCESSED_DATA_DIR / "detection" / "dataset.yaml",
        epochs=30,
        patience=5,
        batch=8,
        project="26m-train",
        name="run-1",
        device=0,  # GPU 0
        seed=SEED,
        multi_scale=0.15,
        profile=True,
        degrees=10,
        shear=5,
        resume=True,
    )

    results = model.val()


if __name__ == "__main__":
    main()
