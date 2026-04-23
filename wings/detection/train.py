from ultralytics import YOLO

from wings.config import PROCESSED_DATA_DIR, MODELS_DIR


def main():
    model = YOLO(MODELS_DIR / "yolo26m.pt")

    results = model.train(
        data=PROCESSED_DATA_DIR / "detection" / "dataset.yaml",
        epochs=3,
        project="test-project",
        name="test-run",
    )

    results = model.val()


if __name__ == "__main__":
    main()
