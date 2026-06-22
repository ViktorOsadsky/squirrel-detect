"""Train a squirrel detector with Ultralytics."""

from pathlib import Path

from ultralytics import YOLO


DATASET = Path("datasets/squirrel/data.yaml")
MODEL = "yolo8n.pt"


def main() -> None:
    model = YOLO(MODEL)
    model.train(data=str(DATASET), epochs=50, imgsz=640)


if __name__ == "__main__":
    main()
