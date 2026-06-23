from pathlib import Path

from ultralytics import YOLO


DATASET = Path("datasets/squirrel_subset/dataset.yaml")
MODEL = "yolo26n.pt"


def main() -> None:
    model = YOLO(MODEL)
    model.train(data=str(DATASET), epochs=50, imgsz=640)
    model.val(data=str(DATASET))


if __name__ == "__main__":
    main()
