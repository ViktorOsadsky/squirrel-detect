import random
import shutil
from pathlib import Path

src_images = Path("datasets/squirrel/dataset_two/images/train")
src_labels = Path("datasets/squirrel/dataset_two/labels/train")

out_images = Path("datasets/squirrel_subset/images/train")
out_labels = Path("datasets/squirrel_subset/labels/train")

out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

images = list(src_images.glob("*.*"))
sample = random.sample(images, 100)

for img in sample:
    label = src_labels / f"{img.stem}.txt"
    shutil.copy2(img, out_images / img.name)
    if label.exists():
        shutil.copy2(label, out_labels / label.name)