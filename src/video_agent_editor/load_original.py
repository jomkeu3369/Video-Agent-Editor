import os

from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = "src/data/Original"
IMAGE_COUNT = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    "microsoft/cats_vs_dogs",
    split="train"
)

saved = 0

for data in tqdm(dataset, desc="Saving cat images"):
    if data["labels"] != 0:
        continue

    image = data["image"].convert("RGB")

    save_path = os.path.join(
        OUTPUT_DIR,
        f"cat_{saved + 1}.png"
    )

    image.save(save_path)

    saved += 1

    if saved >= IMAGE_COUNT:
        break


print(f"Saved {saved} cat images")
print(f"Output directory: {OUTPUT_DIR}")