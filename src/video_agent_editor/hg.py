from datasets import load_dataset

dataset = load_dataset(
    "microsoft/cats_vs_dogs",
    split="train"
)

for data in dataset:
    if data["labels"] == 0:  # 0 = cat
        image = data["image"]
        image.show()
        break