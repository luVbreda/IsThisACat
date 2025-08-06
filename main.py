import os
from datasets import load_dataset

# Class name for download
TARGET_TAG = "cat"

# Max number of samples to download
MAX_SAMPLES = 750

# Path to save the dataset
OUTPUT_DIR = f"./data/{TARGET_TAG}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading dataset...")
dataset = load_dataset("cifar10", split="train") # Split can be "test" also.

# Discovery the IDs of the target class
try:
    label_id = dataset.features['label'].names.index(TARGET_TAG)
    print(f"Label ID for '{TARGET_TAG}': {label_id}")

except ValueError:
    print(f"Error: The label '{TARGET_TAG}' does not exist in the dataset.")
    print("Available labels:", dataset.features['label'].names)
    exit()

# Filter the dataset for the target class
print("Filtering dataset for the target class...")
filtered_dataset = dataset.filter(lambda x: x['label'] == label_id)

print(f"Number of samples in the filtered dataset: {len(filtered_dataset)}")

for i, x in enumerate(filtered_dataset.select(range(min(MAX_SAMPLES, len(filtered_dataset))))):
    image = x['img']
    image.save(os.path.join(OUTPUT_DIR, f"{TARGET_TAG}_{i}.png"))

print(f"Dataset saved to {OUTPUT_DIR}")