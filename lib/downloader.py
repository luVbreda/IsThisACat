import os
from typing import List
from datasets import load_dataset

def download_images(tags: List[str], max_samples: int):
    dataset = load_dataset("cifar10", split="train") # Split can be "test" also.

    for tag in tags:
        OUTPUT_DIR = f"./data/{tag}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if len(os.listdir(OUTPUT_DIR)) != max_samples:
            # Clear the output directory if it exists but does not have the required number of samples
            print(f"Clearing existing files in {OUTPUT_DIR} because it does not have {max_samples} samples.")
            for f in os.listdir(OUTPUT_DIR):
                os.remove(os.path.join(OUTPUT_DIR, f))
            
            print(f"Downloading dataset for tag '{tag}'...")

            try:
                label_id = dataset.features['label'].names.index(tag)
                print(f"Label ID for '{tag}': {label_id}")
            
            except ValueError:
                print(f"Error: The label '{tag}' does not exist in the dataset.")
                print("Available labels:", dataset.features['label'].names)
                continue

            print(f"Filtering dataset for the target class '{tag}'...")
            filtered_dataset = dataset.filter(lambda x: x['label'] == label_id)

            for i, x in enumerate(filtered_dataset.select(range(min(max_samples, len(filtered_dataset))))):
                image = x['img']
                image.save(os.path.join(OUTPUT_DIR, f"{tag}_{i}.png"))
            
            print(f"Dataset for tag '{tag}' saved to {OUTPUT_DIR}")

        else:
            print(f"Dataset for tag '{tag}' already exists in {OUTPUT_DIR}, skipping download.")
