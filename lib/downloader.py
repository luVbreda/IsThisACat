import os
from typing import List
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from PIL import Image

def download_images(tags: List[str], max_samples: int):
    # Load the dataset with all necessary classes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        classes=tags,
        max_samples=max_samples * len(tags),
        shuffle=True
    )

    for tag in tags:
        OUTPUT_DIR = f"./data/{tag}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Check if images already exist
        existing_files = os.listdir(OUTPUT_DIR)
        if len(existing_files) >= max_samples:
            print(f"[✓] Dataset for tag '{tag}' already exists in {OUTPUT_DIR}, skipping download.")
            continue

        # Clear existing files if the count is incorrect
        if existing_files:
            print(f"[-] Clearing existing files in {OUTPUT_DIR} (found {len(existing_files)}, expected {max_samples})")
            for f in existing_files:
                os.remove(os.path.join(OUTPUT_DIR, f))

        # Filter samples with the specific label
        print(f"[→] Filtering samples with label '{tag}'")
        view = dataset.filter_labels("ground_truth", F("label") == tag)

        print(f"[→] Found {len(view)} samples for tag '{tag}'")

        for i, sample in enumerate(view.take(max_samples)):
            image = Image.open(sample.filepath).convert("RGB")
            save_path = os.path.join(OUTPUT_DIR, f"{tag}_{i}.jpg")
            image.save(save_path)

        print(f"[✓] Saved {min(max_samples, len(view))} images to {OUTPUT_DIR}")
