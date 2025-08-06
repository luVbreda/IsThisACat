import os
from datasets import load_dataset
from lib.downloader import download_images

# Class name for download
TARGET_TAGS = ["cat", "dog", "ship", "horse", "truck", "airplane", "bird", "automobile"]

# Max number of samples to download
MAX_SAMPLES = 750

download_images(TARGET_TAGS, MAX_SAMPLES)