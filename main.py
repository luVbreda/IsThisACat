from lib.downloader import download_images
from lib.trainer import train_model
from lib.predicter import is_cat
from lib.pre_processer import preprocess_images

# Class name for download
TARGET_TAGS = ["airplane", "bird", "cat", "dog", "horse", "truck"]

# Max number of samples to download
MAX_SAMPLES = 300

MODEL_PATH = "./model/image_classifier.h5"
IMAGE_TEST_PATH = "image_test/golden.png"

download_images(TARGET_TAGS, MAX_SAMPLES)
preprocess_images("data", (128, 128))
train_model()
is_cat(IMAGE_TEST_PATH, MODEL_PATH)