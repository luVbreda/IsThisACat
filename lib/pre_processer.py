from PIL import Image
import os

def preprocess_images(data_dir, image_size):
    for foldername, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(foldername, filename)
                img = Image.open(img_path)
                img = img.resize(image_size)
                img.save(img_path)