import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
import os
from lib.gradcam import make_gradcam_heatmap, display_gradcam

class_labels = ["airplane", "bird", "cat", "dog", "horse", "truck"]

def is_cat(image_path: str, model_path: str):
    if not os.path.exists(model_path):
        print("Model or class names file not found. Please train the model first using the 'train' command.")
        return None
    
    # Load the trained model
    print("Loading model for prediction...")
    model = keras.models.load_model(model_path)

    # Reconstruir o modelo para garantir que .output esteja definido
    inputs = Input(shape=(128, 128, 3))
    outputs = model(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    # Preprocess the image
    img = keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(img_array)
    score = predictions[0]
    class_id = tf.argmax(score).numpy()
    confidence = 100 * tf.reduce_max(score).numpy()

    predicted_class = class_labels[class_id]
    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}% for image {image_path}")

    # Gerar o heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv3")
    display_gradcam(image_path, heatmap)

    if predicted_class == "cat":
        print(f"The image {image_path} is a cat with {confidence:.2f}% confidence.")
    else:
        print(f"The image {image_path} is not a cat.")