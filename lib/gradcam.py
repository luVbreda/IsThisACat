import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Acessa o submodelo real que tem a arquitetura CNN
    inner_model = model.get_layer("image_classifier")

    # Pega a última camada convolucional dentro do submodelo
    last_conv_layer = inner_model.get_layer(last_conv_layer_name)

    # Cria o modelo para Grad-CAM usando o submodelo
    grad_model = tf.keras.models.Model(
        [inner_model.input],
        [last_conv_layer.output, inner_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def display_gradcam(image_path, heatmap, alpha=0.4):
    target_size = (1024, 1024)

    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    heatmap = cv2.resize(heatmap, target_size)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    cv2.imshow("Grad-CAM", np.uint8(superimposed_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
