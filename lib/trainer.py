import tensorflow as tf
from tensorflow import keras
import json
import os

# Tranning parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "./data"

def train_model():
    if not os.path.exists("./model/image_classifier.h5"):
        print("Initiating model training...")

        train_dataset = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )

        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )

        class_names = train_dataset.class_names
        num_classes = len(class_names)

        print(f"Classes found: {class_names}")
        print(f"Number of classes: {num_classes}")

        # Optimize data loading
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        # Buld the model
        print("Building the model...")
        inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")

        x = keras.layers.Rescaling(1./255, name="rescaling")(inputs)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv1")(x)
        x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv2")(x)
        x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        x = keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv3")(x)

        x = keras.layers.Flatten(name="flatten")(x)
        x = keras.layers.Dense(64, activation='relu', name="dense1")(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax', name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="image_classifier")


        # Compile the model

        print("Compiling the model...")
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        model.summary()

        # Train the model
        print(f"Starting model training for {EPOCHS} epochs...")
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS
        )

        # Save the model
        model_save_path = "./model/image_classifier.h5"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save class names to ensure consistency
        class_names_path = os.path.join(os.path.dirname(model_save_path), "class_names.json")
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)
        print(f"Class names saved to {class_names_path}")

        return history
    
    else:
        print("Model already exists, skipping training.")
        return None

if __name__ == "__main__":
    # Make sure the TensorFlow finds the GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs found: {gpus}")

        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
        
    else:
        print("No GPUs found, using CPU for training.")
    
    train_model()