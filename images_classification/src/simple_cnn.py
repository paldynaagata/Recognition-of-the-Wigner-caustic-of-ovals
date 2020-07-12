import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow import keras
from tensorflow.keras import layers


##########################################################################################################


def read_data(images_dir, class_names, validation_split, subset, seed, image_size, batch_size):
    dataset = keras.preprocessing.image_dataset_from_directory(
        images_dir,
        class_names = class_names,
        validation_split = validation_split,
        subset = subset,
        seed = seed,
        image_size = image_size,
        batch_size = batch_size
    )

    return dataset


def make_simple_cnn(input_shape, num_classes):
    inputs = keras.Input(shape = input_shape)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    x = data_augmentation(inputs)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    for i in range(4):
        size = 32 * 2 ** i
        x = layers.Conv2D(size, 3, strides = 2, padding = "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(3, strides = 2, padding = "same")(x)
    
    x = layers.Conv2D(512, 3, strides = 2, padding = "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation = "softmax")(x)
    return keras.Model(inputs, outputs)


def _plot_results(model_info, metric):
    plt.plot(model_info.history[metric])
    plt.plot(model_info.history[f"val_{metric}"])
    plt.title(f"Model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["train", "validation"])
    plt.savefig(f"./results/plots/{metric}.png")
    plt.close()


def plot_results(model_info, metrics):
    for metric in metrics:
        _plot_results(model_info, metric)


##########################################################################################################


# Data parameters
images_dir = "./../../images/ovals/128x128"
class_names = ["class_3", "class_5", "class_7"]
validation_split = 0.2
image_size = (128, 128)
batch_size = 32
seed = 123


# Read data
train_ds = read_data(images_dir, class_names, validation_split, "training", seed, image_size, batch_size)
val_ds = read_data(images_dir, class_names, validation_split, "validation", seed, image_size, batch_size)


# Build simple CNN
input_shape = image_size + (3, )
num_classes = 3
simple_cnn = make_simple_cnn(input_shape, num_classes)


# Train CNN
epochs = 50
metrics = ["accuracy"]#, "AUC", "Precision", "Recall"]

simple_cnn.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = metrics
)

start = time.time()
model_info  = simple_cnn.fit(
    train_ds, 
    epochs = epochs, 
    validation_data = val_ds
)
end = time.time()

print(f"Model took {end - start:0.2f} seconds to train")
print(model_info.history)


# Plots
plot_results(model_info, ["loss"] + metrics)
