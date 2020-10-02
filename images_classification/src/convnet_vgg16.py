from convnet import ConvNet
from tensorflow import keras
from tensorflow.keras import layers


class ConvNetVGG16(ConvNet):
    def prepare_convnet_vgg16(self, fc1 = 4096, fc2 = 4096, activation = "softmax", freeze = 0):
        convnet_vgg16 = keras.applications.VGG16(weights = "imagenet", include_top = False)
        
        if freeze:
            for layer in convnet_vgg16.layers:
                layer.trainable = False
        
        inputs = keras.Input(shape = self.input_shape, name = "image_input")
        
        output_vgg16 = convnet_vgg16(inputs)
        
        x = layers.Flatten(name = "flatten")(output_vgg16)
        x = layers.Dense(fc1, activation = "relu", name = "fc1")(x)
        x = layers.Dense(fc2, activation = "relu", name = "fc2")(x)
        outputs = layers.Dense(self.num_classes, activation = activation, name = "predictions")(x)
        
        return keras.Model(inputs, outputs)