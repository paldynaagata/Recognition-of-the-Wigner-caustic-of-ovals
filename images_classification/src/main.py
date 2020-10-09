from data_reader import DataReader
from convnet_vgg16 import ConvNetVGG16


if __name__ == "__main__":
    # Data parameters
    images_dir = "./../../images/v1/train/90_per_class/ovals/64x64"
    class_names = ["class_3", "class_5", "class_7"]
    validation_split = 0.3
    image_size = (64, 64)
    batch_size = 32
    seed = 123


    # Read data
    train_data_reader = DataReader(images_dir, class_names, validation_split, "training", seed, image_size, batch_size)
    validation_data_reader = DataReader(images_dir, class_names, validation_split, "validation", seed, image_size, batch_size)

    train_data = train_data_reader.read_data()
    validation_data = validation_data_reader.read_data()


    # Prepare model
    input_shape = image_size + (3, )
    vgg16 = ConvNetVGG16(input_shape)
    vgg16_model = vgg16.prepare_convnet_vgg16(freeze = 1)


    # Compile and fit model
    epochs = 20
    metrics = ["accuracy"]
    vgg16_model_trained, vgg16_model_info = vgg16.compile_and_fit_model(vgg16_model, 
                                                                        train_data,
                                                                        validation_data,
                                                                        batch_size,
                                                                        epochs)
    
    # Plot results
    vgg16.plot_results(vgg16_model_info, metrics)


    # Save model
    vgg16_model_trained.save("./models/vgg16_v1_090_64.h5")