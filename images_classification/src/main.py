import numpy as np
import os

from data_reader import DataReader
from convnet_vgg16 import ConvNetVGG16
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    # Set testing parameters
    generators_types_list = ["v1"] #["v1", "v2"]
    images_per_class_num_list = [90] #[90, 270, 900]
    test_images_num = 270
    images_sizes_list = [64] #[64 * 2 ** i for i in range(3)]
    epochs = 10 #50
    batch_sizes_list = [32] #[32 * 2 ** i for i in range(3)]
    fully_connected_layers_sizes_list = [4096] #[16, 256, 4096]
    metrics = ["sparse_categorical_accuracy"]

    for generator_type in generators_types_list:
        for images_per_class_num in images_per_class_num_list:
            for images_size in images_sizes_list:
                # Set data directories
                train_images_dir = f"./../../images/{generator_type}/train/{images_per_class_num}_per_class/ovals/{images_size}x{images_size}"
                test_images_dir = f"./../../images/{generator_type}/test/ovals/{images_size}x{images_size}"

                # Set images parameters
                image_size = (images_size, images_size)
                input_shape = image_size + (3, )

                # Prepare directories
                plots_root_directory = f"./results/vgg16/plots/{generator_type}/{images_per_class_num}_per_class/{images_size}x{images_size}/"
                if not os.path.exists(plots_root_directory):
                    os.makedirs(plots_root_directory)
                
                training_results_root_directory = f"./results/vgg16/training_results/{generator_type}/{images_per_class_num}_per_class/{images_size}x{images_size}/"
                if not os.path.exists(training_results_root_directory):
                    os.makedirs(training_results_root_directory)
                
                models_root_directory = f"./models/vgg16/{generator_type}/{images_per_class_num}_per_class/{images_size}x{images_size}/"
                if not os.path.exists(models_root_directory):
                    os.makedirs(models_root_directory)

                for batch_size in batch_sizes_list:
                    # Read training and validation data
                    data_reader = DataReader(train_images_dir, image_size, batch_size)
                    train_data = data_reader.read_data("training")
                    validation_data = data_reader.read_data("validation")

                    # Read test data
                    test_data_reader = DataReader(test_images_dir, image_size, test_images_num, None, None)
                    test_data = test_data_reader.read_data()
                    test_data_batch, test_labels = next(iter(test_data))

                    for fc_size in fully_connected_layers_sizes_list:
                        results_name_suffix = f"{batch_size}_{fc_size}_{fc_size}"

                        # Prepare model
                        vgg16 = ConvNetVGG16(input_shape)
                        vgg16_model = vgg16.prepare_convnet_vgg16(fc1 = fc_size, fc2 = fc_size)

                        # Compile and fit model
                        vgg16_model_trained, vgg16_model_info = vgg16.compile_and_fit_model(vgg16_model, 
                                                                                            train_data,
                                                                                            validation_data,
                                                                                            batch_size,
                                                                                            epochs)
                        vgg16_model_info_history = vgg16_model_info.history
                        
                        # Plot training results
                        vgg16.plot_results(vgg16_model_info_history, metrics, plots_root_directory, results_name_suffix)

                        # Save training results
                        training_results_path = f"{training_results_root_directory}training_results_{results_name_suffix}"
                        vgg16.save_training_results_csv(vgg16_model_info_history, training_results_path)

                        # Test model on test data
                        test_prediction_results = vgg16_model_trained.evaluate(test_data)
                        test_predictions = vgg16_model_trained.predict(test_data_batch)
                        test_labels_pred = np.argmax(test_predictions, axis = 1)
                        test_confusion_matrix = confusion_matrix(test_labels, test_labels_pred).ravel()

                        # Save model
                        model_path = f"{models_root_directory}vgg16_{results_name_suffix}.h5"
                        vgg16_model_trained.save(model_path)

                        # TO DO: save results to csv