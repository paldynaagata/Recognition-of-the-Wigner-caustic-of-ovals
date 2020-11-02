import pandas as pd
import numpy as np
import itertools
import os
import gc

from data_reader import DataReader
from convnet_vgg16 import ConvNetVGG16
from sklearn.metrics import confusion_matrix


def prepare_labels(prefix):
    return [f"{prefix}_{x}" for x in [3, 5, 7]]


if __name__ == "__main__":
    # Set testing parameters
    generators_types_list = ["v1", "v2"]
    images_per_class_num_list = [90, 270, 900]
    images_sizes_list = [64 * 2 ** i for i in range(3)]
    epochs = 50
    batch_sizes_list = [32]
    fully_connected_layers_sizes_list = [16, 256, 4096]
    metrics = ["sparse_categorical_accuracy"]

    # Prepare test confusion matrix columns names
    test_confusion_matrix_col_names = ["_".join(label) for label in itertools.product(prepare_labels("true"), prepare_labels("pred"))]

    # Set parameters for saving results
    results_path = "./results/vgg16/results.csv"
    results_file_exists = os.path.isfile(results_path)

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
                    test_data_reader = DataReader(test_images_dir, image_size, batch_size, None, None)
                    test_data = test_data_reader.read_data()

                    for fc_size in fully_connected_layers_sizes_list:
                        results_name_suffix = f"{batch_size}_{fc_size}_{fc_size}"

                        # Prepare model
                        print("# Prepare model #")
                        vgg16 = ConvNetVGG16(input_shape)
                        vgg16_model = vgg16.prepare_convnet_vgg16(fc1 = fc_size, fc2 = fc_size)

                        # Compile and fit model
                        print("# Compile and fit model #")
                        vgg16_model_trained, vgg16_model_info, training_time = vgg16.compile_and_fit_model(vgg16_model, 
                                                                                                            train_data,
                                                                                                            validation_data,
                                                                                                            batch_size,
                                                                                                            epochs)
                        vgg16_model_info_history = vgg16_model_info.history
                        
                        # Plot training results
                        print("# Plot training results #")
                        vgg16.plot_results(vgg16_model_info_history, metrics, plots_root_directory, results_name_suffix)

                        # Save training results
                        print ("# Save training results #")
                        training_results_path = f"{training_results_root_directory}training_results_{results_name_suffix}"
                        vgg16.save_training_results_csv(vgg16_model_info_history, training_results_path)

                        # Save model
                        print("# Save model #")
                        model_path = f"{models_root_directory}vgg16_{results_name_suffix}.h5"
                        vgg16_model_trained.save(model_path)

                        # Test model on test data
                        print("# Test model on test data #")
                        test_prediction_results = vgg16_model_trained.evaluate(test_data)

                        test_labels_pred = np.array([])
                        test_labels = np.array([])

                        for test_data_batch, test_labels_batch in test_data:
                            test_predictions_batch = vgg16_model_trained.predict(test_data_batch)
                            test_labels_pred_batch = np.argmax(test_predictions_batch, axis = 1)
                            test_labels_pred = np.append(test_labels_pred, test_labels_pred_batch)
                            test_labels = np.append(test_labels, test_labels_batch)
                        
                        test_confusion_matrix = confusion_matrix(test_labels, test_labels_pred).ravel()

                        # Prepare dictionary with model results
                        print("# Prepare dictionary with model results #")
                        results_dict = {
                            "generator_type": [generator_type],
                            "images_per_class_num": [images_per_class_num],
                            "images_size": [images_size],
                            "batch_size": [batch_size],
                            "fc1": [fc_size],
                            "fc2": [fc_size],
                            "training_time": [training_time]
                        }

                        for key, value in vgg16_model_info_history.items():
                            prefix = "" if key.startswith("val") else "train_"
                            results_dict[f"{prefix}{key}_last_epoch"] = value[-1]
                            results_dict[f"{prefix}{key}_avg5"] = np.mean(value[-5:])
                            results_dict[f"{prefix}{key}_avg10"] = np.mean(value[-10:])
                        
                        for idx, metric in enumerate(["loss"] + metrics):
                            results_dict[f"test_{metric}"] = test_prediction_results[idx]
                        
                        for idx, col_name in enumerate(test_confusion_matrix_col_names):
                            results_dict[col_name] = test_confusion_matrix[idx]
                        
                        # Add model results to results file
                        print("# Add model results to results file #")
                        results_df_new_row = pd.DataFrame(results_dict)
                        results_df_new_row.to_csv(results_path, sep = ";", index = False, header = not results_file_exists, mode = "a")

                        if not results_file_exists:
                            results_file_exists = True

                        # Run garbage collector collection
                        gc.collect()