import numpy as np
import matplotlib.pyplot as plt
import csv
import time


class ConvNet:
    def __init__(self, input_shape, num_classes = 3):
        self.input_shape = input_shape
        self.num_classes = num_classes
    

    def compile_and_fit_model(self, 
                                model, 
                                train_data,
                                validation_data,
                                batch_size,
                                epochs,
                                optimizer = "adam", 
                                loss = "sparse_categorical_crossentropy", 
                                metrics = ["sparse_categorical_accuracy"]):
        model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )

        start = time.time()
        model_info  = model.fit(
            train_data, 
            batch_size = batch_size,
            epochs = epochs, 
            validation_data = validation_data
        )
        end = time.time()

        print(f"Model took {end - start:0.2f} seconds to train")

        return model, model_info
    

    def _plot_results(self, model_info_history, metric, n_epochs, xticks_step, plots_path, plots_name_suffix, extensions):
        metric_name = metric.replace("_", " ")
        plt.plot(model_info_history[metric])
        plt.plot(model_info_history[f"val_{metric}"])
        plt.xticks(np.arange(0, n_epochs + xticks_step, step = xticks_step))
        plt.title(f"Model {metric_name}")
        plt.xlabel("epoch")
        plt.ylabel(metric_name)
        plt.legend(["train", "validation"])
        for extension in extensions:
            plt.savefig(f"{plots_path}{metric}_{plots_name_suffix}.{extension}")
        plt.close()


    def plot_results(self, model_info_history, metrics, plots_path, plots_name_suffix, extensions = ["png", "pdf"]):
        metrics = ["loss"] + metrics
        n_epochs = len(model_info_history["loss"])
        xticks_step = 1 if n_epochs < 10 else n_epochs / 10
        for metric in metrics:
            self._plot_results(model_info_history, metric, n_epochs, xticks_step, plots_path, plots_name_suffix, extensions)
    

    def save_training_results_csv(self, model_info_history, training_results_path):
        training_results_path = f"{training_results_path}.csv"
        model_info_history["epoch"] = list(range(len(model_info_history["loss"])))
        keys = sorted(model_info_history.keys())
        with open(training_results_path, "w", newline = "") as outfile:
            writer = csv.writer(outfile, delimiter = ";")
            writer.writerow(keys)
            writer.writerows(zip(*[model_info_history[key] for key in keys]))