import matplotlib.pyplot as plt
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
    

    def _plot_results(self, model_info, metric):
        plt.plot(model_info.history[metric])
        plt.plot(model_info.history[f"val_{metric}"])
        plt.title(f"Model {metric}")
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.legend(["train", "validation"])
        # TO DO: improve path for savefig
        plt.savefig(f"./results/plots/{metric}.png")
        plt.close()


    def plot_results(self, model_info, metrics):
        metrics = ["loss"] + metrics
        for metric in metrics:
            self._plot_results(model_info, metric)