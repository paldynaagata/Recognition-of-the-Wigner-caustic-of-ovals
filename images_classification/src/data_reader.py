from tensorflow import keras


class DataReader:
    def __init__(self, images_dir, class_names, validation_split, subset_name, seed, image_size, batch_size):
        self.images_dir = images_dir
        self.class_names = class_names
        self.validation_split = validation_split
        self.subset_name = subset_name
        self.seed = seed
        self.image_size = image_size
        self.batch_size = batch_size
    

    def read_data(self):
        dataset = keras.preprocessing.image_dataset_from_directory(
            self.images_dir,
            class_names = self.class_names,
            validation_split = self.validation_split,
            subset = self.subset_name,
            seed = self.seed,
            image_size = self.image_size,
            batch_size = self.batch_size
        )

        return dataset