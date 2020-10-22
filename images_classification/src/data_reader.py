from tensorflow import keras


class DataReader:
    def __init__(self, 
                images_dir, 
                image_size, 
                batch_size, 
                validation_split = 0.2, 
                seed = 123456,
                class_names = ["class_3", "class_5", "class_7"]):
        self.images_dir = images_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.class_names = class_names
    

    def read_data(self, subset_name = None):
        dataset = keras.preprocessing.image_dataset_from_directory(
            self.images_dir,
            image_size = self.image_size,
            batch_size = self.batch_size,
            validation_split = self.validation_split,
            seed = self.seed,
            class_names = self.class_names,
            subset = subset_name
        )

        return dataset