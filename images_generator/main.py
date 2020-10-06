import time

from images_generator import ImagesGenerator


if __name__ == "__main__":
    start = time.time()

    generators_dict = {
        "v1": {
            "type": 1,
            "images_per_type_divisor": 3
        },
        "v2": {
            "type": 2,
            "images_per_type_divisor": 9
        }
    }

    images_per_class_num_dict = {
        "train": [90, 270, 900],
        "test": [90]
    }

    images_sizes_list = [64 * 2 ** i for i in range(3)]

    for key, value in generators_dict.items():
        generator_type = value["type"]
        root_directory = f"./../../images/{key}/"

        for set_name, images_per_class_num_list in images_per_class_num_dict.items():
            set_directory = root_directory + f"{set_name}/"

            for images_per_class_num in images_per_class_num_list:
                if set_name == "train":
                    directory = set_directory + f"{images_per_class_num:d}_per_class/"
                elif set_name == "test":
                    directory = set_directory
                
                images_per_type_num = int(images_per_class_num / value["images_per_type_divisor"])
                generator = ImagesGenerator(generator_type, directory, images_per_type_num, images_sizes_list)
                generator.generate_images()
    
    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    hours = minutes / 60

    print(f"Generation of images took {seconds:0.2f} seconds == {minutes:0.2f} minutes == {hours:0.2f} .")