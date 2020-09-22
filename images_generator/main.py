from images_generator import ImagesGenerator


if __name__ == "__main__":
    generator_type = 1
    root_directory = "./../../images/v10/"
    images_per_type_num = 5

    generator = ImagesGenerator(generator_type, root_directory, images_per_type_num)
    generator.generate_images()