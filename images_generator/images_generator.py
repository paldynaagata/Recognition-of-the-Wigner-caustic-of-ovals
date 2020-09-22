import matplotlib.pyplot as plt
import random
import os
import uuid

from oval import Oval
from wigner_caustic import WignerCaustic


class ImagesGenerator:
    def __init__(self, generator_type, root_directory, images_per_type_num):
        self.generator_type = generator_type
        self.root_directory = root_directory
        self.images_per_type_num = images_per_type_num


    def plot_curve(self, curve, curve_type, cusps_num, img_num, img_size = 64):
        if curve_type == "oval":
            my_dpi = 120
            plt.figure(figsize = (img_size / my_dpi, img_size / my_dpi), dpi = my_dpi)
            img_path = f"ovals/{img_size:d}x{img_size:d}/class_{cusps_num:d}/"
        elif curve_type == "wc":
            plt.figure(figsize = (8, 8))
            img_path = "wigner_caustics/"
        
        directory = f"{self.root_directory}{img_path}"

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.plot(curve[0], curve[1], c = "black")
        plt.axis("off")
        plt.savefig(f"{directory}{cusps_num:d}_{curve_type}{img_num:04d}.png")
        plt.close()


    def generate_images(self):
        if not os.path.exists(self.root_directory):
            os.makedirs(self.root_directory)

        seed = hash(uuid.uuid4())
        random.seed(seed)
        self._write_log(f"### Seed {seed} ###\n\n")
        
        oval_idx = 0
        consistent_cusps_num_counter = 0

        for cusps_num in range(3, 8, 2):
            for random_range in range(5, 50, 20):
                for _ in range(self.images_per_type_num):
                    sin_params, cos_params = self._generate_parameters(cusps_num, random_range)

                    start = cusps_num if self.generator_type == 2 else cusps_num + 2
                    end = cusps_num + 3

                    for param_idx in range(start, end):
                        oval_idx += 1

                        oval = Oval(sin_params[0:param_idx], cos_params[0:param_idx])
                        oval_parameterization = oval.parameterization()
                        wc = WignerCaustic(oval)
                        wc_parameterization = wc.wigner_caustic()
                        real_cusps_num = wc.get_number_of_cusps()

                        consistent_cusps_num_counter += 1 if real_cusps_num == cusps_num else 0

                        log_text = f"### Oval no {oval_idx} ### Cusps num {cusps_num} ### Real cusps num {real_cusps_num} ###\n### sin_params {sin_params[0:param_idx]} ###\n### cos_params {cos_params[0:param_idx]} ###\n### bias {oval.bias} ###\n"
                        print(log_text)
                        self._write_log(log_text)

                        self.plot_curve(oval_parameterization, "oval", cusps_num, oval_idx)
                        self.plot_curve(oval_parameterization, "oval", cusps_num, oval_idx, img_size = 128)
                        self.plot_curve(wc_parameterization, "wc", cusps_num, oval_idx)

        print(f"Percentage of consistent cusps num: {consistent_cusps_num_counter/oval_idx:.2%}")


    def _generate_parameters(self, cusps_num, random_range):
        sin_params = []
        cos_params = []

        for j in range(cusps_num + 2):
            if j % 2:
                rand_sin_param = random.uniform(-random_range, random_range)
                rand_cos_param = random.uniform(-random_range, random_range)
            else:
                rand_sin_param = random.uniform(-5, 5)
                rand_cos_param = random.uniform(-5, 5)
                
            sin_params.append(rand_sin_param)
            cos_params.append(rand_cos_param)

        sin_params[cusps_num - 1] = random.uniform(15, 20)
        cos_params[cusps_num - 1] = random.uniform(15, 20)

        return sin_params, cos_params

    
    def _write_log(self, log_text):
        log_file_path = f"{self.root_directory}log_file.txt"
        with open(log_file_path, "a+") as log_file:
            log_file.write(f"{log_text}\n")