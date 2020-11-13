import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import os
import uuid

from oval import Oval
from wigner_caustic import WignerCaustic
from curve_type import CurveType


class ImagesGenerator:
    """
    Class for generating images of ovals and wigner caustics
    """

    def __init__(self, generator_type, root_directory, images_per_type_num, images_sizes_list):
        self.generator_type = generator_type
        self.root_directory = root_directory
        self.images_per_type_num = images_per_type_num
        self.images_sizes_list = images_sizes_list


    def plot_curve(self, curve, curve_type: CurveType, cusps_num, img_num, img_size = 64):
        if curve_type == CurveType.oval:
            my_dpi = 120
            plt.figure(figsize = (img_size / my_dpi, img_size / my_dpi), dpi = my_dpi)
            img_path = f"ovals/{img_size:d}x{img_size:d}/class_{cusps_num:d}/"
        elif curve_type == CurveType.wigner_caustic:
            plt.figure(figsize = (8, 8))
            img_path = "wigner_caustics/"
        else:
            raise ValueError("Unexpected curve_type")
        
        directory = f"{self.root_directory}{img_path}"

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.plot(curve[0], curve[1], c = "black")
        plt.axis("off")
        plt.savefig(f"{directory}{cusps_num:d}_{curve_type.name}_{img_num:04d}.png")
        plt.close()


    def generate_images(self):
        if not os.path.exists(self.root_directory):
            os.makedirs(self.root_directory)

        seed = hash(uuid.uuid4())
        random.seed(seed)
        self._write_log(f"### Seed {seed} ###\n")
        
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

                        sin_params_subset = sin_params[0:param_idx]
                        cos_params_subset = cos_params[0:param_idx]

                        oval = Oval(sin_params_subset, cos_params_subset)
                        oval_parameterization = oval.parameterization()
                        wc = WignerCaustic(oval)
                        wc_parameterization = wc.wigner_caustic()
                        real_cusps_num = wc.get_number_of_cusps()

                        log_text = f"### Oval no {oval_idx} ### Cusps num {cusps_num} ### Real cusps num {real_cusps_num} ###\n### sin_params {sin_params_subset} ###\n### cos_params {cos_params_subset} ###\n### bias {oval.bias} ###\n"
                        print(log_text)
                        self._write_log(log_text)

                        if real_cusps_num == cusps_num:
                            consistent_cusps_num_counter += 1
                        elif real_cusps_num % 2 == 0:
                            real_cusps_num -= 1

                        for img_size in self.images_sizes_list:
                            self.plot_curve(oval_parameterization, CurveType.oval, real_cusps_num, oval_idx, img_size = img_size)
                        
                        self.plot_curve(wc_parameterization, CurveType.wigner_caustic, real_cusps_num, oval_idx)

        print(f"Percentage of consistent cusps num: {consistent_cusps_num_counter/oval_idx:.2%}\n")


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