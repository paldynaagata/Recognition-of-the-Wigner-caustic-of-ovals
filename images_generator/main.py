import matplotlib.pyplot as plt
import random
import os

from oval import Oval
from wigner_caustic import WignerCaustic


def plot_curve(curve, curve_type, spikes_num, img_num, img_size = 64):
    root = "./../../images/v02/"

    if curve_type == "oval":
        my_dpi = 120
        plt.figure(figsize = (img_size / my_dpi, img_size / my_dpi), dpi = my_dpi)
        img_path = f"ovals/{img_size:d}x{img_size:d}/class_{spikes_num:d}/"
    elif curve_type == "wc":
        plt.figure(figsize = (8, 8))
        img_path = "wigner_caustics/"
    
    directory = f"{root}{img_path}"

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.plot(curve[0], curve[1], c = "black")
    plt.axis("off")
    plt.savefig(f"{directory}{spikes_num:d}_{curve_type}{img_num:04d}.png")
    plt.close()


if __name__ == "__main__":
    for spikes_num in range(3, 8, 2):
        # bias = 300 * spikes_num
        oval_idx = 0
        images_per_class_num = 20

        for i in range(images_per_class_num):
            sin_params = []
            cos_params = []

            for j in range(spikes_num + 2):
                if j % 2:
                    rand_sin_param = random.uniform(10, 40)
                    rand_cos_param = random.uniform(10, 40)
                else:
                    rand_sin_param = random.uniform(-3, 3)
                    rand_cos_param = random.uniform(-3, 3)
                    
                sin_params.append(rand_sin_param)
                cos_params.append(rand_cos_param)

            sin_params[spikes_num - 1] = random.uniform(40, 60)
            cos_params[spikes_num - 1] = random.uniform(40, 60)

            limit = 0

            for idx in range(len(sin_params)):
                limit += ((idx + 1) ** 2 - 1) * (sin_params[idx] + cos_params[idx])
            
            bias = limit + 100
            print(f"### bias: {bias} ### limit: {limit} ###")

            oval_idx += 1

            print(f"### Num of spikes {spikes_num} ### Oval no {oval_idx} ### sin_params {sin_params} ### cos_params {cos_params} ###")

            oval = Oval(bias, sin_params, cos_params)
            oval_parameterization = oval.parameterization()
            wc = WignerCaustic(oval)
            wc_parameterization = wc.wigner_caustic()

            plot_curve(oval_parameterization, "oval", spikes_num, oval_idx)
            plot_curve(oval_parameterization, "oval", spikes_num, oval_idx, img_size = 128)
            plot_curve(wc_parameterization, "wc", spikes_num, oval_idx)
    