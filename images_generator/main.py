import matplotlib.pyplot as plt

from oval import Oval
from wigner_caustic import WignerCaustic
from random import randint


def plot_curve(curve, curve_type, img_num, img_size = 64):
    root = "./../../images/"

    if curve_type == "oval":
        my_dpi = 120
        plt.figure(figsize = (img_size / my_dpi, img_size / my_dpi), dpi = my_dpi)
        img_path = f"ovals/{img_size:d}x{img_size:d}/"
    elif curve_type == "wc":
        plt.figure(figsize = (8, 8))
        img_path = "wigner_caustics/"
    
    plt.plot(curve[0], curve[1], c = "black")
    plt.axis("off")
    plt.savefig(f"{root}{img_path}{curve_type}{img_num:04d}.png")
    plt.close()


if __name__ == "__main__":
    oval_idx = 0

    for spikes_num in range(3, 8, 2):
        bias = 200 * spikes_num

        for i in range(1, 21):
            sin_params = []
            cos_params = []

            for j in range(spikes_num + 2):
                rand_sin_param = randint(-3, 3)
                sin_params.append(rand_sin_param)
                rand_cos_param = randint(-3, 3)
                cos_params.append(rand_cos_param)

            sin_params[spikes_num - 1] = randint(10, 15)
            cos_params[spikes_num - 1] = randint(10, 15)

            oval_idx += 1

            print(f"### Oval no {oval_idx} ### Num of spikes {spikes_num} ### sin_params {sin_params} ### cos_params {cos_params} ###")

            oval = Oval(bias, sin_params, cos_params)
            oval_parameterization = oval.parameterization()
            wc = WignerCaustic(oval)
            wc_parameterization = wc.wigner_caustic()

            plot_curve(oval_parameterization, "oval", oval_idx)
            plot_curve(oval_parameterization, "oval", oval_idx, img_size = 128)
            plot_curve(wc_parameterization, "wc", oval_idx)
    