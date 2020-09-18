import matplotlib.pyplot as plt
import random
import os
import uuid

from oval import Oval
from wigner_caustic import WignerCaustic


def plot_curve(root, curve, curve_type, cusps_num, img_num, img_size = 64):
    if curve_type == "oval":
        my_dpi = 120
        plt.figure(figsize = (img_size / my_dpi, img_size / my_dpi), dpi = my_dpi)
        img_path = f"ovals/{img_size:d}x{img_size:d}/class_{cusps_num:d}/"
    elif curve_type == "wc":
        plt.figure(figsize = (8, 8))
        img_path = "wigner_caustics/"
    
    directory = f"{root}{img_path}"

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.plot(curve[0], curve[1], c = "black")
    plt.axis("off")
    plt.savefig(f"{directory}{cusps_num:d}_{curve_type}{img_num:04d}.png")
    plt.close()


if __name__ == "__main__":
    root = "./../../images/v10/"

    if not os.path.exists(root):
        os.makedirs(root)

    seed = hash(uuid.uuid4())
    random.seed(seed)

    log_file_path = f"{root}log_file.txt"
    with open(log_file_path, "a+") as log_file:
        log_file.write(f"### Seed {seed} ###\n\n")
    
    images_per_type_num = 5
    oval_idx = 0
    ovals_num = 0
    consistent_cusps_num_counter = 0

    for cusps_num in range(3, 8, 2):
        for x in range(5, 50, 20):
            for i in range(images_per_type_num):
                sin_params = []
                cos_params = []

                for j in range(cusps_num + 2):
                    if j % 2:
                        rand_sin_param = random.uniform(-x, x)
                        rand_cos_param = random.uniform(-x, x)
                    else:
                        rand_sin_param = random.uniform(-5, 5)
                        rand_cos_param = random.uniform(-5, 5)
                        
                    sin_params.append(rand_sin_param)
                    cos_params.append(rand_cos_param)

                sin_params[cusps_num - 1] = random.uniform(15, 20)
                cos_params[cusps_num - 1] = random.uniform(15, 20)

                ### v11
                oval_idx += 1

                oval = Oval(sin_params, cos_params)
                oval_parameterization = oval.parameterization()
                wc = WignerCaustic(oval)
                wc_parameterization = wc.wigner_caustic()
                real_cusps_num = wc.get_number_of_cusps()

                consistent_cusps_num_counter += 1 if real_cusps_num == cusps_num else 0
                ovals_num += 1

                log_text = f"### Oval no {oval_idx} ### Cusps num {cusps_num} ### Real cusps num {real_cusps_num} ###\n### sin_params {sin_params} ###\n### cos_params {cos_params} ###\n### bias {oval.bias} ###\n"
                print(log_text)

                plot_curve(root, oval_parameterization, "oval", real_cusps_num, oval_idx)
                plot_curve(root, oval_parameterization, "oval", real_cusps_num, oval_idx, img_size = 128)
                plot_curve(root, wc_parameterization, "wc", real_cusps_num, oval_idx)
                
                with open(log_file_path, "a+") as log_file:
                    log_file.write(f"{log_text}\n")
                
                ### v12
                # for param_idx in range(cusps_num, cusps_num + 3):
                #     oval_idx += 1

                #     oval = Oval(sin_params[0:param_idx], cos_params[0:param_idx])
                #     oval_parameterization = oval.parameterization()
                #     wc = WignerCaustic(oval)
                #     wc_parameterization = wc.wigner_caustic()

                #     log_text = f"### Num of cusps {cusps_num} ### Oval no {oval_idx} ###\n### sin_params {sin_params[0:param_idx]} ###\n### cos_params {cos_params[0:param_idx]} ###\n### bias {oval.bias} ###\n"
                #     print(log_text)

                #     plot_curve(root, oval_parameterization, "oval", cusps_num, oval_idx)
                #     plot_curve(root, oval_parameterization, "oval", cusps_num, oval_idx, img_size = 128)
                #     plot_curve(root, wc_parameterization, "wc", cusps_num, oval_idx)

                #     log_file_path = f"{root}log_file.txt"
                #     with open(log_file_path, "a+") as log_file:
                #         log_file.write(f"{log_text}\n")

    print(f"Percentage of consistent cusps num: {consistent_cusps_num_counter/ovals_num:.2%}")
    