import matplotlib.pyplot as plt

from oval import Oval
from wigner_caustic import WignerCaustic


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
    plt.savefig(f"{root}{img_path}{curve_type}{img_num:03d}.png")


if __name__ == "__main__":
    oval = Oval()
    oval_parameterization = oval.parameterization()
    wc = WignerCaustic(oval)
    wc_parameterization = wc.wigner_caustic()

    plot_curve(oval_parameterization, "oval", 1)
    plot_curve(wc_parameterization, "wc", 1)