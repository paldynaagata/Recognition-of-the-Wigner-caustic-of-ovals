import numpy as np
import math

from scipy import misc


class Oval:
    def __init__(self, t = np.linspace(0, 2 * math.pi, 1024)):
        self.t = t


    def _support_function(self, t):
        return 80 + 4 * np.sin(3 * t) - np.sin(5 * t) - np.cos(5 * t)


    def parameterization(self, shift = 0):
        t = self.t + shift
        curve_t = self._support_function(t)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        curve_dx = misc.derivative(self._support_function, t, dx = 1e-6, n = 1)
        return (curve_t * cos_t - curve_dx * sin_t, curve_t * sin_t + curve_dx * cos_t)