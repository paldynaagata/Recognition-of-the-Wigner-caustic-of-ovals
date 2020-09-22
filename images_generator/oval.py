import numpy as np
import math

from scipy import misc


class Oval:
    """
    Class generating oval for given sin and cos parameters
    """

    def __init__(self, sin_params, cos_params, t = np.linspace(0, 2 * math.pi, 1024)):
        self.t = t
        self.bias = 0
        self.sin_params = sin_params
        self.cos_params = cos_params
        self.bias = self._calculate_bias()


    def parameterization(self, shift = 0):
        t = self.t + shift
        curve_t = self._support_function(t)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        curve_dx = misc.derivative(self._support_function, t, dx = 1e-6, n = 1)
        param0 = curve_t * cos_t - curve_dx * sin_t
        param1 = curve_t * sin_t + curve_dx * cos_t
        return (param0, param1)


    def _condition_function(self):
        return self._support_function(self.t) + misc.derivative(self._support_function, self.t, dx = 1e-6, n = 2)


    def _calculate_bias(self):
        max_val = np.max(self._condition_function())
        return 1.15 * max_val


    def _generate_fourier_series(self, t):
        equation = self.bias
        params_len = len(self.sin_params)

        for i in range(params_len):
            arg = (i + 1) * t
            equation += self.sin_params[i] * np.sin(arg) + self.cos_params[i] * np.cos(arg)
        
        return equation


    def _support_function(self, t):
        return self._generate_fourier_series(t)