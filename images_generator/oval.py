import numpy as np
import math

from scipy import misc


class Oval:
    def __init__(self, sin_params, cos_params, t = np.linspace(0, 2 * math.pi, 1024)):
        self.t = t
        self.bias = 0
        self.sin_params = sin_params
        self.cos_params = cos_params
        self.bias = self._calculate_bias()


    def _condition_function(self, t, shift = 0):
        t += shift
        return self._support_function(t) + misc.derivative(self._support_function, t, dx = 1e-6, n = 2)
    

    def _calculate_bias(self):
        max_val = np.max(self._condition_function(self.t))
        return 1.15 * max_val
    

    def _generate_fourier_series(self, t):
        equation = self.bias
        sin_params_len = len(self.sin_params)
        cos_params_len = len(self.cos_params)

        if sin_params_len > 0:
            for i in range(sin_params_len):
                equation = equation + self.sin_params[i] * np.sin((i+1) * t)

        if cos_params_len > 0:
            for i in range(cos_params_len):
                equation = equation + self.cos_params[i] * np.cos((i+1) * t)
        
        return equation


    def _support_function(self, t):
        return self._generate_fourier_series(t)


    def parameterization(self, shift = 0):
        t = self.t + shift
        curve_t = self._support_function(t)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        curve_dx = misc.derivative(self._support_function, t, dx = 1e-6, n = 1)
        return (curve_t * cos_t - curve_dx * sin_t, curve_t * sin_t + curve_dx * cos_t)