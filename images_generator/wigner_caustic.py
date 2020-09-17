import numpy as np
import math

from oval import Oval
from roots_finder import RootsFinder


class WignerCaustic:
    def __init__(self, oval: Oval):
        self.oval = oval


    def _wigner_caustic_i(self, parameterization_t, parameterization_t_pi, idx):
        return (parameterization_t[idx] + parameterization_t_pi[idx]) / 2


    def wigner_caustic(self):
        parameterization_t = self.oval.parameterization()
        parameterization_t_pi = self.oval.parameterization(math.pi)
        return (self._wigner_caustic_i(parameterization_t, parameterization_t_pi, 0), self._wigner_caustic_i(parameterization_t, parameterization_t_pi, 1))
    

    def _spikes_condition_function(self, t):
        params_len = len(self.oval.sin_params)
        equation = 0

        if params_len > 0:
            for i in range(params_len):
                if i % 2 == 0:
                    equation = equation + (1 - (i + 1) ** 2) * (self.oval.sin_params[i] * np.sin((i+1) * t) + self.oval.cos_params[i] * np.cos((i+1) * t))
        
        return 2 * equation


    def get_number_of_spikes(self):
        roots_finder = RootsFinder(self._spikes_condition_function)
        spikes = roots_finder.naive_global_newton(0, math.pi, 10)
        return len(spikes)