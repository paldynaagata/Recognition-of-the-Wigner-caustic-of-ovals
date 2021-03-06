import numpy as np
import math

from oval import Oval
from roots_finder import RootsFinder


class WignerCaustic:
    """
    Class generating wigner caustic for the given oval
    """

    def __init__(self, oval: Oval):
        self.oval = oval


    def wigner_caustic(self):
        parameterization_t = self.oval.parameterization()
        parameterization_t_pi = self.oval.parameterization(math.pi)
        wigner_caustic_0 = self._wigner_caustic_i(parameterization_t, parameterization_t_pi, 0)
        wigner_caustic_1 = self._wigner_caustic_i(parameterization_t, parameterization_t_pi, 1)
        return (wigner_caustic_0, wigner_caustic_1)


    def get_number_of_cusps(self):
        roots_finder = RootsFinder(self._cusps_condition_function)
        cusps = roots_finder.naive_global_newton(0, math.pi, 100)
        return len(cusps)


    def _wigner_caustic_i(self, parameterization_t, parameterization_t_pi, idx):
        return (parameterization_t[idx] + parameterization_t_pi[idx]) / 2


    def _cusps_condition_function(self, t):
        params_len = len(self.oval.sin_params)
        equation = 0

        for i in range(params_len):
            if i % 2 == 0:
                arg = (i + 1) * t
                sin_with_param = self.oval.sin_params[i] * np.sin(arg)
                cos_with_param = self.oval.cos_params[i] * np.cos(arg)
                equation += (1 - (i + 1) ** 2) * (sin_with_param + cos_with_param)
        
        return 2 * equation