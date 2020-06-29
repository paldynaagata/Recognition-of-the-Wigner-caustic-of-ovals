import numpy as np
import math

from oval import Oval


class WignerCaustic:
    def __init__(self, oval: Oval):
        self.oval = oval


    def _wigner_caustic_i(self, parameterization_t, parameterization_t_pi, idx):
        return (parameterization_t[idx] + parameterization_t_pi[idx]) / 2


    def wigner_caustic(self):
        parameterization_t = self.oval.parameterization()
        parameterization_t_pi = self.oval.parameterization(math.pi)
        return (self._wigner_caustic_i(parameterization_t, parameterization_t_pi, 0), self._wigner_caustic_i(parameterization_t, parameterization_t_pi, 1))