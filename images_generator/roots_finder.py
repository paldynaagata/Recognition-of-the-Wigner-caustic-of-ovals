from scipy import misc


class RootsFinder:
    def __init__(self, f):
        self.f = f


    def _newton_method(self, x_i):
        x_j = x_i - self.f(x_i) / misc.derivative(self.f, x_i, dx = 1e-6, n = 1)
        if abs(self.f(x_j)) < 0.01:
            return x_j
        else:
            return self._newton_method(x_j)


    def newton_method(self, x_0):
        return self._newton_method(x_0)


    def naive_global_newton(self, a, b, n):
        roots = []
        step = (b - a) / n
        x_0 = a
        root = 0

        while x_0 < b:
            new_root = self.newton_method(x_0)
            if x_0 == a or abs(root - new_root) > 0.01:
                roots.append(new_root)
                root = new_root
            x_0 += step
        
        return roots