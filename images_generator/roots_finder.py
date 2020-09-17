from scipy import misc


class RootsFinder:
    def __init__(self, f):
        self.f = f
        self.recursions_counter = 0


    def _newton_method(self, x_i):
        x_j = x_i - self.f(x_i) / misc.derivative(self.f, x_i, dx = 1e-6, n = 1)

        if self.recursions_counter > 100:
            return None

        if abs(x_i - x_j) < 0.00001 or abs(self.f(x_j)) < 0.00001:
            return x_j
        else:
            self.recursions_counter += 1
            return self._newton_method(x_j)


    def newton_method(self, x_0):
        self.recursions_counter = 0
        return self._newton_method(x_0)


    def naive_global_newton(self, a, b, n):
        roots = []
        step = (b - a) / n
        x_0 = a

        while x_0 < b:
            new_root = self.newton_method(x_0)

            if new_root is not None:
                new_root_is_new = True

                for root in roots:
                    if abs(root - new_root) < 0.01:
                        new_root_is_new = False
                        break

                if a <= new_root < b and new_root_is_new:
                    roots.append(new_root)

            x_0 += step
        
        return roots