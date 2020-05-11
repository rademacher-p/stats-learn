import copy

class A:
    def __init__(self, a):
        self.a = a
        self._f = None

    # @property
    # def f(self):
    #     return self._f
    #
    # def up(self, b):
    #     def _f(x):
    #         return x + b
    #
    #     self._f = _f

    def f(self, x):
        return x + 1


out = []
for i in range(2):
    def f(x):
        return x+i
    out.append(copy.deepcopy(f))

