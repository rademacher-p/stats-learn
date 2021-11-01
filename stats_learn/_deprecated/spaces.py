class Categorical(Finite):
    def __init__(self, values):
        self.values = np.sort(np.array(values, dtype='U').flatten())
        super().__init__((), self.values.dtype)

        if len(self.values) != len(np.unique(self.values)):
            raise ValueError("Input 'values' must have unique values")

    def __repr__(self):
        return f"Categorical({self.values})"

    def __eq__(self, other):
        if isinstance(other, Categorical):
            return (self.values == other.values).all()
        return NotImplemented

    def __contains__(self, item):
        return item in self.values

    def _minimize(self, f):
        i_opt = np.argmin(f(self.values))
        return self.values[i_opt]

    def integrate(self, f):
        return sum(f(self.values))

    def set_x_plot(self, x=None):
        if x is None:
            self.x_plt = self.values
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)
        if len(set_shape) != 1:
            raise ValueError("Input 'x' must be 1-D")

        return ax.stem(x, y, use_line_collection=True, label=label)


class Grid(Finite):     # FIXME: 1-D special?
    def __new__(cls, *vecs):
        if len(vecs) == 1:
            return super().__new__(Grid1D)
        else:
            return super().__new__(cls)

    def __init__(self, *vecs):
        # self.vecs = list(map(lambda v: np.sort(np.array(v, dtype=np.float).flatten()), vecs))
        self.vecs = tuple(np.sort(np.array(list(vec), dtype=np.float).flatten()) for vec in vecs)
        super().__init__((len(self.vecs),), np.float)

        self.set_shape = tuple(vec.size for vec in self.vecs)
        self.set_size = math.prod(self.set_shape)
        self.set_ndim = len(self.set_shape)

    def __repr__(self):
        return f"Grid({self.vecs})"

    def __eq__(self, other):
        if isinstance(other, Grid):
            return self.vecs == other.vecs
        return NotImplemented

    def __contains__(self, item):
        return all(x_i in vec for x_i, vec in zip(item, self.vecs))

    def _minimize(self, f):
        def _idx_to_vec(idx):
            return [vec[i] for i, vec in zip(idx, self.vecs)]

        ranges = tuple(slice(size_) for size_ in self.set_shape)
        i_opt = int(optimize.brute(lambda idx: f(_idx_to_vec(idx)), ranges))
        return _idx_to_vec(i_opt)

    def integrate(self, f):
        y = f(plotting.mesh_grid(*self.vecs))
        return y.reshape(self.set_size, self.shape).sum(0)

    def set_x_plot(self, x=None):
        if x is None:
            self.x_plt = plotting.mesh_grid(*self.vecs)
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)

        set_ndim = len(set_shape)
        if set_ndim == 1 and self.shape == ():
            # return ax.stem(x, y, use_line_collection=True, label=label)
            return ax.plot(x, y, '.', label=label)

        elif set_ndim == 2 and self.shape == (2,):
            # return ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(), 0, 1, 1, y.flatten(), shade=True)
            return ax.plot(x[..., 0].flatten(), x[..., 1].flatten(), y.flatten(), marker='.', linestyle='', label=label)

        elif set_ndim == 3 and self.shape == (3,):
            plt_data = ax.scatter(x[..., 0], x[..., 1], x[..., 2], s=15, c=y, label=label)
            c_bar = plt.colorbar(plt_data, ax=ax)
            c_bar.set_label('$y$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


class Grid1D(Grid):     # FIXME: DRY from categorical?
    def __init__(self, vec):
        self.vec = np.sort(np.array(list(vec), dtype=np.float).flatten())
        super().__init__((len(self.vecs),), np.float)

        self.set_shape = tuple(vec.size for vec in self.vecs)
        self.set_size = math.prod(self.set_shape)
        self.set_ndim = len(self.set_shape)

    def __repr__(self):
        return f"Grid1D({self.vec})"

    def __eq__(self, other):
        if isinstance(other, Grid1D):
            return self.vec == other.vec
        return NotImplemented

    def __contains__(self, item):
        return item in self.vec

    def _minimize(self, f):
        i_opt = np.argmin(f(self.vec))
        return self.vec[i_opt]

        # ranges = (slice(self.vec.size), )
        # i_opt = int(optimize.brute(lambda idx: f(self.vec[idx]), ranges))
        # return self.vec[i_opt]

    def integrate(self, f):
        return sum(f(self.vec))
        # y = f(plotting.mesh_grid(*self.vecs))
        # return y.reshape(self.set_size, self.shape).sum(0)

    def set_x_plot(self, x=None):
        if x is None:
            self.x_plt = self.vec
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)
        if len(set_shape) != 1:
            raise ValueError("Input 'x' must be 1-D")

        return ax.plot(x, y, '.', label=label)
