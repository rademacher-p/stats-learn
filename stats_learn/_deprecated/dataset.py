class DataSet(Base):
    # TODO: update with `deque` changes from CRM

    def __init__(
        self, data, space=None, iter_mode="once", shuffle_mode="never", rng=None
    ):
        """
        Model from data.

        Parameters
        ----------
        data : np.ndarray
            Structured array with fields `x` and `y`.
        space : dict of str, optional
            The data spaces. Each defaults to `Euclidean` or `FiniteGeneric` based on data type.
        iter_mode : {'once', 'repeat'}, optional
            Controls whether data can be yielded more than once using `sample`.
        shuffle_mode : {'never', 'once', 'repeat'}, optional
            Enable shuffle at instantiation only or after every pass through the dataset.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        """
        super().__init__(rng)

        self.data = data

        if space is not None:
            self._space = space
        else:
            for c in "xy":
                dtype = data.dtype[c]
                if np.issubdtype(dtype.base, np.number):
                    self._space[c] = spaces.Euclidean(dtype.shape)
                else:
                    self._space[c] = spaces.FiniteGeneric(
                        data[c], shape=dtype.shape
                    )  # TODO: check...

        self.iter_mode = iter_mode
        self.shuffle_mode = shuffle_mode

        self.idx = None
        self.restart(shuffle=(self.shuffle_mode in {"once", "repeat"}))

    def __repr__(self):
        return f"DataSet({len(self.data)})"

    def model_y_x(self, x):
        raise NotImplementedError

    @classmethod
    def from_xy(
        cls, x, y, space=None, iter_mode="once", shuffle_mode="never", rng=None
    ):
        data = np.array(
            list(zip(x, y)),
            dtype=[("x", x.dtype, x.shape[1:]), ("y", y.dtype, y.shape[1:])],
        )

        return cls(data, space, iter_mode, shuffle_mode, rng)

    @classmethod
    def from_csv(
        cls, path, y_name, space=None, iter_mode="once", shuffle_mode="never", rng=None
    ):
        df_x = pd.read_csv(path)
        df_y = df_x.pop(y_name)
        x, y = df_x.to_numpy(), df_y.to_numpy()

        return cls.from_xy(x, y, space, iter_mode, shuffle_mode, rng)

    def restart(self, shuffle=False, rng=None):
        self.idx = 0
        if shuffle:
            self.shuffle(rng)

    def shuffle(self, rng=None):
        rng = self._get_rng(rng)
        rng.shuffle(self.data)

    def _sample(self, n, rng):
        if self.idx + n > len(self.data):
            if self.iter_mode == "once":
                raise ValueError("DataSet model is exhausted.")
            elif self.iter_mode == "repeat":
                self.restart(shuffle=(self.shuffle_mode == "repeat"), rng=rng)
                # TODO: use trailing samples?

        out = self.data[self.idx : self.idx + n]
        self.idx += n
        return out
