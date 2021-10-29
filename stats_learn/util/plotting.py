import numpy as np


# def format_math_tex(s):
#     return f"${s.strip('$')}$"


def box_grid(lims, n=100, endpoint=True):
    lims = np.array(lims)

    # if endpoint:
    #     n += 1

    if lims.shape == (2,):
        return np.linspace(*lims, n, endpoint=endpoint)
    elif lims.ndim == 2 and lims.shape[-1] == 2:
        x_dim = [np.linspace(*lims_i, n, endpoint=endpoint) for lims_i in lims]
        # return np.stack(np.meshgrid(*x_dim), axis=-1)
        return mesh_grid(*x_dim)
    else:
        raise ValueError("Shape must be (2,) or (*, 2)")

    # if not (lims[..., 0] <= lims[..., 1]).all():
    #     raise ValueError("Upper values must meet or exceed lower values.")


def mesh_grid(*args):
    # return np.stack(np.meshgrid(*args[::-1])[::-1], axis=-1)
    return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)
