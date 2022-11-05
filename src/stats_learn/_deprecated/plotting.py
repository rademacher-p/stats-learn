# import numpy as np


# def format_math_tex(s):
#     return f"${s.strip('$')}$"


# def box_grid(lims, n=100, endpoint=True):
#     lims = np.array(lims)
#
#     # if endpoint:
#     #     n += 1
#
#     if lims.shape == (2,):
#         return np.linspace(*lims, n, endpoint=endpoint)
#     elif lims.ndim == 2 and lims.shape[-1] == 2:
#         x_dim = [np.linspace(*lims_i, n, endpoint=endpoint) for lims_i in lims]
#         # return mesh_grid(*x_dim)
#         return np.stack(np.meshgrid(*x_dim, indexing='ij'), axis=-1)
#     else:
#         raise ValueError("Shape must be (2,) or (*, 2)")
#
#     # if not (lims[..., 0] <= lims[..., 1]).all():
#     #     raise ValueError("Upper values must meet or exceed lower values.")


# def mesh_grid(*args):
#     return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)


# def main():
#     # lims = [0, 2]
#     lims = [[0, 1], [0, 2]]
#     n = 3
#     g = box_grid(lims, n)
#     print(g)
#     print(g.shape)
#
#     # args = ([0, 1, 2, 3], [4, 5, 6])
#     # m = mesh_grid(*args)
#     # print(m)
#     # print(m.shape)
#
#
# if __name__ == '__main__':
#     main()
