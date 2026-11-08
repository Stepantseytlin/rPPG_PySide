import numpy as np
from scipy.interpolate import PchipInterpolator
from tests.NaNsCheck import CheckNaNs
def pchip_first_derivative_coefficients(t, unwPha):
    t = np.asarray(t)
    unwPha = np.asarray(unwPha)

    axis = unwPha.ndim - 1
    p = PchipInterpolator(t, unwPha, axis=axis)

    cof = np.moveaxis(p.c, 0, -1)
    cof = np.moveaxis(cof, 0, axis)

    d_unwPha = np.take(cof, 2, axis=-1)
    d_unwPha = np.concatenate(
        (d_unwPha, np.take(d_unwPha, [-1], axis=axis)),
        axis=axis,
    )

    return d_unwPha, cof, p

# def pchip_first_derivative_coefficients(t, unwPha):
#     #CheckNaNs(unwPha,"PCHIP")
#     p = PchipInterpolator(t, unwPha,axis=1)
#     cof = p.c.T
    
#     # коэффициент при (x - t_i), то есть производная в левом конце интервала
#     d_unwPha = cof[:, 2]
#     while len(d_unwPha) < t.shape[0]:
#         d_unwPha = np.concatenate((d_unwPha,np.asarray(d_unwPha[-1:])))
#     return d_unwPha, cof, p