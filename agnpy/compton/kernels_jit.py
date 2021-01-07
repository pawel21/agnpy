# integration kernels for inverse Compton evaluation
import numpy as np
from numba import float64, jit, vectorize
from ..utils.math import log
from ..utils.geometry import cos_psi

__all__ = ["F_c_jit", "isotropic_kernel_jit"]

@vectorize([float64(float64, float64)])
def F_c_jit(q, gamma_e):
    """isotropic Compton kernel, Eq. 6.75 in [DermerMenon2009]_, Eq. 10 in [Finke2008]_"""
    term_1 = 2 * q * np.log(q)
    term_2 = (1 + 2 * q) * (1 - q)
    term_3 = 1 / 2 * np.power(gamma_e * q, 2) / (1 + gamma_e * q) * (1 - q)
    return term_1 + term_2 + term_3


def isotropic_kernel_jit(gamma, epsilon, epsilon_s):
    """Compton kernel for isotropic nonthermal electrons scattering photons of
    an isotropic external radiation field.
    Integrand of Eq. 6.74 in [DermerMenon2009]_.
    Parameters
    ----------
    gamma : :class:`~numpy.ndarray`
        Lorentz factors of the electrons distribution
    epsilon : :class:`~numpy.ndarray`
        dimensionless energies (in electron rest mass units) of the target photons
    epsilon_s : :class:`~numpy.ndarray`
        dimensionless energies (in electron rest mass units) of the scattered photons
    """
    gamma_e = 4 * gamma * epsilon
    q = (epsilon_s / gamma)/ (gamma_e * (1 - epsilon_s / gamma))
    q_min = 1 / (4 * np.power(gamma, 2))
    return np.where((q_min <= q) * (q <= 1), F_c_jit(q, gamma_e), 0)
