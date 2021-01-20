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

#@vectorize([float64(float64, float64, float64)])
#def get_q_jit(epsilon_s, gamma, gamma_e):
#    q =  (epsilon_s / gammma)/ (gammma_e * (1 - epsilon_s / gammma))
#    return q

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

@jit
def cos_psi_jit(mu_s, mu, phi):
    """compute the angle between the blob (with zenith mu_s) and a photon with
    zenith and azimuth (mu, phi). The system is symmetric in azimuth for the
    electron phi_s = 0, Eq. 8 in [Finke2016]_."""
    term_1 = mu * mu_s
    term_2 = np.sqrt(1 - np.power(mu, 2)) * np.sqrt(1 - np.power(mu_s, 2))
    term_3 = np.cos(phi)
    return term_1 + term_2 * term_3

@jit
def get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi):
    """minimum Lorentz factor for Compton integration,
    Eq. 29 in [Dermer2009]_, Eq. 38 in [Finke2016]_."""
    sqrt_term = np.sqrt(1 + 2 / (epsilon * epsilon_s * (1 - cos_psi_jit(mu_s, mu, phi))))
    return epsilon_s / 2 * (1 + sqrt_term)

@vectorize([float64(float64, float64)])
def get_y(epsilon_s, gamma):
    y = 1 - epsilon_s / gamma
    return y

@vectorize([float64(float64, float64, float64, float64)])
def get_y_1(epsilon_s, gamma, epsilon_bar, y):
    y_1 = -(2 * epsilon_s) / (gamma * epsilon_bar * y)
    return y_1

@vectorize([float64(float64, float64, float64, float64)])
def get_y_2(epsilon_s, gamma, epsilon_bar, y):
    y_2 =  np.power(epsilon_s, 2) / np.power(gamma * epsilon_bar * y, 2)
    return y_2

@vectorize([float64(float64, float64, float64)])
def get_values(y, y_1, y_2):
    values = y + 1 / y + y_1 + y_2
    return values

def compton_kernel_jit(gamma, epsilon_s, epsilon, mu_s, mu, phi):
    epsilon_bar = gamma * epsilon * (1 - cos_psi_jit(mu_s, mu, phi))
    y = get_y(epsilon_s, gamma)
    y_1 = get_y_1(epsilon_s, gamma, epsilon_bar, y)
    y_2 = get_y_2(epsilon_s, gamma, epsilon_bar, y)
    y_total = get_values(y, y_1, y_2)
    gamma_min = get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi)
    values = np.where(gamma >= gamma_min, y_total, 0)
    return values
