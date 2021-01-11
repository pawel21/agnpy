import numpy as np
import astropy.units as u
from astropy.constants import e, h, c, m_e, G
from agnpy.utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c
from agnpy.utils.math import axes_reshaper, gamma_to_integrate
from numba import float64, jit, vectorize

__all__ = ["SynchrotronJit"]

e = e.gauss

@jit(nopython=True)
def fun_to_integrand(x, B_cgs, epsilon, gamma, N_e):
    P = single_electron_synch_power_x_jit(x, B_cgs, epsilon, gamma)
    return N_e*P

def calc_x(B_cgs, epsilon, gamma):
    """ratio of the frequency to the critical synchrotron frequency from
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units"""
    x = (
        4
        * np.pi
        * epsilon
        * np.power(m_e, 2)
        * np.power(c, 3)
        / (3 * e * B_cgs * h * np.power(gamma, 2))
    )
    return x.to_value("")

@jit(nopython=True)
def single_electron_synch_power_x_jit(x, B_cgs, epsilon, gamma):
    """angle-averaged synchrotron power for a single electron,
    to be folded with the electron distribution
    """
    prefactor = np.sqrt(3) * np.power(e, 3) * B_cgs / h
    return prefactor * R(x)

@vectorize([float64(float64)])
def R(x):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)

def single_electron_synch_power(B_cgs, epsilon, gamma):
    """angle-averaged synchrotron power for a single electron,
    to be folded with the electron distribution
    """
    x = calc_x(B_cgs, epsilon, gamma)
    prefactor = np.sqrt(3) * np.power(e, 3) * B_cgs / h
    return prefactor * R(x)

def tau_to_attenuation(tau):
    """Converts the synchrotron self-absorption optical depth to an attenuation
    Eq. 7.122 in [DermerMenon2009]_."""
    u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1, 3 * u / tau)


class SynchrotronJit:
    """Class for synchrotron radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
        emitting region and electron distribution
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
        The absorption factor will be taken into account in
        :func:`~agnpy.synchrotron.Synchrotron.com_sed_emissivity`, in order to be
        propagated to :func:`~agnpy.synchrotron.Synchrotron.sed_luminosity` and
        :func:`~agnpy.synchrotron.Synchrotron.sed_flux`.
    integrator : (`~agnpy.math.utils.trapz_loglog`, `~numpy.trapz`)
        function to be used for the integration
	"""

    def __init__(self, blob, ssa=False, integrator=np.trapz):
        self.blob = blob
        self.ssa = ssa
        self.integrator = integrator

    @staticmethod
    def evaluate_tau_ssa(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
    ):
        """Computes the syncrotron self-absorption opacity for a general set
        of model parameters, see
        :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
        for parameters defintion.
        Eq. before 7.122 in [DermerMenon2009]_."""
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D)
        B_cgs = B_to_cgs(B)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        SSA_integrand = n_e.evaluate_SSA_integrand(_gamma, *args)
        integrand = SSA_integrand * single_electron_synch_power(B_cgs, _epsilon, _gamma)
        integral = integrator(integrand, gamma, axis=0)
        prefactor_k_epsilon = (
            -1 / (8 * np.pi * m_e * np.power(epsilon, 2)) * np.power(lambda_c / c, 3)
        )
        k_epsilon = (prefactor_k_epsilon * integral).to("cm-1")
        return (2 * k_epsilon * R_b).to_value("")

    @staticmethod
    def evaluate_sed_flux(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_e,
        *args,
        ssa=False,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
    ):
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D)
        B_cgs = B_to_cgs(B)
        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma, *args)
        # fold the electron distribution with the synchrotron power
        # integrand = N_e * single_electron_synch_power
        x = np.float64(calc_x(B_cgs, _epsilon, _gamma))
        integrand=fun_to_integrand(x, B_cgs, _epsilon, _gamma, N_e)
        #print(integrand*(u.Fr**3*u.g**0.5)/(u.cm**0.5*u.J*u.s**2))
        integrand *= (u.Fr**3*u.g**0.5)/(u.cm**0.5*u.J*u.s**2)
        emissivity = integrator(integrand, gamma, axis=0)
        prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(d_L, 2))
        sed = (prefactor * epsilon * emissivity).to("erg cm-2 s-1")

        if ssa:
            tau = SynchrotronJit.evaluate_tau_ssa(
                nu,
                z,
                d_L,
                delta_D,
                B,
                R_b,
                n_e,
                *args,
                integrator=integrator,
                gamma=gamma,
            )
            attenuation = tau_to_attenuation(tau)
            sed *= attenuation

        return sed
